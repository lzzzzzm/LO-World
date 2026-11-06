import json
from pathlib import Path

from einops import rearrange

from mmengine.dist import (broadcast_object_list, get_rank, get_world_size,
                           is_main_process)
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from mmengine.logging import MMLogger
from mmengine.logging import print_log
from terminaltables import AsciiTable

import torch

from chamferdist import ChamferDistance
from chamferdist.chamfer import knn_points
from projects.evaluation.functional.lidar_3d_utils import (
    pretrained_pointnet, 
    point_cloud_to_histogram,
    compute_jsd_2d,
    compute_frechet_distance,
)
import numpy as np
from projects.visualizer.lidar_visualizer import LidarBEVVisualizer
from .visualization import should_visualize, visualization_save_name


@torch.no_grad()
def _kernel_sum(x, y, kernel, block_size, symmetric=False, exclude_diagonal=False):
    """Exact pairwise sum with bounded workspace and float64 accumulation."""
    total = torch.zeros((), dtype=torch.float64, device=x.device)
    for i in range(0, len(x), block_size):
        left = x[i:i + block_size]
        for j in range(i if symmetric else 0, len(y), block_size):
            right = y[j:j + block_size]
            if kernel == 'polynomial':
                values = (left @ right.T / x.shape[1] + 1).pow(3)
            elif kernel == 'rbf':
                distances = torch.cdist(
                    left, right, p=2., compute_mode='use_mm_for_euclid_dist')
                values = torch.exp(-2. * distances.square())
            else:
                raise ValueError(f'Unknown kernel: {kernel}')
            value = values.sum(dtype=torch.float64)
            if exclude_diagonal and i == j:
                value -= values.diagonal().sum(dtype=torch.float64)
            total += value * (2 if symmetric and i != j else 1)
    return float(total.item())


@torch.no_grad()
def full_squared_mmd(x, y, kernel, block_size=512):
    """Original kernels: unbiased polynomial MMD², biased RBF MMD².

    Every input sample participates. Unlike the historical Point MMD helper,
    the polynomial statistic is evaluated exactly rather than estimated by
    averaging 100 randomly selected subsets of at most 1000 samples.
    """
    if block_size <= 0 or x.ndim != 2 or y.ndim != 2:
        raise ValueError('Expected matrices and a positive block size')
    n, m = len(x), len(y)
    if min(n, m) < 2 or x.shape[1] != y.shape[1]:
        raise ValueError('MMD needs at least two samples with matching features')
    if x.device != y.device or x.dtype != y.dtype:
        raise ValueError('Input device and dtype must match')
    unbiased = kernel == 'polynomial'
    xx = _kernel_sum(x, x, kernel, block_size, True, unbiased)
    yy = _kernel_sum(y, y, kernel, block_size, True, unbiased)
    xy = _kernel_sum(x, y, kernel, block_size)
    return (xx / (n * (n - 1) if unbiased else n * n)
            + yy / (m * (m - 1) if unbiased else m * m)
            - 2 * xy / (n * m))


def pool_future_frames(frames):
    """One row per future point cloud; frame zero never enters the pool."""
    if len(frames) < 2:
        raise ValueError('Future pooling needs at least one future frame')
    n = len(frames[0])
    if n < 2 or any(len(frame) != n for frame in frames):
        raise ValueError('Every frame must contain the same valid sample count')
    return torch.cat(frames[1:], dim=0)


@torch.no_grad()
def pooled_distribution_metrics(frames, device, block_size=512, logger=None):
    """Compute four distributions on all pooled future rows, without a cap."""
    def report(message):
        if logger is not None:
            logger.info(message)

    pred_feats = pool_future_frames(frames['pred_feats']).numpy().astype(np.float64)
    targ_feats = pool_future_frames(frames['targ_feats']).numpy().astype(np.float64)
    count = len(pred_feats)
    report(f'Pooled future FID: all {count} predicted / GT features')
    fid = compute_frechet_distance(pred_feats, targ_feats)
    report('Pooled future Point MMD: exact full polynomial U-statistic')
    point_mmd = full_squared_mmd(
        torch.from_numpy(pred_feats).to(device),
        torch.from_numpy(targ_feats).to(device), 'polynomial', block_size)
    del pred_feats, targ_feats

    pred_hists = pool_future_frames(frames['pred_hists']).flatten(1)
    targ_hists = pool_future_frames(frames['targ_hists']).flatten(1)
    if len(pred_hists) != count or len(targ_hists) != count:
        raise RuntimeError('Feature/histogram pool counts differ')
    if not torch.all(pred_hists.sum(1) > 0) or not torch.all(targ_hists.sum(1) > 0):
        raise RuntimeError('Empty BEV histogram cannot be normalized')
    report('Pooled future BEV JSD: aggregate all future histograms')
    jsd = float(compute_jsd_2d(pred_hists, targ_hists))
    if torch.device(device).type == 'cuda':
        # Release temporary distributed-gather/feature allocations before the
        # two full histogram matrices are transferred to the metric device.
        torch.cuda.empty_cache()
    pred = pred_hists.to(device=device, dtype=torch.float32)
    targ = targ_hists.to(device=device, dtype=torch.float32)
    pred /= pred.sum(1, keepdim=True)
    targ /= targ.sum(1, keepdim=True)
    report('Pooled future BEV MMD: exact full RBF V-statistic')
    bev_mmd = full_squared_mmd(pred, targ, 'rbf', block_size)
    result = {
        'Point_Feature_FID': float(fid),
        'Point_Squared_MMD': float(point_mmd),
        'BEV_Feature_JSDx10^2': jsd * 1e2,
        'BEV_Feature_MMDx10^4': bev_mmd * 1e4,
        'Future_PointCloud_Count': float(count),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise RuntimeError(f'Non-finite pooled distribution metric: {result}')
    return result


@METRICS.register_module()
class LiDAR4DMetric(BaseMetric):
    """Evaluate per-frame geometry and optional point-cloud distributions.

    detail=True evaluates all future point clouds as one distribution,
    excluding frame zero. Geometry metrics remain per-frame. The legacy
    max_mmd_samples argument is accepted for config compatibility but does not
    truncate this population. cache_dir optionally saves per-frame features
    for offline recomputation.
    """
    def __init__(
        self, 
        eval_frame_number=7, 
        detail=False, 
        collect_device='cpu', 
        prefix=None, 
        vis_interval=0, 
        vis_dir='work_dirs/vis_lidar_4d', 
        max_mmd_samples=10000,
        warmup_frame_number=0,
        compute_lidar_metrics=True,
        axis_diagnostics=False,
        frame_interval=0.5,
        max_pred_points=50000,
        cache_dir=None,
        mmd_block_size=512,
        **kwargs
    ):
        if detail:
            if not compute_lidar_metrics:
                raise ValueError('detail=True requires LiDAR metrics')
            if eval_frame_number < 2:
                raise ValueError('detail=True requires at least one future frame')
        if mmd_block_size <= 0:
            raise ValueError('mmd_block_size must be positive')
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.mmd_block_size = mmd_block_size
        self.max_pred_points = max_pred_points
        if max_pred_points is not None and max_pred_points <= 0:
            raise ValueError('max_pred_points must be positive or None.')
        self.DATASET_MAX_DEPTH = 80.0
        self.eval_frame_number = eval_frame_number
        self.detail = detail
        self.warmup_frame_number = warmup_frame_number
        self.compute_lidar_metrics = compute_lidar_metrics
        self.axis_diagnostics = axis_diagnostics
        self.frame_interval = float(frame_interval)

        if torch.cuda.is_available():
            local_rank = int(torch.cuda.current_device())
            self.metric_device = torch.device(f'cuda:{local_rank}')
        else:
            self.metric_device = torch.device('cpu')
        
        index_to_time = [
            f'{f * self.frame_interval:g}s'
            for f in range(self.eval_frame_number)]
        self.index_to_time = index_to_time

        if self.detail and self.compute_lidar_metrics:
            self.point_extractor = pretrained_pointnet(
                dataset="shapenet",
                compile=True,
                device=self.metric_device
            )
        self.chamfer_dist = (
            ChamferDistance() if self.compute_lidar_metrics else None)
    
        self.vis_interval = vis_interval
        self.step = 0
        if self.vis_interval > 0:
            self.visualizer = LidarBEVVisualizer(save_dir=vis_dir)

        self._reset_streaming_state()

    def _reset_streaming_state(self):
        self._num_samples = 0
        self._processed_count = 0
        self._pending = None
        self._chamfer_sum = [0.0] * self.eval_frame_number
        self._axis_sums = {
            key: [0.0] * self.eval_frame_number
            for key in ('xy', 'z', 'z_bias', 'z_mae')
        }
        self._pred_feat_samples = [[] for _ in range(self.eval_frame_number)]
        self._targ_feat_samples = [[] for _ in range(self.eval_frame_number)]
        self._pred_hist_samples = [[] for _ in range(self.eval_frame_number)]
        self._targ_hist_samples = [[] for _ in range(self.eval_frame_number)]

    def _commit_pending(self):
        if self._pending is None:
            return
        self._num_samples += 1
        for f in range(self.eval_frame_number):
            self._chamfer_sum[f] += self._pending['chamfer_sum'][f]
            if self.axis_diagnostics:
                for key in self._axis_sums:
                    self._axis_sums[key][f] += self._pending['axis_sums'][key][f]
            if self.detail:
                self._pred_feat_samples[f].append(self._pending['pred_feats'][f])
                self._targ_feat_samples[f].append(self._pending['targ_feats'][f])
                self._pred_hist_samples[f].append(self._pending['pred_hists'][f])
                self._targ_hist_samples[f].append(self._pending['targ_hists'][f])
        self._pending = None

    def _valid_local_count(self, size):
        rank = get_rank()
        world_size = get_world_size()
        if size <= rank:
            return 0
        return min(self._processed_count,
                   (size - 1 - rank) // world_size + 1)

    def _finalize_pending(self, size):
        valid_count = self._valid_local_count(size)
        if valid_count == self._processed_count:
            self._commit_pending()
            return valid_count
        if valid_count == self._processed_count - 1:
            return valid_count
        if valid_count < self._processed_count:
            # SequenceGroupDistributedSampler assigns complete sequences to
            # ranks, so local sample counts can differ by more than one without
            # any distributed padding. In that case, keep the true rank tail.
            self._commit_pending()
            return self._processed_count
        raise RuntimeError(
            f'Unexpected distributed eval padding state: '
            f'processed={self._processed_count}, valid={valid_count}')

    @staticmethod
    def _get_sample_meta(data_sample, key):
        if data_sample is None:
            return None
        if hasattr(data_sample, key):
            return getattr(data_sample, key)
        if hasattr(data_sample, 'metainfo') and key in data_sample.metainfo:
            return data_sample.metainfo[key]
        if isinstance(data_sample, dict):
            if key in data_sample:
                return data_sample.get(key)
            metainfo = data_sample.get('metainfo', {})
            if key in metainfo:
                return metainfo[key]
        return None

    @staticmethod
    def _get_sample_field(data_sample, key):
        if data_sample is None:
            return None
        if hasattr(data_sample, key):
            return getattr(data_sample, key)
        if isinstance(data_sample, dict):
            return data_sample.get(key)
        try:
            return data_sample[key]
        except Exception:
            return None

    def _has_valid_history(self, data_sample):
        if self.warmup_frame_number <= 0:
            return True
        sequence_frame_idx = self._get_sample_meta(
            data_sample, 'sequence_frame_idx')
        if sequence_frame_idx is None:
            return True
        return int(sequence_frame_idx) >= self.warmup_frame_number

    def process(self, data_batchs, data_samples):
        batch_data_samples = data_batchs.get('data_samples', [])
        for index, data_sample in enumerate(data_samples):
            self._commit_pending()
            batch_data_sample = (
                batch_data_samples[index]
                if index < len(batch_data_samples) else batch_data_samples[0])
            total_token = self._get_sample_meta(batch_data_sample, 'total_token')
            is_valid_sequence = (
                total_token is None or len(set(total_token)) == len(total_token))
            has_valid_history = self._has_valid_history(batch_data_sample)
            self._processed_count += 1
            if not is_valid_sequence or not has_valid_history:
                continue

            pred_points = self._get_sample_field(data_sample, 'pred_points')
            targ_points = self._get_sample_field(data_sample, 'targ_points')
            if pred_points is None:
                continue

            chamfer_sum = [0.0] * self.eval_frame_number
            axis_sums = {
                key: [0.0] * self.eval_frame_number
                for key in ('xy', 'z', 'z_bias', 'z_mae')
            }
            pred_feats = []
            targ_feats = []
            pred_hists = []
            targ_hists = []
            pred_points_vis = []
            targ_points_vis = []
            visualize_sample, vis_index = should_visualize(
                (batch_data_sample, data_sample), self._get_sample_meta,
                self.vis_interval, self.step)

            if self.compute_lidar_metrics and targ_points is not None:
                with torch.no_grad():
                    for f in range(self.eval_frame_number):
                        pred_frame_points = pred_points[f][:self.max_pred_points].to(
                            self.metric_device)
                        targ_frame_points = targ_points[f].to(
                            self.metric_device)
                        pred_frame_xyz = pred_frame_points[:, :3]
                        targ_frame_xyz = targ_frame_points[:, :3]
                        if self.max_pred_points is None and (
                                not len(pred_frame_xyz) or not len(targ_frame_xyz)):
                            raise RuntimeError('Empty point cloud in CD evaluation')
                        chamfer_dist = self.chamfer_dist(
                            pred_frame_xyz.unsqueeze(0),
                            targ_frame_xyz.unsqueeze(0),
                            point_reduction='mean',
                            bidirectional=True
                        )
                        if self.max_pred_points is None and not torch.isfinite(chamfer_dist):
                            raise RuntimeError('Non-finite CD in evaluation')
                        chamfer_sum[f] = float(chamfer_dist.item())

                        if self.axis_diagnostics:
                            pred_batch = pred_frame_xyz.unsqueeze(0)
                            targ_batch = targ_frame_xyz.unsqueeze(0)
                            pred_nn = knn_points(pred_batch, targ_batch, K=1)
                            targ_nn = knn_points(targ_batch, pred_batch, K=1)
                            pred_match = targ_frame_xyz[pred_nn.idx[0, :, 0]]
                            targ_match = pred_frame_xyz[targ_nn.idx[0, :, 0]]
                            pred_delta = pred_frame_xyz - pred_match
                            # Keep both directions in pred-minus-GT convention.
                            targ_delta = targ_match - targ_frame_xyz
                            axis_sums['xy'][f] = float(
                                (pred_delta[:, :2].square().sum(-1).mean() +
                                 targ_delta[:, :2].square().sum(-1).mean()).item())
                            axis_sums['z'][f] = float(
                                (pred_delta[:, 2].square().mean() +
                                 targ_delta[:, 2].square().mean()).item())
                            axis_sums['z_bias'][f] = float(
                                (0.5 * (pred_delta[:, 2].mean() +
                                        targ_delta[:, 2].mean())).item())
                            axis_sums['z_mae'][f] = float(
                                (0.5 * (pred_delta[:, 2].abs().mean() +
                                        targ_delta[:, 2].abs().mean())).item())

                        if self.detail:
                            input_pred_pts = rearrange(pred_frame_xyz, 'n c->1 c n') / self.DATASET_MAX_DEPTH
                            input_targ_pts = rearrange(targ_frame_xyz, 'm c->1 c m') / self.DATASET_MAX_DEPTH
                            pred_point_feat = self.point_extractor(input_pred_pts.clone().detach())
                            targ_point_feat = self.point_extractor(input_targ_pts.clone().detach())
                            pred_feats.append(pred_point_feat.cpu())
                            targ_feats.append(targ_point_feat.cpu())
                            pred_hists.append(point_cloud_to_histogram(pred_frame_xyz).cpu())
                            targ_hists.append(point_cloud_to_histogram(targ_frame_xyz).cpu())

                        if visualize_sample:
                            pred_points_vis.append(pred_frame_xyz.detach().cpu())
                            targ_points_vis.append(targ_frame_xyz.detach().cpu())
            elif self.compute_lidar_metrics:
                for f in range(self.eval_frame_number):
                    if visualize_sample:
                        pred_points_vis.append(
                            pred_points[f][:self.max_pred_points, :3].detach().cpu())

            self._pending = dict(chamfer_sum=chamfer_sum)
            if self.axis_diagnostics:
                self._pending['axis_sums'] = axis_sums
            if self.detail:
                self._pending.update(
                    pred_feats=pred_feats, targ_feats=targ_feats,
                    pred_hists=pred_hists, targ_hists=targ_hists)

            if (self.compute_lidar_metrics and targ_points is not None and
                    visualize_sample):
                self.visualizer.draw_lidar_bev_compare(
                    pred_points_vis,
                    targ_points_vis,
                    name_prefix='compare',
                    save_name=visualization_save_name(
                        batch_data_sample, self._get_sample_meta, vis_index)
                )
            self.step += 1

    def evaluate(self, size):
        self._finalize_pending(size)

        rank_summary = dict(
            num_samples=self._num_samples,
            chamfer_sum=self._chamfer_sum,
            axis_sums=self._axis_sums,
            pred_feat_samples=self._pred_feat_samples,
            targ_feat_samples=self._targ_feat_samples,
            pred_hist_samples=self._pred_hist_samples,
            targ_hist_samples=self._targ_hist_samples,
        )
        if get_world_size() > 1:
            gathered = [None for _ in range(get_world_size())]
            torch.distributed.all_gather_object(gathered, rank_summary)
            results = gathered if is_main_process() else None
        else:
            results = [rank_summary]

        if is_main_process():
            metrics = self.compute_metrics(results)
            if self.prefix:
                metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in metrics.items()
                }
        else:
            metrics = None

        metrics_list = [metrics]
        broadcast_object_list(metrics_list)
        self.results.clear()
        self._reset_streaming_state()
        self.step = 0
        if hasattr(self, 'visualizer'):
            self.visualizer.step = 0
        return metrics_list[0]

    def compute_metrics(self, results):
        logger: MMLogger = MMLogger.get_current_instance()

        logger.info('----------- Computing LiDAR 4D Metrics ---------------')
        num_samples = sum(res['num_samples'] for res in results)
        logger.info(f'Valid Eval number of Samples: {num_samples}')

        total_chamfer_distance = [0.0] * self.eval_frame_number
        total_axis = {
            key: [0.0] * self.eval_frame_number
            for key in ('xy', 'z', 'z_bias', 'z_mae')
        }

        if self.detail:
            pred_feats_list = [[] for _ in range(self.eval_frame_number)]
            targ_feats_list = [[] for _ in range(self.eval_frame_number)]
            pred_hists_list = [[] for _ in range(self.eval_frame_number)]
            targ_hists_list = [[] for _ in range(self.eval_frame_number)]

        for res in results:
            if res['num_samples'] == 0:
                continue
            for f in range(self.eval_frame_number):
                total_chamfer_distance[f] += res['chamfer_sum'][f]
                if self.axis_diagnostics:
                    for key in total_axis:
                        total_axis[key][f] += res['axis_sums'][key][f]

                if self.detail:
                    pred_feats_list[f].extend(res['pred_feat_samples'][f])
                    targ_feats_list[f].extend(res['targ_feat_samples'][f])
                    pred_hists_list[f].extend(res['pred_hist_samples'][f])
                    targ_hists_list[f].extend(res['targ_hist_samples'][f])
            
        metrics = {}
        
        has_future_avg = self.eval_frame_number > 1
        table_header = ['Metric'] + self.index_to_time[:self.eval_frame_number]
        if has_future_avg:
            table_header.append('future_avg')
        summary_table_data = [table_header]
        
        if self.compute_lidar_metrics:
            cd_row = ['Chamfer_Distance']
            cd_values = []
            
            for f in range(self.eval_frame_number):
                avg_cd = total_chamfer_distance[f] / num_samples if num_samples > 0 else 0.0
                time_str = self.index_to_time[f]
                metrics[f'CD_{time_str}'] = avg_cd
                cd_values.append(avg_cd)
                cd_row.append(f'{avg_cd:.4f}')

            if has_future_avg:
                cd_future_avg = float(np.mean(cd_values[1:]))
                metrics['CD_future_avg'] = cd_future_avg
                cd_row.append(f'{cd_future_avg:.4f}')

            summary_table_data.append(cd_row)

            if self.axis_diagnostics:
                axis_rows = {
                    'xy': ['CD_XY_component'],
                    'z': ['CD_Z_component'],
                    'z_fraction': ['CD_Z_fraction'],
                    'z_bias': ['Z_bias_m'],
                    'z_mae': ['Z_MAE_m'],
                }
                axis_values = {key: [] for key in axis_rows}
                for f in range(self.eval_frame_number):
                    values = {
                        key: total_axis[key][f] / num_samples
                        if num_samples > 0 else 0.0
                        for key in total_axis
                    }
                    denom = values['xy'] + values['z']
                    values['z_fraction'] = values['z'] / denom if denom else 0.0
                    for key, row in axis_rows.items():
                        value = values[key]
                        axis_values[key].append(value)
                        metrics[f'{row[0]}_{self.index_to_time[f]}'] = value
                        row.append(f'{value:.4f}')
                if has_future_avg:
                    for key, row in axis_rows.items():
                        value = float(np.mean(axis_values[key][1:]))
                        metrics[f'{row[0]}_future_avg'] = value
                        row.append(f'{value:.4f}')
                summary_table_data.extend(axis_rows.values())

        if self.detail and self.compute_lidar_metrics:
            if len(summary_table_data) > 1:
                print_log('\n' + AsciiTable(summary_table_data).table, logger=logger)
            metrics.update(self._compute_pooled_distribution(
                dict(pred_feats=pred_feats_list, targ_feats=targ_feats_list,
                     pred_hists=pred_hists_list, targ_hists=targ_hists_list),
                num_samples, [res['num_samples'] for res in results], metrics))
            return metrics

        if len(summary_table_data) > 1:
            table = AsciiTable(summary_table_data)
            print_log('\n' + table.table, logger=logger)

        return metrics

    def _compute_pooled_distribution(self, frame_samples, num_samples,
                                     rank_valid_sequences, geometry_metrics):
        logger = MMLogger.get_current_instance()
        if num_samples < 2:
            raise RuntimeError('Insufficient valid future point clouds')
        frames = {key: [] for key in frame_samples}
        counts = []
        for f in range(self.eval_frame_number):
            frame_counts = []
            for key, samples_by_frame in frame_samples.items():
                samples = samples_by_frame[f]
                frame_counts.append(len(samples))
                if len(samples) != num_samples:
                    raise RuntimeError(
                        f'Truncated/missing {key} at frame {f}: '
                        f'{len(samples)} != {num_samples}')
                combined = (torch.cat(samples, dim=0) if key.endswith('feats')
                            else torch.stack(samples, dim=0))
                frames[key].append(combined.cpu())
            counts.append(frame_counts)
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save({key: value[f] for key, value in frames.items()},
                           self.cache_dir / f'frame_{f}.pt')
        if self.cache_dir is not None:
            metadata = dict(
                valid_sequences=num_samples,
                future_point_clouds=num_samples * (self.eval_frame_number - 1),
                future_frame_indices=list(range(1, self.eval_frame_number)),
                times=self.index_to_time, fields=list(frames),
                frame_field_counts=counts, rank_valid_sequences=rank_valid_sequences,
                point_mmd_estimator='exact_full_polynomial_unbiased',
                bev_mmd_estimator='exact_full_rbf_biased',
                mmd_block_size=self.mmd_block_size,
                cd={key: value for key, value in geometry_metrics.items()
                    if key.startswith('CD_')})
            (self.cache_dir / 'metadata.json').write_text(
                json.dumps(metadata, indent=2) + '\n')
            logger.info(f'Saved all per-frame features / histograms to {self.cache_dir}')
        metrics = pooled_distribution_metrics(
            frames, self.metric_device, self.mmd_block_size, logger)
        metrics['Valid_Sequence_Count'] = float(num_samples)
        table = [['Metric', f'pooled_future ({self.index_to_time[1]}–'
                            f'{self.index_to_time[-1]})']]
        table.extend([[key, f'{value:.8f}'] for key, value in metrics.items()])
        print_log('\n' + AsciiTable(table).table, logger=logger)
        return metrics
