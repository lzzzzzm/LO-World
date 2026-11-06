import os

from einops import rearrange
import numpy as np
import scipy.linalg
import torch
from chamferdist import ChamferDistance
from mmengine.dist import (broadcast_object_list, get_rank, get_world_size,
                           is_main_process)
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from mmdet3d.registry import METRICS
from projects.evaluation.functional.lidar_3d_utils import (
    compute_jsd_2d,
    compute_mmd_2d,
    compute_squared_mmd,
    point_cloud_to_histogram,
    pretrained_pointnet,
)
from projects.visualizer.lidar_visualizer import LidarBEVVisualizer
from .visualization import should_visualize, visualization_save_name


@METRICS.register_module()
class LiDAR3DMetric(BaseMetric):
    def __init__(self,
                 collect_device='cpu',
                 prefix=None,
                 vis_interval=0,
                 vis_dir='work_dirs/vis_lidar',
                 max_mmd_samples=10000,
                 intensity_match_max_points=4096,
                 max_pred_points=50000,
                 cd_only=False):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.max_pred_points = max_pred_points
        self.cd_only = bool(cd_only)
        if max_pred_points is not None and max_pred_points <= 0:
            raise ValueError('max_pred_points must be positive or None.')
        self.DATASET_MAX_DEPTH = 80.0
        self.max_mmd_samples = max_mmd_samples
        self.intensity_match_max_points = int(intensity_match_max_points)
        if self.intensity_match_max_points <= 0:
            raise ValueError('intensity_match_max_points must be positive.')

        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            self.metric_device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(self.metric_device)
        else:
            if torch.cuda.is_available():
                local_rank = torch.cuda.current_device()
                self.metric_device = torch.device(f'cuda:{local_rank}')
            else:
                self.metric_device = torch.device('cpu')

        if not self.cd_only:
            self.point_extractor = pretrained_pointnet(
                dataset='shapenet',
                compile=True,
                device=self.metric_device,
            )
        self.chamfer_dist = ChamferDistance()

        self.vis_interval = vis_interval
        self.step = 0
        if self.vis_interval > 0:
            self.visualizer = LidarBEVVisualizer(save_dir=vis_dir)

        self._reset_streaming_state()

    def _reset_streaming_state(self):
        self._num_samples = 0
        self._processed_count = 0
        self._pending = None
        self._chamfer_sum = 0.0
        self._pred_feat_sum = None
        self._targ_feat_sum = None
        self._pred_feat_outer = None
        self._targ_feat_outer = None
        self._pred_hist_sum = None
        self._targ_hist_sum = None
        self._intensity_abs_sum = 0.0
        self._intensity_sq_sum = 0.0
        self._intensity_match_count = 0
        self._pred_feat_samples = []
        self._targ_feat_samples = []
        self._pred_hist_samples = []
        self._targ_hist_samples = []

    def _sample_summary(self,
                        chamfer,
                        pred_feat,
                        targ_feat,
                        pred_hist,
                        targ_hist,
                        intensity_abs_sum=0.0,
                        intensity_sq_sum=0.0,
                        intensity_match_count=0):
        pred_feat = pred_feat.detach().cpu().to(torch.float64)
        targ_feat = targ_feat.detach().cpu().to(torch.float64)
        pred_vec = pred_feat.reshape(-1)
        targ_vec = targ_feat.reshape(-1)
        pred_hist = pred_hist.detach().cpu().to(torch.float32)
        targ_hist = targ_hist.detach().cpu().to(torch.float32)

        return dict(
            chamfer=chamfer,
            pred_feat_sum=pred_vec,
            targ_feat_sum=targ_vec,
            pred_feat_outer=torch.outer(pred_vec, pred_vec),
            targ_feat_outer=torch.outer(targ_vec, targ_vec),
            pred_feat=pred_feat.to(torch.float32),
            targ_feat=targ_feat.to(torch.float32),
            pred_hist=pred_hist,
            targ_hist=targ_hist,
            intensity_abs_sum=float(intensity_abs_sum),
            intensity_sq_sum=float(intensity_sq_sum),
            intensity_match_count=int(intensity_match_count),
        )

    def _commit_summary(self, summary):
        self._num_samples += 1
        self._chamfer_sum += summary['chamfer']
        if self.cd_only:
            return

        if self._pred_feat_sum is None:
            self._pred_feat_sum = summary['pred_feat_sum'].clone()
            self._targ_feat_sum = summary['targ_feat_sum'].clone()
            self._pred_feat_outer = summary['pred_feat_outer'].clone()
            self._targ_feat_outer = summary['targ_feat_outer'].clone()
            self._pred_hist_sum = summary['pred_hist'].clone()
            self._targ_hist_sum = summary['targ_hist'].clone()
        else:
            self._pred_feat_sum += summary['pred_feat_sum']
            self._targ_feat_sum += summary['targ_feat_sum']
            self._pred_feat_outer += summary['pred_feat_outer']
            self._targ_feat_outer += summary['targ_feat_outer']
            self._pred_hist_sum += summary['pred_hist']
            self._targ_hist_sum += summary['targ_hist']

        self._intensity_abs_sum += summary['intensity_abs_sum']
        self._intensity_sq_sum += summary['intensity_sq_sum']
        self._intensity_match_count += summary['intensity_match_count']

        if len(self._pred_feat_samples) < self.max_mmd_samples:
            self._pred_feat_samples.append(summary['pred_feat'])
            self._targ_feat_samples.append(summary['targ_feat'])
            self._pred_hist_samples.append(summary['pred_hist'])
            self._targ_hist_samples.append(summary['targ_hist'])

    def _commit_pending(self):
        if self._pending is not None:
            self._commit_summary(self._pending)
            self._pending = None

    def _valid_local_count(self, size):
        rank = get_rank()
        world_size = get_world_size()
        if size <= rank:
            return 0
        return min(self._processed_count,
                   (size - 1 - rank) // world_size + 1)

    @staticmethod
    def _get_sample_meta(data_sample, key):
        if data_sample is None:
            return None
        if hasattr(data_sample, key):
            return getattr(data_sample, key)
        if isinstance(data_sample, dict) and key in data_sample:
            return data_sample[key]
        metainfo = getattr(data_sample, 'metainfo', {})
        if key in metainfo:
            return metainfo[key]
        if isinstance(data_sample, dict):
            metainfo = data_sample.get('metainfo', {})
            if key in metainfo:
                return metainfo[key]
        return None

    def _subsample_points_for_intensity(self, points):
        if points.shape[0] <= self.intensity_match_max_points:
            return points
        indices = torch.linspace(
            0,
            points.shape[0] - 1,
            self.intensity_match_max_points,
            device=points.device).long()
        return points[indices]

    def _directed_intensity_error(self, source_points, target_points):
        source_points = self._subsample_points_for_intensity(source_points)
        target_points = self._subsample_points_for_intensity(target_points)
        if source_points.shape[0] == 0 or target_points.shape[0] == 0:
            return 0.0, 0.0, 0

        dist = torch.cdist(source_points[:, :3].float(),
                           target_points[:, :3].float())
        nn_idx = dist.argmin(dim=1)
        diff = (source_points[:, 3].float() -
                target_points[nn_idx, 3].float())
        diff = diff[torch.isfinite(diff)]
        if diff.numel() == 0:
            return 0.0, 0.0, 0
        return (diff.abs().sum().item(), diff.square().sum().item(),
                int(diff.numel()))

    def _points_to_intensity_error(self, pred_points, targ_points):
        if pred_points.shape[1] <= 3 or targ_points.shape[1] <= 3:
            return 0.0, 0.0, 0

        pred_points = pred_points[torch.isfinite(pred_points[:, 3])]
        targ_points = targ_points[torch.isfinite(targ_points[:, 3])]
        if pred_points.shape[0] == 0 or targ_points.shape[0] == 0:
            return 0.0, 0.0, 0

        if pred_points.shape[0] == targ_points.shape[0]:
            diff = pred_points[:, 3].float() - targ_points[:, 3].float()
            diff = diff[torch.isfinite(diff)]
            if diff.numel() == 0:
                return 0.0, 0.0, 0
            return (diff.abs().sum().item(), diff.square().sum().item(),
                    int(diff.numel()))

        pred_abs, pred_sq, pred_count = self._directed_intensity_error(
            pred_points, targ_points)
        targ_abs, targ_sq, targ_count = self._directed_intensity_error(
            targ_points, pred_points)
        return (pred_abs + targ_abs, pred_sq + targ_sq,
                pred_count + targ_count)

    def process(self, data_batchs, data_samples):
        input_data_samples = data_batchs.get('data_samples', [])
        for sample_idx, data_sample in enumerate(data_samples):
            self._commit_pending()

            pred_points_full = data_sample['pred_points'][:self.max_pred_points].to(
                self.metric_device)
            targ_points_full = data_sample['targ_points'].to(self.metric_device)
            pred_points = pred_points_full[:, :3]
            targ_points = targ_points_full[:, :3]

            if self.cd_only:
                if not len(pred_points) or not len(targ_points):
                    raise RuntimeError('Empty point cloud in CD evaluation')
                with torch.no_grad():
                    value = self.chamfer_dist(
                        pred_points[None], targ_points[None],
                        point_reduction='mean', bidirectional=True)
                if not torch.isfinite(value):
                    raise RuntimeError('Non-finite CD in evaluation')
                self._pending = dict(chamfer=float(value.item()))
                self._processed_count += 1
                self.step += 1
                continue

            with torch.no_grad():
                chamfer_dist = self.chamfer_dist(
                    pred_points.unsqueeze(0),
                    targ_points.unsqueeze(0),
                    point_reduction='mean',
                    bidirectional=True,
                )

                input_pred_pts = rearrange(
                    pred_points, 'n c->1 c n') / self.DATASET_MAX_DEPTH
                input_targ_pts = rearrange(
                    targ_points, 'm c->1 c m') / self.DATASET_MAX_DEPTH
                pred_points_feat = self.point_extractor(
                    input_pred_pts.clone().detach())
                targ_points_feat = self.point_extractor(
                    input_targ_pts.clone().detach())

            pred_points_bev_hist = point_cloud_to_histogram(pred_points)
            targ_points_bev_hist = point_cloud_to_histogram(targ_points)
            intensity_abs_sum, intensity_sq_sum, intensity_count = (
                self._points_to_intensity_error(
                    pred_points_full, targ_points_full))
            self._pending = self._sample_summary(
                float(chamfer_dist.item()),
                pred_points_feat,
                targ_points_feat,
                pred_points_bev_hist,
                targ_points_bev_hist,
                intensity_abs_sum,
                intensity_sq_sum,
                intensity_count,
            )
            input_data_sample = (
                input_data_samples[sample_idx]
                if sample_idx < len(input_data_samples) else None)
            self._processed_count += 1

            visualize_sample, vis_index = should_visualize(
                (input_data_sample, data_sample), self._get_sample_meta,
                self.vis_interval, self.step)
            if visualize_sample:
                self.visualizer.draw_lidar_bev_compare(
                    [pred_points.detach().cpu()],
                    [targ_points.detach().cpu()],
                    name_prefix='compare',
                    save_name=visualization_save_name(
                        input_data_sample, self._get_sample_meta, vis_index),
                )
            self.step += 1

    def _rank_summary(self):
        return dict(
            num_samples=self._num_samples,
            chamfer_sum=self._chamfer_sum,
            pred_feat_sum=self._pred_feat_sum,
            targ_feat_sum=self._targ_feat_sum,
            pred_feat_outer=self._pred_feat_outer,
            targ_feat_outer=self._targ_feat_outer,
            pred_hist_sum=self._pred_hist_sum,
            targ_hist_sum=self._targ_hist_sum,
            intensity_abs_sum=self._intensity_abs_sum,
            intensity_sq_sum=self._intensity_sq_sum,
            intensity_match_count=self._intensity_match_count,
            pred_feat_samples=self._pred_feat_samples,
            targ_feat_samples=self._targ_feat_samples,
            pred_hist_samples=self._pred_hist_samples,
            targ_hist_samples=self._targ_hist_samples,
        )

    def evaluate(self, size):
        valid_count = self._valid_local_count(size)
        if valid_count == self._processed_count:
            self._commit_pending()
        elif valid_count != self._processed_count - 1:
            raise RuntimeError(
                f'Unexpected distributed eval padding state: '
                f'processed={self._processed_count}, valid={valid_count}')

        rank_summary = self._rank_summary()
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

    @staticmethod
    def _fid_from_stats(pred_sum, pred_outer, targ_sum, targ_outer,
                        num_samples):
        pred_mu_tensor = pred_sum / num_samples
        targ_mu_tensor = targ_sum / num_samples
        pred_mu = pred_mu_tensor.numpy()
        targ_mu = targ_mu_tensor.numpy()

        if num_samples > 1:
            pred_cov = (
                pred_outer -
                num_samples * torch.outer(pred_mu_tensor, pred_mu_tensor)
            ) / (num_samples - 1)
            targ_cov = (
                targ_outer -
                num_samples * torch.outer(targ_mu_tensor, targ_mu_tensor)
            ) / (num_samples - 1)
        else:
            pred_cov = torch.zeros_like(pred_outer)
            targ_cov = torch.zeros_like(targ_outer)

        pred_cov = pred_cov.numpy()
        targ_cov = targ_cov.numpy()
        m = np.square(pred_mu - targ_mu).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(pred_cov, targ_cov), disp=False)
        return float(np.real(m + np.trace(pred_cov + targ_cov - s * 2)))

    def compute_metrics(self, results):
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('----------- Computing LiDAR 3D Metrics ---------------')

        num_samples = sum(res['num_samples'] for res in results)
        logger.info(f'Eval number of Samples: {num_samples}')
        if self.cd_only:
            total = sum(res['chamfer_sum'] for res in results)
            metrics = dict(
                Chamfer_Distance=total / num_samples if num_samples else 0.0)
            table_data = [['Metric', 'Value']]
            for key, value in metrics.items():
                table_data.append([key, f'{value:.4f}'])
            print_log('\n' + AsciiTable(table_data).table, logger=logger)
            return metrics

        total_chamfer_distance = 0.0
        pred_feat_sum = None
        targ_feat_sum = None
        pred_feat_outer = None
        targ_feat_outer = None
        pred_hist_sum = None
        targ_hist_sum = None
        intensity_abs_sum = 0.0
        intensity_sq_sum = 0.0
        intensity_match_count = 0
        pred_feats_list = []
        targ_feats_list = []
        pred_hists_list = []
        targ_hists_list = []
        for res in results:
            if res['num_samples'] == 0:
                continue
            total_chamfer_distance += res['chamfer_sum']

            if pred_feat_sum is None:
                pred_feat_sum = res['pred_feat_sum'].cpu()
                targ_feat_sum = res['targ_feat_sum'].cpu()
                pred_feat_outer = res['pred_feat_outer'].cpu()
                targ_feat_outer = res['targ_feat_outer'].cpu()
                pred_hist_sum = res['pred_hist_sum'].cpu()
                targ_hist_sum = res['targ_hist_sum'].cpu()
            else:
                pred_feat_sum += res['pred_feat_sum'].cpu()
                targ_feat_sum += res['targ_feat_sum'].cpu()
                pred_feat_outer += res['pred_feat_outer'].cpu()
                targ_feat_outer += res['targ_feat_outer'].cpu()
                pred_hist_sum += res['pred_hist_sum'].cpu()
                targ_hist_sum += res['targ_hist_sum'].cpu()

            intensity_abs_sum += float(res.get('intensity_abs_sum', 0.0))
            intensity_sq_sum += float(res.get('intensity_sq_sum', 0.0))
            intensity_match_count += int(
                res.get('intensity_match_count', 0))

            for pred_feat, targ_feat, pred_hist, targ_hist in zip(
                    res['pred_feat_samples'], res['targ_feat_samples'],
                    res['pred_hist_samples'], res['targ_hist_samples']):
                if len(pred_feats_list) >= self.max_mmd_samples:
                    break
                pred_feats_list.append(pred_feat)
                targ_feats_list.append(targ_feat)
                pred_hists_list.append(pred_hist)
                targ_hists_list.append(targ_hist)

        avg_chamfer_distance = (
            total_chamfer_distance / num_samples if num_samples > 0 else 0.0)

        point_fid = self._fid_from_stats(
            pred_feat_sum, pred_feat_outer, targ_feat_sum, targ_feat_outer,
            num_samples)

        pred_feats = torch.cat(
            pred_feats_list, dim=0).numpy().astype(np.float64)
        targ_feats = torch.cat(
            targ_feats_list, dim=0).numpy().astype(np.float64)
        pred_hists = torch.stack(pred_hists_list, dim=0)
        targ_hists = torch.stack(targ_hists_list, dim=0)

        point_mmd = compute_squared_mmd(pred_feats, targ_feats)
        bev_jsd = compute_jsd_2d(pred_hist_sum[None].float(),
                                 targ_hist_sum[None].float())
        bev_mmd = compute_mmd_2d(pred_hists.to(self.metric_device).float(),
                                 targ_hists.to(self.metric_device).float())

        metrics = {
            'Chamfer_Distance': avg_chamfer_distance,
            'Point_Feature_FID': point_fid,
            'Point_Squared_MMD': point_mmd,
            'BEV_Feature_JSDx10^2': bev_jsd * 1e2,
            'BEV_Feature_MMDx10^4': bev_mmd * 1e4,
        }
        if intensity_match_count > 0:
            metrics['Intensity_Matched_MAE'] = (
                intensity_abs_sum / intensity_match_count)
            metrics['Intensity_Matched_RMSE'] = np.sqrt(
                intensity_sq_sum / intensity_match_count)

        table_data = [['Metric', 'Value']]
        for key, value in metrics.items():
            table_data.append([key, f'{value:.4f}'])

        table = AsciiTable(table_data)
        print_log('\n' + table.table, logger=logger)
        return metrics
