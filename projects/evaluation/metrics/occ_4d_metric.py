
import torch
from mmengine.dist import (broadcast_object_list, get_rank, get_world_size,
                           is_main_process)
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.logging import print_log

from mmdet3d.registry import METRICS
from terminaltables import AsciiTable
from projects.visualizer.occ_visualizer import OccBEVVisualizer
from .visualization import should_visualize, visualization_save_name

@METRICS.register_module()
class Occ4DMetric(BaseMetric):
    def __init__(
        self, 
        eval_frame_number=7,
        num_classes=18, 
        empty_cls=17, 
        ignore_index=255, 
        label2cat=None, 
        collect_device='cpu', 
        prefix=None, 
        vis_interval=0, 
        vis_dir='work_dirs/vis_4d_occ',
        detail=False,
        warmup_frame_number=0,
        frame_interval=0.5,
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.empty_cls = empty_cls
        self.vis_interval = vis_interval
        self.step = 0
        self.eval_frame_number = eval_frame_number
        self.detail = detail
        self.warmup_frame_number = warmup_frame_number
        self.frame_interval = float(frame_interval)

        if self.vis_interval > 0:
            self.visualizer = OccBEVVisualizer(
                save_dir=vis_dir, 
                empty_cls=empty_cls, 
                ignore_index=ignore_index
            )
        
        default_categories = ['others', 'barrier', 'bicycle', 'bus', 'car', 
                              'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 
                              'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 
                              'terrain', 'manmade', 'vegetation', 'free']
        
        index_to_time = [
            f'{f * self.frame_interval:g}s'
            for f in range(self.eval_frame_number)]

        self.index_to_time = index_to_time
        self.label2cat = label2cat if label2cat is not None else default_categories
        self._hist = torch.zeros(
            (self.eval_frame_number, self.num_classes, self.num_classes),
            dtype=torch.float64)
        self._pending_hist = None
        self._processed_count = 0
        self._num_samples = 0

    def _commit_pending(self):
        if self._pending_hist is not None:
            self._hist += self._pending_hist
            self._pending_hist = None
            self._num_samples += 1

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
        return data_sample.get(key)

    def _has_valid_history(self, data_sample):
        if self.warmup_frame_number <= 0:
            return True
        sequence_frame_idx = self._get_sample_meta(
            data_sample, 'sequence_frame_idx')
        if sequence_frame_idx is None:
            return True
        return int(sequence_frame_idx) >= self.warmup_frame_number

    def process(self, data_batchs, data_samples):
        """
            Process one batch of data samples and predictions.
            Accumulate compact per-frame confusion matrices instead of storing
            full 4D occupancy tensors until ``compute_metrics``.
        """
        for batch_data_sample, data_sample in zip(
                data_batchs['data_samples'], data_samples, strict=True):
            self._commit_pending()
            total_token = self._get_sample_meta(batch_data_sample, 'total_token')
            is_valid_sequence = (
                total_token is None or len(set(total_token)) == len(total_token))
            has_valid_history = self._has_valid_history(batch_data_sample)
            self._processed_count += 1
            if not is_valid_sequence or not has_valid_history:
                continue

            pred_voxel_semantics = data_sample['pred_voxel_semantics']  # [F, Y, X, Z]
            targ_voxel_semantics = data_sample['targ_voxel_semantics']  # [F, Y, X, Z]
            hist = torch.zeros_like(self._hist)

            for f in range(self.eval_frame_number):
                pred = pred_voxel_semantics[f].flatten().cpu()
                targ = targ_voxel_semantics[f].flatten().cpu()

                mask = (targ != self.ignore_index) & (targ >= 0) & (targ < self.num_classes)
                pred = pred[mask]
                targ = targ[mask]

                valid_pred_mask = pred < self.num_classes
                pred = pred[valid_pred_mask]
                targ = targ[valid_pred_mask]

                idx = targ * self.num_classes + pred
                bincount = torch.bincount(idx, minlength=self.num_classes**2)
                hist[f] = bincount.reshape(
                    self.num_classes, self.num_classes).to(torch.float64)

            self._pending_hist = hist

            visualize_sample, vis_index = should_visualize(
                (batch_data_sample, data_sample), self._get_sample_meta,
                self.vis_interval, self.step)
            if visualize_sample:
                self.visualizer.draw_occ_bev_compare(
                    pred_voxel_semantics.cpu(),
                    targ_voxel_semantics.cpu(),
                    name_prefix='compare',
                    save_name=visualization_save_name(
                        batch_data_sample, self._get_sample_meta, vis_index)
                )
            self.step += 1

    def evaluate(self, size):
        self._finalize_pending(size)

        rank_summary = dict(hist=self._hist, num_samples=self._num_samples)
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
        self._hist.zero_()
        self._pending_hist = None
        self._processed_count = 0
        self._num_samples = 0
        self.step = 0
        if hasattr(self, 'visualizer'):
            self.visualizer.step = 0
        return metrics_list[0]

    def compute_metrics(self, results):
        """
            Compute the metrics from processed results.
            Occ4DMetric computes:
            1. Per-class IoU for all classes, across all frames
            2. Mean IoU (mIoU) across all classes that appear in GT, excluding empty_cls, across all frames
            3. Non-empty (Occupied) IoU treating all non-empty classes as one and empty class as the other, across all frames
        """
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('----------- Computing Occupancy 4D Metric ---------------')
        logger.info(f'Valid Eval number of Samples: {sum(res["num_samples"] for res in results)}')

        hist = torch.zeros(
            (self.eval_frame_number, self.num_classes, self.num_classes),
            dtype=torch.float64)
        for res in results:
            hist += res['hist'].cpu()
            
        metrics = {}
        eps = 1e-6
        
        # We will create a summary table showing mIoU and Non-empty IoU for each frame
        has_future_avg = self.eval_frame_number > 1
        table_header = ['Metric'] + self.index_to_time[:self.eval_frame_number]
        if has_future_avg:
            table_header.append('future_avg')
        summary_table_data = [table_header]
        
        class_rows = {i: [self.label2cat[i] if isinstance(self.label2cat, list) and i < len(self.label2cat) else self.label2cat.get(i, self.label2cat.get(str(i), f'class_{i}'))] for i in range(self.num_classes)}
        
        miou_row = ['mIoU']
        non_empty_iou_row = ['Non-empty IoU']
        miou_values = []
        non_empty_iou_values = []
        class_iou_values = {i: [] for i in range(self.num_classes)}
        
        for f in range(self.eval_frame_number):
            hist_f = hist[f]
            intersect = torch.diag(hist_f)
            union = hist_f.sum(dim=1) + hist_f.sum(dim=0) - intersect
            ious = intersect / (union + eps)
            
            # Modify: Set 0 IoU to NaN
            ious[ious == 0] = float('nan')
            
            # Calculate mIoU over classes that appear in GT, excluding empty_cls
            valid_classes = union > 0
            valid_classes_no_empty = valid_classes.clone()
            if self.empty_cls < self.num_classes:
                valid_classes_no_empty[self.empty_cls] = False
                
            # Filter out NaN ious for mIoU calculation
            nan_mask = torch.isnan(ious)
            valid_classes_for_miou = valid_classes_no_empty & (~nan_mask)
            
            miou = ious[valid_classes_for_miou].mean().item() if valid_classes_for_miou.any() else 0.0
            
            # Calculate Non-empty (Occupied) IoU
            empty_mask = torch.zeros(self.num_classes, dtype=torch.bool)
            if self.empty_cls < self.num_classes:
                empty_mask[self.empty_cls] = True
            non_empty_mask = ~empty_mask
            
            tp = hist_f[non_empty_mask][:, non_empty_mask].sum().item()
            fp = hist_f[empty_mask][:, non_empty_mask].sum().item() if empty_mask.any() else 0.0
            fn = hist_f[non_empty_mask][:, empty_mask].sum().item() if empty_mask.any() else 0.0
            non_empty_iou = tp / (tp + fp + fn + eps)
            
            time_str = self.index_to_time[f]
            metrics[f'mIoU_{time_str}'] = miou
            metrics[f'Non-empty_IoU_{time_str}'] = non_empty_iou
            miou_values.append(miou)
            non_empty_iou_values.append(non_empty_iou)
            
            miou_row.append(f'{miou:.4f}')
            non_empty_iou_row.append(f'{non_empty_iou:.4f}')
            
            # Store per-class IoU for detailed analysis if needed
            for i in range(self.num_classes):
                class_name = class_rows[i][0]
                if valid_classes[i]:
                    class_iou = ious[i].item()
                    metrics[f'{class_name}_{time_str}'] = class_iou
                    class_rows[i].append(f'{class_iou:.4f}')
                    class_iou_values[i].append(class_iou)
                else:
                    metrics[f'{class_name}_{time_str}'] = float('nan')
                    class_rows[i].append('NaN')
                    class_iou_values[i].append(float('nan'))

        if has_future_avg:
            miou_future_avg = float(torch.tensor(miou_values[1:]).mean().item())
            non_empty_iou_future_avg = float(
                torch.tensor(non_empty_iou_values[1:]).mean().item())
            metrics['mIoU_future_avg'] = miou_future_avg
            metrics['Non-empty_IoU_future_avg'] = non_empty_iou_future_avg
            miou_row.append(f'{miou_future_avg:.4f}')
            non_empty_iou_row.append(f'{non_empty_iou_future_avg:.4f}')

            for i in range(self.num_classes):
                values = torch.tensor(class_iou_values[i][1:])
                valid = torch.isfinite(values)
                if valid.any():
                    class_rows[i].append(f'{values[valid].mean().item():.4f}')
                else:
                    class_rows[i].append('NaN')
                    
        if self.detail:
            for i in range(self.num_classes):
                summary_table_data.append(class_rows[i])
            summary_table_data.append(['-'*10] * len(table_header))

        summary_table_data.append(miou_row)
        summary_table_data.append(non_empty_iou_row)
        
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)
                
        return metrics
