
import torch
from mmengine.evaluator import BaseMetric
from mmengine.dist import (broadcast_object_list, get_rank, get_world_size,
                           is_main_process)
from mmengine.logging import MMLogger
from mmengine.logging import print_log

from mmdet3d.registry import METRICS
from terminaltables import AsciiTable
from projects.visualizer.occ_visualizer import OccBEVVisualizer
from .visualization import should_visualize, visualization_save_name

@METRICS.register_module()
class Occ3DMetric(BaseMetric):
    def __init__(
            self, 
        num_classes=18, 
        empty_cls=17, 
        ignore_index=255, 
        label2cat=None, 
        collect_device='cpu', 
        prefix=None, 
        vis_interval=0, 
        vis_dir='work_dirs/vis_occ'
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.empty_cls = empty_cls
        self.vis_interval = vis_interval
        self.step = 0
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
        
        self.label2cat = label2cat if label2cat is not None else default_categories
        self._hist = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.float64)
        self._pending_hist = None
        self._processed_count = 0

    def _commit_pending(self):
        if self._pending_hist is not None:
            self._hist += self._pending_hist
            self._pending_hist = None

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

    def process(self, data_batchs, data_samples):
        """
            Process one batch of data samples and predictions.
            Accumulate compact streaming state. The newest sample is kept
            pending until the next sample arrives, so distributed sampler
            padding at each rank tail can be dropped in ``evaluate``.
        """
        batch_data_samples = data_batchs.get('data_samples', [])
        for sample_idx, data_sample in enumerate(data_samples):
            self._commit_pending()
            batch_data_sample = (
                batch_data_samples[sample_idx]
                if sample_idx < len(batch_data_samples) else None)
            pred_voxel_semantics = data_sample['pred_voxel_semantics']  # [Y, X, Z]
            targ_voxel_semantics = data_sample['targ_voxel_semantics']  # [Y, X, Z]

            pred = pred_voxel_semantics.flatten().cpu()
            targ = targ_voxel_semantics.flatten().cpu()

            # Mask out ignore_index and invalid labels
            mask = (targ != self.ignore_index) & (targ >= 0) & (targ < self.num_classes)
            pred = pred[mask]
            targ = targ[mask]

            valid_pred_mask = pred < self.num_classes
            pred = pred[valid_pred_mask]
            targ = targ[valid_pred_mask]

            idx = targ * self.num_classes + pred
            bincount = torch.bincount(idx, minlength=self.num_classes**2)
            self._pending_hist = bincount.reshape(
                self.num_classes, self.num_classes).to(torch.float64)
            self._processed_count += 1

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
        valid_count = self._valid_local_count(size)
        if valid_count == self._processed_count:
            self._commit_pending()
        elif valid_count != self._processed_count - 1:
            raise RuntimeError(
                f'Unexpected distributed eval padding state: '
                f'processed={self._processed_count}, valid={valid_count}')

        rank_results = [dict(hist=self._hist, num_samples=valid_count)]
        if get_world_size() > 1:
            gathered = [None for _ in range(get_world_size())]
            torch.distributed.all_gather_object(gathered, rank_results[0])
            results = gathered if is_main_process() else None
        else:
            results = rank_results

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
        self.step = 0
        if hasattr(self, 'visualizer'):
            self.visualizer.step = 0
        return metrics_list[0]

    def compute_metrics(self, results):
        """
            Compute the metrics from processed results.
            Occ3DMetric computes:
            1. Per-class IoU for all classes.
            2. Mean IoU (mIoU) across all classes that appear in GT, excluding empty_cls.
            3. Non-empty (Occupied) IoU treating all non-empty classes as one and empty class as the other.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        logger.info('----------- Computing Occupancy 3D Metric ---------------')
        logger.info(f'Eval number of Samples: {sum(res["num_samples"] for res in results)}')

        hist = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.float64)
        for res in results:
            hist += res['hist'].cpu()
            
        eps = 1e-6
        intersect = torch.diag(hist)
        union = hist.sum(dim=1) + hist.sum(dim=0) - intersect
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
        
        tp = hist[non_empty_mask][:, non_empty_mask].sum().item()
        fp = hist[empty_mask][:, non_empty_mask].sum().item() if empty_mask.any() else 0.0
        fn = hist[non_empty_mask][:, empty_mask].sum().item() if empty_mask.any() else 0.0
        non_empty_iou = tp / (tp + fp + fn + eps)
        
        metrics = {'mIoU': miou, 'Non-empty_IoU': non_empty_iou}
        
        table_data = [['Class', 'IoU']]
        for i in range(self.num_classes):
            if isinstance(self.label2cat, list):
                class_name = self.label2cat[i] if i < len(self.label2cat) else f'class_{i}'
            else:
                class_name = self.label2cat.get(i, self.label2cat.get(str(i), f'class_{i}'))
                
            if valid_classes[i]:
                metrics[class_name] = ious[i].item()
                table_data.append([class_name, f'{ious[i].item():.4f}'])
            else:
                metrics[class_name] = float('nan')
                table_data.append([class_name, 'NaN'])
        
        # Add Line for mIoU and Non-empty IoU
        table_data.append(['-'*10, '-'*10])
        table_data.append(['mIoU', f'{miou:.4f}'])
        table_data.append(['-'*10, '-'*10])
        table_data.append(['Non-empty IoU', f'{non_empty_iou:.4f}'])
        
        table = AsciiTable(table_data)
        print_log('\n' + table.table, logger=logger)
                
        return metrics
