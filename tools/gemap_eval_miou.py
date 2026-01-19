#!/usr/bin/env python3
"""
GeMap mIoU-Based Evaluation Script
===================================
Evaluates GeMap predictions using rasterized semantic segmentation approach (mIoU).
Based on StreamMapNet's raster_eval.py methodology.

This script:
1. Runs GeMap inference (or loads existing predictions)
2. Rasterizes both predictions and ground truth vectors into BEV semantic maps
3. Computes IoU per class and mean IoU (mIoU)

Usage Examples:
---------------
# Front camera only with mIoU evaluation
python tools/gemap_eval_miou.py --cameras CAM_FRONT

# Front + Back cameras
python tools/gemap_eval_miou.py --cameras CAM_FRONT CAM_BACK

# All 6 cameras (baseline)
python tools/gemap_eval_miou.py --cameras all

# Skip inference if predictions exist
python tools/gemap_eval_miou.py --skip-inference --predictions-pkl existing_preds.pkl --cameras CAM_FRONT
"""

import argparse
import mmcv
import os
import torch
import warnings
import numpy as np
import pickle
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import prettytable
from copy import deepcopy

# GeMap imports
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import get_root_logger
import os.path as osp

# NuScenes and geometry utilities
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes import NuScenes
from shapely.geometry import LineString, Point, box as shapely_box
from shapely.affinity import affine_transform, rotate
import cv2

# Add GeMap project path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==================== CAMERA CONFIGURATION ====================
CAMERA_MAP = {
    'CAM_FRONT': 0,
    'CAM_FRONT_RIGHT': 1,
    'CAM_FRONT_LEFT': 2,
    'CAM_BACK': 3,
    'CAM_BACK_LEFT': 4,
    'CAM_BACK_RIGHT': 5
}

def parse_camera_config(camera_args: List[str]) -> List[int]:
    """
    Parse camera configuration from command line arguments.
    Handles both space-separated and comma-separated camera names.
    
    Args:
        camera_args: List of camera names or 'all'
                    Examples: ['CAM_FRONT', 'CAM_BACK'] or ['CAM_FRONT,CAM_BACK']
    
    Returns:
        List of camera indices (0-5)
    """
    if not camera_args or camera_args[0] == 'all':
        return list(range(6))  # All cameras
    
    # Handle comma-separated camera names
    camera_names_flat = []
    for arg in camera_args:
        if ',' in arg:
            camera_names_flat.extend([name.strip() for name in arg.split(',')])
        else:
            camera_names_flat.append(arg)
    
    camera_indices = []
    for cam_name in camera_names_flat:
        if cam_name in CAMERA_MAP:
            camera_indices.append(CAMERA_MAP[cam_name])
        else:
            print(f"Warning: Unknown camera name '{cam_name}'. Valid names: {list(CAMERA_MAP.keys())}")
    
    if not camera_indices:
        print("Warning: No valid cameras specified. Using all cameras.")
        return list(range(6))
    
    return camera_indices


# ==================== INFERENCE CODE (GeMap-specific) ====================
def run_gemap_inference(
    config_path: str,
    checkpoint_path: str,
    output_pkl: str,
    camera_indices: List[int],
    score_thresh: float = 0.0,
    samples_pkl: str = None,
    nuscenes_path: str = None
) -> str:
    """
    Run GeMap inference with specified camera configuration.
    """
    print("\n" + "="*80)
    print("STEP 1: Running GeMap Inference")
    print("="*80)
    
    cfg = Config.fromfile(config_path)
    
    # Override dataset paths if provided
    if nuscenes_path is not None:
        cfg.data.test.data_root = nuscenes_path
        print(f"Overriding NuScenes data root to: {nuscenes_path}")
    if samples_pkl is not None:
        cfg.data.test.ann_file = samples_pkl
        print(f"Overriding dataset annotation file to: {samples_pkl}")
    
    # Import plugin modules
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            plg_lib = importlib.import_module(_module_path)
    
    # Setup CUDA
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    
    # Setup logger
    logger = get_root_logger()
    logger.info('Building dataset...')
    
    # Build dataset
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info(f'Built dataset with {len(dataset)} samples')
    
    # Build model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    logger.info(f'Loading checkpoint from {checkpoint_path}...')
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    logger.info('Model loaded and ready')
    
    # Print camera configuration
    camera_names = [name for name, idx in CAMERA_MAP.items() if idx in camera_indices]
    logger.info(f'\nCamera configuration:')
    logger.info(f'  Active cameras ({len(camera_indices)}/6): {", ".join(camera_names)}')
    if len(camera_indices) < 6:
        inactive_names = [name for name, idx in CAMERA_MAP.items() if idx not in camera_indices]
        logger.info(f'  Inactive cameras (zeroed out): {", ".join(inactive_names)}')
    
    # Storage for predictions
    predictions = {}
    
    # Run inference
    logger.info('\nRunning inference...')
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        try:
            img_metas = data['img_metas'][0].data[0]
            
            # Get sample token
            if i == 0:
                logger.info(f"DEBUG: img_metas[0] keys: {list(img_metas[0].keys())}")
            
            if 'sample_idx' in img_metas[0]:
                sample_token = img_metas[0]['sample_idx']
            else:
                pts_filename = img_metas[0]['pts_filename']
                sample_token = osp.basename(pts_filename).replace('__LIDAR_TOP__', '_').split('.')[0]
            
            # Zero out inactive cameras
            if len(camera_indices) < 6 and 'img' in data and data['img'][0] is not None:
                imgs = data['img'][0].data[0]
                if i == 0:
                    logger.info(f"DEBUG: Image tensor shape: {imgs.shape}")
                
                if len(imgs.shape) == 5:  # [B, N_views, C, H, W]
                    for view_idx in range(imgs.shape[1]):
                        if view_idx not in camera_indices:
                            imgs[:, view_idx, :, :, :] = 0
                elif len(imgs.shape) == 4:  # [N_views, C, H, W]
                    for view_idx in range(imgs.shape[0]):
                        if view_idx not in camera_indices:
                            imgs[view_idx, :, :, :] = 0
            
            # Run inference
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            
            # Extract predictions
            result_dic = result[0]['pts_bbox']
            pred_boxes = result_dic['boxes_3d']
            pred_scores = result_dic['scores_3d']
            pred_labels = result_dic['labels_3d']
            pred_vectors = result_dic['pts_3d']
            
            # Filter by score threshold
            keep = pred_scores > score_thresh
            pred_vectors = pred_vectors[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]
            
            # Convert to numpy
            if torch.is_tensor(pred_vectors):
                pred_vectors = pred_vectors.cpu().numpy()
            if torch.is_tensor(pred_labels):
                pred_labels = pred_labels.cpu().numpy()
            if torch.is_tensor(pred_scores):
                pred_scores = pred_scores.cpu().numpy()
            
            # Store predictions
            predictions[sample_token] = {
                'vectors': pred_vectors,
                'labels': pred_labels,
                'scores': pred_scores
            }
            
        except Exception as e:
            logger.warning(f'Error processing sample {i}: {str(e)}')
        
        prog_bar.update()
    
    # Save predictions
    logger.info(f'\nSaving {len(predictions)} predictions to {output_pkl}...')
    with open(output_pkl, 'wb') as f:
        pickle.dump(predictions, f)
    
    logger.info('✓ Inference complete!')
    return output_pkl


# ==================== RASTERIZATION & mIoU EVALUATION ====================
class RasterMapEvaluator:
    """
    Evaluator for rasterized map using mIoU metric.
    Based on StreamMapNet's raster_eval.py approach.
    
    Args:
        nuscenes_data_path: Path to NuScenes dataset
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        canvas_size: Size of rasterized map (width, height) in pixels
        line_width: Width of rendered lines in meters (for rasterization)
        class_names: List of class names
    """
    
    def __init__(
        self,
        nuscenes_data_path: str,
        pc_range: List[float] = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        canvas_size: Tuple[int, int] = (200, 400),  # (width, height) in pixels
        line_width: float = 2.0,  # meters
        class_names: List[str] = ['divider', 'ped_crossing', 'boundary']
    ):
        self.nuscenes = NuScenes(version='v1.0-trainval', dataroot=nuscenes_data_path, verbose=False)
        self.pc_range = pc_range
        self.canvas_size = canvas_size  # (width, height)
        self.line_width = line_width
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Compute resolution (meters per pixel)
        self.x_range = pc_range[3] - pc_range[0]  # 30m
        self.y_range = pc_range[4] - pc_range[1]  # 60m
        self.x_resolution = self.x_range / canvas_size[0]  # meters per pixel
        self.y_resolution = self.y_range / canvas_size[1]  # meters per pixel
        
        # Storage for predictions and ground truths (rasterized)
        self.pred_masks = {}  # token -> [num_classes, H, W]
        self.gt_masks = {}    # token -> [num_classes, H, W]
        
        print(f"\nRasterMapEvaluator initialized:")
        print(f"  PC range: {pc_range}")
        print(f"  Canvas size: {canvas_size} (W x H)")
        print(f"  X resolution: {self.x_resolution:.3f} m/pixel")
        print(f"  Y resolution: {self.y_resolution:.3f} m/pixel")
        print(f"  Line width: {line_width} meters")
        print(f"  Classes: {class_names}")
    
    def ego_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """
        Convert ego coordinates to pixel coordinates.
        
        Args:
            points: [..., 2] array in ego coordinates (x, y)
        
        Returns:
            [..., 2] array in pixel coordinates (u, v)
        """
        # Ego: x forward, y left
        # Pixel: u horizontal (x), v vertical (y, downward)
        
        # Normalize to [0, 1]
        x_norm = (points[..., 0] - self.pc_range[0]) / self.x_range
        y_norm = (points[..., 1] - self.pc_range[1]) / self.y_range
        
        # Convert to pixel coordinates
        u = x_norm * self.canvas_size[0]
        v = y_norm * self.canvas_size[1]
        
        # Flip v axis (ego y=left is up, but pixel v=down)
        v = self.canvas_size[1] - v
        
        return np.stack([u, v], axis=-1)
    
    def rasterize_vector(self, vector: np.ndarray, class_id: int, canvas: np.ndarray):
        """
        Rasterize a single vector (polyline) onto the canvas.
        
        Args:
            vector: [N, 2] array of points in ego coordinates
            class_id: Class index (0, 1, 2 for divider, ped_crossing, boundary)
            canvas: [num_classes, H, W] boolean array to draw on (modified in-place)
        """
        if len(vector) < 2:
            return
        
        # Convert to pixel coordinates
        pixel_coords = self.ego_to_pixel(vector)
        
        # Compute line width in pixels
        line_width_pixels = int(self.line_width / min(self.x_resolution, self.y_resolution))
        line_width_pixels = max(1, line_width_pixels)
        
        # Create a temporary single-channel canvas for this vector
        temp_canvas = np.zeros((self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)
        
        # Draw polyline
        pts = pixel_coords.astype(np.int32)
        for i in range(len(pts) - 1):
            pt1 = tuple(pts[i])
            pt2 = tuple(pts[i + 1])
            cv2.line(temp_canvas, pt1, pt2, color=255, thickness=line_width_pixels)
        
        # Add to class channel
        canvas[class_id] = canvas[class_id] | (temp_canvas > 0)
    
    def rasterize_sample(
        self,
        vectors: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Rasterize all vectors for a sample.
        
        Args:
            vectors: [N, num_pts, 2] array of vectors
            labels: [N] array of class labels
        
        Returns:
            [num_classes, H, W] boolean array
        """
        canvas = np.zeros((self.num_classes, self.canvas_size[1], self.canvas_size[0]), dtype=bool)
        
        for vector, label in zip(vectors, labels):
            if label < 0 or label >= self.num_classes:
                continue
            self.rasterize_vector(vector, int(label), canvas)
        
        return canvas
    
    def load_gt_vectors(self, sample_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ground truth vectors from NuScenes for a sample.
        
        Args:
            sample_info: Sample information dictionary
        
        Returns:
            gt_vectors: [N, num_pts, 2] array
            gt_labels: [N] array
        """
        # Extract from sample_info (same as GeMap loading)
        gt_vectors = []
        gt_labels = []
        
        # Assuming sample_info has 'vectors' and 'labels' keys
        # This follows the MapTR/GeMap dataset format
        if 'vectors' in sample_info:
            vectors_data = sample_info['vectors']
            for class_id, class_vectors in enumerate(vectors_data):
                for vector in class_vectors:
                    if len(vector) > 0:
                        gt_vectors.append(vector)
                        gt_labels.append(class_id)
        
        if len(gt_vectors) == 0:
            return np.zeros((0, 2, 2)), np.zeros((0,), dtype=np.int64)
        
        # Convert to arrays
        gt_vectors = np.array(gt_vectors)
        gt_labels = np.array(gt_labels, dtype=np.int64)
        
        return gt_vectors, gt_labels
    
    def accumulate_sample(
        self,
        sample_token: str,
        sample_info: Dict,
        pred_vectors: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray = None
    ):
        """
        Rasterize and accumulate a single sample.
        
        Args:
            sample_token: NuScenes sample token
            sample_info: Sample information dictionary
            pred_vectors: [N, num_pts, 2] predicted vectors
            pred_labels: [N] predicted labels
            pred_scores: [N] prediction scores (optional, for thresholding)
        """
        # Rasterize predictions
        pred_mask = self.rasterize_sample(pred_vectors, pred_labels)
        self.pred_masks[sample_token] = pred_mask
        
        # Load and rasterize ground truth
        gt_vectors, gt_labels = self.load_gt_vectors(sample_info)
        gt_mask = self.rasterize_sample(gt_vectors, gt_labels)
        self.gt_masks[sample_token] = gt_mask
    
    def compute_iou_per_class(self) -> Dict[str, float]:
        """
        Compute IoU for each class across all samples.
        
        Returns:
            Dictionary with IoU per class and mIoU
        """
        # Stack all predictions and ground truths
        pred_masks_list = []
        gt_masks_list = []
        
        for token in self.gt_masks.keys():
            if token in self.pred_masks:
                pred_masks_list.append(self.pred_masks[token])
                gt_masks_list.append(self.gt_masks[token])
        
        if len(pred_masks_list) == 0:
            return {name: 0.0 for name in self.class_names}
        
        # Stack to [N, num_classes, H, W]
        preds = np.stack(pred_masks_list, axis=0)
        gts = np.stack(gt_masks_list, axis=0)
        
        results = {}
        total_iou = 0.0
        
        # Compute IoU for each class
        for class_id, class_name in enumerate(self.class_names):
            pred = preds[:, class_id]
            gt = gts[:, class_id]
            
            # Compute intersection and union
            intersect = (pred & gt).sum()
            union = (pred | gt).sum()
            
            # IoU
            iou = float(intersect) / (float(union) + 1e-7)
            results[class_name] = iou
            total_iou += iou
        
        # mIoU
        results['mIoU'] = total_iou / self.num_classes
        
        return results
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation and return results.
        
        Returns:
            Dictionary with IoU per class and mIoU
        """
        print(f"\nEvaluating on {len(self.gt_masks)} samples...")
        results = self.compute_iou_per_class()
        return results
    
    def print_results(self, results: Dict[str, float]):
        """
        Pretty print evaluation results.
        
        Args:
            results: Dictionary with IoU per class and mIoU
        """
        table = prettytable.PrettyTable([' ', *self.class_names, 'mean'])
        table.add_row(
            ['IoU', *[f"{results[name]:.4f}" for name in self.class_names], f"{results['mIoU']:.4f}"]
        )
        
        print("\n" + "="*80)
        print("mIoU EVALUATION RESULTS")
        print("="*80)
        print(table)
        print(f"\nmIoU = {results['mIoU']:.4f}")
        print("="*80)


# ==================== MAIN WORKFLOW ====================
def main():
    parser = argparse.ArgumentParser(
        description='GeMap mIoU-Based Evaluation (Rasterized Semantic Segmentation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Front camera only with mIoU evaluation
  python %(prog)s --cameras CAM_FRONT
  
  # Front + Back cameras
  python %(prog)s --cameras CAM_FRONT CAM_BACK
  
  # All 6 cameras (baseline)
  python %(prog)s --cameras all
  
  # Skip inference if predictions exist
  python %(prog)s --skip-inference --predictions-pkl existing_preds.pkl --cameras CAM_FRONT
        """)
    
    # Inference arguments
    parser.add_argument('--config', type=str,
                       default='/home/runw/Project/GeMap/config/gemap_full_r50_110ep.py',
                       help='GeMap config file path')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/runw/Project/GeMap/ckpts/gemap_full_r50_110ep.pth',
                       help='GeMap checkpoint file path')
    parser.add_argument('--score-thresh', type=float, default=0.0,
                       help='Score threshold for predictions (default: 0.0 = keep all)')
    
    # Camera configuration
    parser.add_argument('--cameras', type=str, nargs='+', default=['CAM_FRONT'],
                       help='Camera names to use or "all" for all 6 cameras')
    
    # Evaluation arguments
    parser.add_argument('--nuscenes-path', type=str,
                       default=None,
                       help='Path to NuScenes dataset (default: use path from config)')
    parser.add_argument('--samples-pkl', type=str,
                       default=None,
                       help='Path to samples pickle file (default: use path from config)')
    parser.add_argument('--pc-range', type=float, nargs=6,
                       default=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                       help='Point cloud range')
    parser.add_argument('--canvas-size', type=int, nargs=2,
                       default=[200, 400],
                       help='Canvas size (width, height) in pixels')
    parser.add_argument('--line-width', type=float, default=2.0,
                       help='Line width for rasterization in meters')
    
    # Output arguments
    parser.add_argument('--predictions-pkl', type=str, default=None,
                       help='Output/input pickle file for predictions')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Output JSON file for results')
    
    # Control flow
    parser.add_argument('--skip-inference', action='store_true',
                       help='Skip inference step (use existing predictions)')
    
    args = parser.parse_args()
    
    # Parse camera configuration
    camera_indices = parse_camera_config(args.cameras)
    camera_names = [name for name, idx in CAMERA_MAP.items() if idx in camera_indices]
    
    # Generate default filenames if not specified
    camera_suffix = '_'.join([name.replace('CAM_', '').lower() for name in camera_names])
    if args.predictions_pkl is None:
        args.predictions_pkl = f'gemap_predictions_miou_{camera_suffix}.pkl'
    if args.output_json is None:
        args.output_json = f'evaluation_results_miou_{camera_suffix}.json'
    
    print("\n" + "="*80)
    print("GeMap mIoU-BASED EVALUATION PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Camera configuration: {camera_names} ({len(camera_names)}/6 cameras)")
    print(f"  Evaluation method: mIoU (rasterized semantic segmentation)")
    print(f"  Canvas size: {args.canvas_size}")
    print(f"  Line width: {args.line_width}m")
    print(f"  Predictions file: {args.predictions_pkl}")
    print(f"  Output file: {args.output_json}")
    
    # STEP 1: Run inference (unless skipped)
    if not args.skip_inference:
        predictions_pkl = run_gemap_inference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_pkl=args.predictions_pkl,
            camera_indices=camera_indices,
            score_thresh=args.score_thresh,
            samples_pkl=args.samples_pkl,
            nuscenes_path=args.nuscenes_path
        )
    else:
        print("\n" + "="*80)
        print("STEP 1: Skipping Inference (using existing predictions)")
        print("="*80)
        predictions_pkl = args.predictions_pkl
        if not os.path.exists(predictions_pkl):
            raise FileNotFoundError(f"Predictions file not found: {predictions_pkl}")
        print(f"Using predictions from: {predictions_pkl}")
    
    # STEP 2: Load samples and predictions
    print("\n" + "="*80)
    print("STEP 2: Loading Data")
    print("="*80)
    
    # Get actual samples path
    if args.samples_pkl:
        samples_path = args.samples_pkl
    else:
        cfg = Config.fromfile(args.config)
        samples_path = cfg.data.test.ann_file
    
    print(f"Loading samples from {samples_path}...")
    with open(samples_path, 'rb') as f:
        samples_data = pickle.load(f)
    samples = samples_data['infos']
    print(f"✓ Loaded {len(samples)} samples")
    
    print(f"\nLoading predictions from {predictions_pkl}...")
    with open(predictions_pkl, 'rb') as f:
        predictions_by_token = pickle.load(f)
    print(f"✓ Loaded predictions for {len(predictions_by_token)} samples")
    
    # STEP 3: Rasterize and evaluate
    print("\n" + "="*80)
    print("STEP 3: Rasterizing and Computing mIoU")
    print("="*80)
    
    # Get actual NuScenes path
    if args.nuscenes_path:
        nuscenes_eval_path = args.nuscenes_path
    else:
        cfg = Config.fromfile(args.config)
        nuscenes_eval_path = cfg.data.test.data_root
    
    # Create evaluator
    evaluator = RasterMapEvaluator(
        nuscenes_data_path=nuscenes_eval_path,
        pc_range=args.pc_range,
        canvas_size=tuple(args.canvas_size),
        line_width=args.line_width
    )
    
    # Accumulate predictions and GT
    print(f"\nProcessing {len(samples)} samples...")
    for sample_info in tqdm(samples, desc="Rasterizing"):
        sample_token = sample_info['token']
        
        if sample_token not in predictions_by_token:
            # Use empty predictions if not found
            pred_vectors = np.zeros((0, 2, 2))
            pred_labels = np.zeros((0,), dtype=np.int64)
            pred_scores = np.zeros((0,))
        else:
            pred_data = predictions_by_token[sample_token]
            pred_vectors = pred_data['vectors']
            pred_labels = pred_data['labels']
            pred_scores = pred_data['scores']
        
        evaluator.accumulate_sample(
            sample_token=sample_token,
            sample_info=sample_info,
            pred_vectors=pred_vectors,
            pred_labels=pred_labels,
            pred_scores=pred_scores
        )
    
    # Compute metrics
    results = evaluator.evaluate()
    
    # STEP 4: Print and save results
    evaluator.print_results(results)
    
    # Save results
    print(f"\nSaving results to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nFiles generated:")
    print(f"  - Predictions: {args.predictions_pkl}")
    print(f"  - Results: {args.output_json}")
    print()


if __name__ == '__main__':
    main()
