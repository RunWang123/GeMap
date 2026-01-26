#!/usr/bin/env python3
"""
Unified GeMap Evaluation Script with NOISE in Camera Parameters
================================================================
Runs GeMap inference with configurable camera inputs AND evaluates with camera-specific FOV clipping.

**KEY MODIFICATION**: Adds Gaussian NOISE to camera extrinsics to test robustness.

This script adds controlled noise to camera2ego parameters to test how GeMap performs 
with imperfect camera calibration, following the robustness experiments from MapTR paper:
- Translation noise: Gaussian noise with std σ₁ (meters) added to [Δx, Δy, Δz]
- Rotation noise: Gaussian noise with std σ₂ (radians) added to [θx, θy, θz]

Usage Examples:
---------------
# Translation noise σ = 0.1m
python tools/gemap_eval_unified_noise.py --cameras CAM_FRONT --noise-type translation --noise-std 0.1

# Rotation noise σ = 0.01 rad
python tools/gemap_eval_unified_noise.py --cameras CAM_FRONT --noise-type rotation --noise-std 0.01

# No noise (baseline)
python tools/gemap_eval_unified_noise.py --cameras CAM_FRONT --noise-std 0
for std in 0 0.05 0.1 0.5 1.0; do
  python tools/gemap_eval_unified_noise.py \
    --cameras all --noise-type translation --noise-std $std
done

# Full MapTR Table 12 replication (rotation)
for std in 0 0.005 0.01 0.02 0.05; do
  python tools/gemap_eval_unified_noise.py \
    --cameras all --noise-type rotation --noise-std $std
done

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
from multiprocessing import Pool, cpu_count
from functools import partial

# Add GeMap project path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
from shapely.geometry import LineString, Point, CAP_STYLE, JOIN_STYLE
from shapely.strtree import STRtree
from scipy.spatial.transform import Rotation
from scipy.spatial import distance

# Import shared camera FOV utilities
from camera_fov_utils import (
    VectorizedLocalMap,
    CameraFOVClipper,
    extract_gt_vectors,
    extract_gt_with_fov_clipping,
    process_predictions_with_fov_clipping
)

# Add GeMap project path
# Import MapTR's Chamfer distance implementation (GeMap uses same evaluation)
try:
    from projects.mmdet3d_plugin.datasets.map_utils.tpfp_chamfer import custom_polyline_score
except ImportError:
    print("Warning: Could not import GeMap's custom_polyline_score. Using fallback implementation.")
    custom_polyline_score = None


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
    # Split any comma-separated arguments into individual camera names
    camera_names_flat = []
    for arg in camera_args:
        if ',' in arg:
            # Split on comma and strip whitespace
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


def add_noise_to_camera_extrinsics(
    img_metas_list: List[Dict], 
    noise_trans_std: float = 0.0,
    noise_rot_std: float = 0.0,
    active_camera_indices: List[int] = None,
    seed: int = None,
    logger=None
) -> List[Dict]:
    """
    Add Gaussian noise to camera extrinsics to test robustness (following MapTR paper).
    Can apply translation and rotation noise simultaneously.
    
    Noise Types:
    - Translation: Add N(0, σ²) noise to camera translation [Δx, Δy, Δz]
    - Rotation: Add N(0, σ²) noise to rotation angles [θx, θy, θz] in radians
    
    Args:
        img_metas_list: List of img_metas dicts, one per timestamp in sequence
        noise_trans_std: Standard deviation of Gaussian noise for translation (meters)
        noise_rot_std: Standard deviation of Gaussian noise for rotation (radians)
        active_camera_indices: Cameras to add noise to (default: all 6)
        seed: Random seed for reproducibility (default: None)
        logger: Optional logger
    
    Returns:
        Modified img_metas_list with noisy camera extrinsics
    """
    if noise_trans_std == 0 and noise_rot_std == 0:
        if logger:
            logger.info("Noise stds are 0, skipping noise addition")
        return img_metas_list
    
    if active_camera_indices is None:
        active_camera_indices = list(range(6))
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    if logger:
        logger.info("\n" + "="*80)
        logger.info(f"Adding noise to camera extrinsics")
        if noise_trans_std > 0:
            logger.info(f"Translation Noise std (σ): {noise_trans_std} meters")
        if noise_rot_std > 0:
            logger.info(f"Rotation Noise std (σ): {noise_rot_std} radians")
        logger.info(f"Active cameras: {[list(CAMERA_MAP.keys())[i] for i in active_camera_indices]}")
        logger.info("="*80)
    
    for t_idx, img_meta in enumerate(img_metas_list):
        if 'camera2ego' not in img_meta:
            if logger and t_idx == 0:
                logger.warning("'camera2ego' not found in img_metas. Skipping noise addition.")
            continue
        
        camera2ego_list = img_meta['camera2ego']
        if not isinstance(camera2ego_list, list) or len(camera2ego_list) != 6:
            if logger and t_idx == 0:
                logger.warning(f"Expected 6 camera2ego matrices, got {len(camera2ego_list) if isinstance(camera2ego_list, list) else 'non-list'}")
            continue
        
        cam_names = list(CAMERA_MAP.keys())
        
        # Add noise to each active camera
        for cam_idx in active_camera_indices:
            cam2ego = camera2ego_list[cam_idx]
            
            # Convert to numpy for manipulation
            if isinstance(cam2ego, np.ndarray):
                cam2ego_np = cam2ego.copy()
            elif torch.is_tensor(cam2ego):
                cam2ego_np = cam2ego.cpu().numpy().copy()
            else:
                cam2ego_np = np.array(cam2ego).copy()
            
            if noise_trans_std > 0:
                # Add Gaussian noise to translation components
                original_trans = cam2ego_np[:3, 3].copy()
                noise = np.random.normal(0, noise_trans_std, size=3)
                noisy_trans = original_trans + noise
                cam2ego_np[:3, 3] = noisy_trans
                
                if logger and t_idx == 0:
                    logger.info(f"\n{cam_names[cam_idx]} (Translation):")
                    logger.info(f"  Original: [{original_trans[0]:6.3f}, {original_trans[1]:6.3f}, {original_trans[2]:6.3f}]")
                    logger.info(f"  Noise:    [{noise[0]:6.3f}, {noise[1]:6.3f}, {noise[2]:6.3f}]")
                    logger.info(f"  Noisy:    [{noisy_trans[0]:6.3f}, {noisy_trans[1]:6.3f}, {noisy_trans[2]:6.3f}]")
            
            if noise_rot_std > 0:
                # Add Gaussian noise to rotation matrix via small angle perturbations
                # Extract rotation matrix
                original_rot = cam2ego_np[:3, :3].copy()
                
                # Generate small angle perturbations (in radians)
                delta_angles = np.random.normal(0, noise_rot_std, size=3)  # [θx, θy, θz]
                
                # Create rotation perturbation matrices for each axis
                # Rx(θx)
                rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(delta_angles[0]), -np.sin(delta_angles[0])],
                    [0, np.sin(delta_angles[0]), np.cos(delta_angles[0])]
                ])
                # Ry(θy)
                ry = np.array([
                    [np.cos(delta_angles[1]), 0, np.sin(delta_angles[1])],
                    [0, 1, 0],
                    [-np.sin(delta_angles[1]), 0, np.cos(delta_angles[1])]
                ])
                # Rz(θz)
                rz = np.array([
                    [np.cos(delta_angles[2]), -np.sin(delta_angles[2]), 0],
                    [np.sin(delta_angles[2]), np.cos(delta_angles[2]), 0],
                    [0, 0, 1]
                ])
                
                # Apply perturbations: R_noisy = Rz * Ry * Rx * R_original
                rotation_perturbation = rz @ ry @ rx
                noisy_rot = rotation_perturbation @ original_rot
                cam2ego_np[:3, :3] = noisy_rot
                
                if logger and t_idx == 0:
                    # Convert to Euler angles for interpretability
                    from scipy.spatial.transform import Rotation as R
                    original_euler = R.from_matrix(original_rot).as_euler('xyz', degrees=True)
                    noisy_euler = R.from_matrix(noisy_rot).as_euler('xyz', degrees=True)
                    logger.info(f"\n{cam_names[cam_idx]} (Rotation):")
                    logger.info(f"  Original (deg): [{original_euler[0]:6.2f}, {original_euler[1]:6.2f}, {original_euler[2]:6.2f}]")
                    logger.info(f"  Noise (rad):    [{delta_angles[0]:6.4f}, {delta_angles[1]:6.4f}, {delta_angles[2]:6.4f}]")
                    logger.info(f"  Noisy (deg):    [{noisy_euler[0]:6.2f}, {noisy_euler[1]:6.2f}, {noisy_euler[2]:6.2f}]")
            
            # Update camera2ego matrix
            if isinstance(camera2ego_list[cam_idx], np.ndarray):
                camera2ego_list[cam_idx] = cam2ego_np
            elif torch.is_tensor(camera2ego_list[cam_idx]):
                camera2ego_list[cam_idx] = torch.from_numpy(cam2ego_np).to(
                    camera2ego_list[cam_idx].device
                ).to(camera2ego_list[cam_idx].dtype)
            else:
                camera2ego_list[cam_idx] = cam2ego_np
        
        # Recompute lidar2img matrices after adding noise
        if 'lidar2img' in img_meta and 'camera_intrinsics' in img_meta and 'lidar2ego' in img_meta:
            lidar2ego = img_meta['lidar2ego']
            if isinstance(lidar2ego, np.ndarray):
                lidar2ego_mat = lidar2ego
            elif torch.is_tensor(lidar2ego):
                lidar2ego_mat = lidar2ego.cpu().numpy()
            else:
                lidar2ego_mat = np.array(lidar2ego)
            
            for cam_idx in active_camera_indices:
                cam2ego = camera2ego_list[cam_idx]
                if isinstance(cam2ego, np.ndarray):
                    cam2ego_mat = cam2ego
                elif torch.is_tensor(cam2ego):
                    cam2ego_mat = cam2ego.cpu().numpy()
                else:
                    cam2ego_mat = np.array(cam2ego)
                
                # Get intrinsics
                cam_intrinsic = img_meta['camera_intrinsics'][cam_idx]
                if isinstance(cam_intrinsic, np.ndarray):
                    intrinsic_mat = cam_intrinsic
                elif torch.is_tensor(cam_intrinsic):
                    intrinsic_mat = cam_intrinsic.cpu().numpy()
                else:
                    intrinsic_mat = np.array(cam_intrinsic)
                
                # Recompute lidar2img
                ego2cam = np.linalg.inv(cam2ego_mat)
                lidar2cam = ego2cam @ lidar2ego_mat
                new_lidar2img = intrinsic_mat @ lidar2cam
                
                # Update lidar2img
                if isinstance(img_meta['lidar2img'][cam_idx], np.ndarray):
                    img_meta['lidar2img'][cam_idx] = new_lidar2img
                elif torch.is_tensor(img_meta['lidar2img'][cam_idx]):
                    img_meta['lidar2img'][cam_idx] = torch.from_numpy(new_lidar2img).to(
                        img_meta['lidar2img'][cam_idx].device
                    ).to(img_meta['lidar2img'][cam_idx].dtype)
    
    if logger:
        logger.info("\n✓ Noise addition complete")
        logger.info("="*80 + "\n")
    
    return img_metas_list


# ==================== INFERENCE CODE (GeMap-specific) ====================
def run_gemap_inference(
    config_path: str,
    checkpoint_path: str,
    output_pkl: str,
    camera_indices: List[int],
    score_thresh: float = 0.0,
    samples_pkl: str = None,
    nuscenes_path: str = None,
    noise_trans_std: float = 0.0,
    noise_rot_std: float = 0.0,
    noise_seed: int = None
) -> str:
    """
    Run GeMap inference with specified camera configuration and noise.
    
    Args:
        noise_trans_std: Standard deviation of Gaussian noise for translation
        noise_rot_std: Standard deviation of Gaussian noise for rotation
        noise_seed: Random seed for reproducibility
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
            
            # ADD NOISE TO CAMERA EXTRINSICS (following MapTR robustness experiments)
            # Apply noise to test robustness to calibration errors
            if noise_trans_std > 0 or noise_rot_std > 0:
                img_metas = add_noise_to_camera_extrinsics(
                    img_metas if isinstance(img_metas, list) else [img_metas],
                    noise_trans_std=noise_trans_std,
                    noise_rot_std=noise_rot_std,
                    active_camera_indices=camera_indices,
                    seed=noise_seed + i if noise_seed is not None else None,  # Different noise per sample
                    logger=logger if i == 0 else None  # Only log first sample
                )
                if not isinstance(data['img_metas'][0].data[0], list):
                    img_metas = img_metas[0]  # Unwrap if originally not a list
            
            # Get sample token - use the actual NuScenes sample_token if available
            # Debug: print available keys on first sample
            if i == 0:
                logger.info(f"DEBUG: img_metas[0] keys: {list(img_metas[0].keys()) if isinstance(img_metas, list) else list(img_metas.keys())}")
            
            # Handle both list and single dict cases
            first_meta = img_metas[0] if isinstance(img_metas, list) else img_metas
            
            if 'sample_idx' in first_meta:
                sample_token = first_meta['sample_idx']  # This might be the token
            else:
                # Fallback to pts_filename
                pts_filename = first_meta['pts_filename']
                sample_token = osp.basename(pts_filename).replace('__LIDAR_TOP__', '_').split('.')[0]
            
            # Zero out all camera views except specified cameras using in-place modification
            # NuScenes camera order: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
            #                        CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
            if len(camera_indices) < 6 and 'img' in data and data['img'][0] is not None:
                imgs = data['img'][0].data[0]  # Shape: [B, N_views, C, H, W] or [N_views, C, H, W]
                if i == 0:
                    logger.info(f"DEBUG: Image tensor shape: {imgs.shape}")
                
                # Zero out inactive cameras using in-place modification (matches original approach)
                if len(imgs.shape) == 5:  # [B, N_views, C, H, W]
                    for view_idx in range(imgs.shape[1]):
                        if view_idx not in camera_indices:
                            imgs[:, view_idx, :, :, :] = 0
                elif len(imgs.shape) == 4:  # [N_views, C, H, W]
                    for view_idx in range(imgs.shape[0]):
                        if view_idx not in camera_indices:
                            imgs[view_idx, :, :, :] = 0
                else:
                    logger.warning(f"Unexpected image tensor shape: {imgs.shape}")
            
            # Run inference
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            
            # Extract predictions (EXACT format from MapTR)
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


# ==================== PARALLEL GEOMETRY BUFFERING WORKER ====================
def buffer_geometries_worker(pred_vectors: np.ndarray, gt_vectors: np.ndarray, linewidth: float = 2.0):
    """
    Worker function to buffer geometries for one sample in parallel.
    Returns buffered shapely geometries for predictions and GT.
    """
    from shapely.geometry import LineString
    from shapely.geometry import CAP_STYLE, JOIN_STYLE
    
    def buffer_vectors(vectors):
        if len(vectors) == 0:
            return []
        geometries = []
        for vec in vectors:
            if len(vec) >= 2:
                line = LineString(vec)
                buffered = line.buffer(linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                geometries.append(buffered)
            else:
                geometries.append(None)
        return geometries
    
    pred_geoms = buffer_vectors(pred_vectors)
    gt_geoms = buffer_vectors(gt_vectors)
    
    return pred_geoms, gt_geoms


# ==================== PARALLEL PROCESSING WORKER ====================
def process_sample_worker(
    sample_info: Dict,
    pred_data: Dict,
    nuscenes_data_path: str,
    pc_range: List[float],
    num_sample_pts: int,
    camera_names: List[str],
    apply_clipping: bool
) -> Dict:
    """
    Worker function to process a single sample in parallel.
    Returns processed GT and predictions for all cameras.
    
    This function is picklable and can be used with multiprocessing.Pool
    """
    from camera_fov_utils import extract_gt_with_fov_clipping, process_predictions_with_fov_clipping
    from shapely.geometry import LineString
    
    def resample_vector(vector: np.ndarray, num_sample: int) -> np.ndarray:
        """Resample vector to fixed number of points"""
        if len(vector) < 2:
            if num_sample > len(vector):
                padding = np.zeros((num_sample - len(vector), 2))
                return np.vstack([vector, padding])
            return vector
        
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_sample)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
                                   for distance in distances]).reshape(-1, 2)
        return sampled_points
    
    results_per_camera = {}
    
    for camera_name in camera_names:
        # Process GT
        gt_data = extract_gt_with_fov_clipping(
            sample_info=sample_info,
            nuscenes_path=nuscenes_data_path,
            pc_range=pc_range,
            camera_name=camera_name,
            fixed_num=20,
            apply_clipping=apply_clipping
        )
        
        gt_vectors = gt_data['vectors']
        gt_labels = gt_data['labels']
        
        # Resample GT to num_sample_pts
        if len(gt_vectors) > 0:
            final_gt_vectors = []
            for vector in gt_vectors:
                if len(vector) >= 2:
                    resampled_vec = resample_vector(vector, num_sample_pts)
                    final_gt_vectors.append(resampled_vec)
            gt_vectors_array = np.array(final_gt_vectors) if final_gt_vectors else np.array([])
            gt_labels_array = np.array(gt_labels) if final_gt_vectors else np.array([])
        else:
            gt_vectors_array = np.array([])
            gt_labels_array = np.array([])
        
        # Process predictions
        pred_vectors_processed, pred_labels_processed, pred_scores_processed = \
            process_predictions_with_fov_clipping(
                pred_vectors=pred_data['vectors'],
                pred_labels=pred_data['labels'],
                pred_scores=pred_data['scores'],
                sample_info=sample_info,
                nuscenes_path=nuscenes_data_path,
                pc_range=pc_range,
                camera_name=camera_name,
                apply_clipping=apply_clipping
            )
        
        # Resample predictions to num_sample_pts
        if len(pred_vectors_processed) > 0:
            final_pred_vectors = []
            final_pred_labels = []
            final_pred_scores = []
            for vec, label, score in zip(pred_vectors_processed, pred_labels_processed, pred_scores_processed):
                if len(vec) >= 2:
                    resampled_vec = resample_vector(vec, num_sample_pts)
                    final_pred_vectors.append(resampled_vec)
                    final_pred_labels.append(label)
                    final_pred_scores.append(score)
            
            pred_vectors_array = np.array(final_pred_vectors) if final_pred_vectors else np.array([])
            pred_labels_array = np.array(final_pred_labels) if final_pred_labels else np.array([])
            pred_scores_array = np.array(final_pred_scores) if final_pred_labels else np.array([])
        else:
            pred_vectors_array = np.array([])
            pred_labels_array = np.array([])
            pred_scores_array = np.array([])
        
        results_per_camera[camera_name] = {
            'gt': {
                'vectors': gt_vectors_array,
                'labels': gt_labels_array
            },
            'pred': {
                'vectors': pred_vectors_array,
                'labels': pred_labels_array,
                'scores': pred_scores_array
            }
        }
    
    return results_per_camera


# ==================== EVALUATION CODE (EXACT COPY FROM evaluate_with_fov_clipping_standalone.py) ====================
class CameraSpecificEvaluator:
    """
    Evaluator using EXACT MapTR official evaluation method.
    EXACT COPY from evaluate_with_fov_clipping_standalone.py
    
    Applies camera-specific FOV clipping and rotation to both GT and predictions,
    then evaluates using MapTR's official matching algorithm.
    """
    
    def __init__(
        self,
        nuscenes_data_path: str,
        pc_range: List[float] = None,
        num_sample_pts: int = 100,
        thresholds_chamfer: List[float] = None,
        camera_names: List[str] = None,
        num_workers: int = 1
    ):
        """
        Args:
            nuscenes_data_path: Path to NuScenes dataset
            pc_range: BEV range [-x, -y, -z, x, y, z]
            num_sample_pts: Number of points to resample vectors to (MUST match training: 100)
            thresholds_chamfer: Chamfer distance thresholds (MapTR uses [0.5, 1.0, 1.5])
            camera_names: List of camera names to evaluate
        """
        self.nuscenes_data_path = nuscenes_data_path
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.num_sample_pts: int = num_sample_pts
        self.thresholds_chamfer = thresholds_chamfer or [0.5, 1.0, 1.5]
        self.camera_names = camera_names or ['CAM_FRONT']
        self.num_workers = num_workers # NEW: Store num_workers
        
        # Calculate patch size from pc_range
        self.patch_size = (self.pc_range[4] - self.pc_range[1], self.pc_range[3] - self.pc_range[0])
        
        # Accumulators
        self.reset()
    
    def reset(self):
        """Reset accumulators"""
        self.predictions_per_camera = {cam: [] for cam in self.camera_names}
        self.ground_truths_per_camera = {cam: [] for cam in self.camera_names}
        self.num_samples_processed = 0
    
    def resample_vector_linestring(self, vector: np.ndarray, num_sample: int) -> np.ndarray:
        """
        Resample a vector to fixed number of points using LineString interpolation.
        EXACT match to MapTR's implementation.
        """
        if len(vector) < 2:
            if num_sample > len(vector):
                padding = np.zeros((num_sample - len(vector), 2))
                return np.vstack([vector, padding])
            return vector
        
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_sample)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
                                   for distance in distances]).reshape(-1, 2)
        
        return sampled_points
    
    def process_gt_with_fov_clipping(
        self,
        sample_info: Dict,
        camera_name: str,
        apply_clipping: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract and process GT vectors for a specific camera.
        Uses shared extract_gt_with_fov_clipping() for 100% identical logic.
        """
        gt_data = extract_gt_with_fov_clipping(
            sample_info=sample_info,
            nuscenes_path=self.nuscenes_data_path,
            pc_range=self.pc_range,
            camera_name=camera_name,
            fixed_num=20,
            apply_clipping=apply_clipping
        )
        
        vectors = gt_data['vectors']
        gt_labels = gt_data['labels']
        
        if len(vectors) == 0:
            return np.array([]), np.array([])
        
        # Resample to num_sample_pts (100) for evaluation
        final_vectors = []
        for vector in vectors:
            if len(vector) >= 2:
                resampled_vec = self.resample_vector_linestring(vector, self.num_sample_pts)
                final_vectors.append(resampled_vec)
        
        if len(final_vectors) == 0:
            return np.array([]), np.array([])
        
        return np.array(final_vectors), np.array(gt_labels)
    
    def process_predictions_with_fov_clipping_and_rotation(
        self,
        pred_vectors: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        sample_info: Dict,
        camera_name: str,
        apply_clipping: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply optional FOV clipping AND camera-centric rotation to predictions.
        Uses shared process_predictions_with_fov_clipping() for 100% identical logic.
        """
        if len(pred_vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        vectors, labels, scores = process_predictions_with_fov_clipping(
            pred_vectors=pred_vectors,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            sample_info=sample_info,
            nuscenes_path=self.nuscenes_data_path,
            pc_range=self.pc_range,
            camera_name=camera_name,
            apply_clipping=apply_clipping
        )
        
        if len(vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Resample to num_sample_pts (100) for evaluation
        final_vectors = []
        for vector in vectors:
            if len(vector) >= 2:
                resampled_vec = self.resample_vector_linestring(vector, self.num_sample_pts)
                final_vectors.append(resampled_vec)
        
        if len(final_vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return np.array(final_vectors), np.array(labels), np.array(scores)
    
    def compute_chamfer_distance_matrix_maptr_official(self,
                                                        pred_vectors: np.ndarray,
                                                        gt_vectors: np.ndarray,
                                                        linewidth: float = 2.0,
                                                        pred_geometries: List = None,
                                                        gt_geometries: List = None) -> np.ndarray:
        """
        Compute Chamfer Distance matrix using EXACT MapTR official method.
        EXACT copy from MapTR's tpfp_chamfer.py:custom_polyline_score()
        
        OPTIMIZED: Accepts pre-computed geometries to avoid redundant buffering.
        
        Returns NEGATIVE CD values (higher = better match).
        """
        num_preds = len(pred_vectors)
        num_gts = len(gt_vectors)
        
        if num_preds == 0 or num_gts == 0:
            return np.full((num_preds, num_gts), -100.0)
        
        # Use pre-computed geometries if provided, otherwise create them
        if pred_geometries is None:
            pred_lines_shapely = [
                LineString(pred_vectors[i]).buffer(
                    linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                for i in range(num_preds)
            ]
        else:
            pred_lines_shapely = pred_geometries
        
        if gt_geometries is None:
            gt_lines_shapely = [
                LineString(gt_vectors[i]).buffer(
                    linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
                for i in range(num_gts)
            ]
        else:
            gt_lines_shapely = gt_geometries
        
        # STRtree spatial indexing
        tree = STRtree(pred_lines_shapely)
        
        # Initialize with -100.0 for non-intersecting pairs
        cd_matrix = np.full((num_preds, num_gts), -100.0)
        
        # Compute CD only for intersecting buffered geometries
        for i, gt_line in enumerate(gt_lines_shapely):
            query_result = tree.query(gt_line)
            
            # Handle both Shapely 1.x and 2.x
            if len(query_result) > 0 and isinstance(query_result[0], (int, np.integer)):
                # Shapely 2.x: returns indices
                for pred_idx in query_result:
                    pred_line = pred_lines_shapely[pred_idx]
                    
                    if pred_line.intersects(gt_line):
                        dist_mat = distance.cdist(
                            pred_vectors[pred_idx], gt_vectors[i], 'euclidean')
                        valid_ab = dist_mat.min(axis=1).mean()
                        valid_ba = dist_mat.min(axis=0).mean()
                        cd_matrix[pred_idx, i] = -(valid_ab + valid_ba) / 2.0
            else:
                # Shapely 1.x: returns geometries
                for pred_idx in range(num_preds):
                    pred_line = pred_lines_shapely[pred_idx]
                    
                    if pred_line.intersects(gt_line):
                        dist_mat = distance.cdist(
                            pred_vectors[pred_idx], gt_vectors[i], 'euclidean')
                        valid_ab = dist_mat.min(axis=1).mean()
                        valid_ba = dist_mat.min(axis=0).mean()
                        cd_matrix[pred_idx, i] = -(valid_ab + valid_ba) / 2.0
        
        return cd_matrix
    
    def precompute_shapely_geometries(self,
                                     vectors: np.ndarray,
                                     linewidth: float = 2.0) -> List:
        """
        Pre-compute buffered Shapely geometries for a set of vectors.
        This is a performance optimization to avoid recomputing geometries
        for each threshold evaluation.
        
        Args:
            vectors: Array of shape (N, num_points, 2)
            linewidth: Buffer width for LineString
        
        Returns:
            List of buffered Shapely polygons
        """
        if len(vectors) == 0:
            return []
        
        geometries = [
            LineString(vectors[i]).buffer(
                linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
            for i in range(len(vectors))
        ]
        return geometries
    
    def compute_chamfer_distance_torch(self,
                                       pred_vectors: np.ndarray,
                                       gt_vectors: np.ndarray) -> float:
        """
        Compute Chamfer Distance for monitoring (returns POSITIVE distance).
        """
        if len(pred_vectors) == 0 or len(gt_vectors) == 0:
            return float('inf')
        
        pred_points = pred_vectors.reshape(-1, 2)
        gt_points = gt_vectors.reshape(-1, 2)
        
        dist_matrix = distance.cdist(pred_points, gt_points, 'euclidean')
        
        valid_ab = dist_matrix.min(axis=1).mean()
        valid_ba = dist_matrix.min(axis=0).mean()
        
        chamfer_dist = (valid_ab + valid_ba) / 2.0
        
        return chamfer_dist
    
    def accumulate_sample(
        self,
        sample_info: Dict,
        pred_vectors: np.ndarray,
        pred_labels: np.ndarray,
        pred_scores: np.ndarray,
        apply_clipping: bool = True
    ):
        """
        Process one sample and accumulate results for each camera.
        """
        for camera_name in self.camera_names:
            # Process GT with optional FOV clipping
            gt_vectors, gt_labels = self.process_gt_with_fov_clipping(
                sample_info, camera_name, apply_clipping=apply_clipping)
            
            # Process predictions with optional FOV clipping
            pred_vectors_clipped, pred_labels_clipped, pred_scores_clipped = \
                self.process_predictions_with_fov_clipping_and_rotation(
                    pred_vectors, pred_labels, pred_scores, sample_info, camera_name,
                    apply_clipping=apply_clipping)
            
            # Store for this camera
            self.predictions_per_camera[camera_name].append({
                'vectors': pred_vectors_clipped,
                'labels': pred_labels_clipped,
                'scores': pred_scores_clipped
            })
            
            self.ground_truths_per_camera[camera_name].append({
                'vectors': gt_vectors,
                'labels': gt_labels
            })
        
        self.num_samples_processed += 1
    
    def match_predictions_to_gt_maptr_official(self,
                                               pred_vectors: np.ndarray,
                                               pred_scores: np.ndarray,
                                               gt_vectors: np.ndarray,
                                               threshold: float,
                                               pred_geometries: List = None,
                                               gt_geometries: List = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match predictions to GT using MapTR's EXACT OFFICIAL method.
        EXACT copy from MapTR's tpfp.py:custom_tpfp_gen()
        
        OPTIMIZED: Accepts pre-computed geometries to avoid redundant buffering.
        """
        num_preds = len(pred_vectors)
        num_gts = len(gt_vectors)
        
        tp = np.zeros(num_preds, dtype=np.float32)
        fp = np.zeros(num_preds, dtype=np.float32)
        
        if num_gts == 0:
            fp[:] = 1
            return tp, fp
        
        if num_preds == 0:
            return tp, fp
        
        # Convert threshold to NEGATIVE
        if threshold > 0:
            threshold = -threshold
        
        # Compute CD matrix (with optional pre-computed geometries)
        cd_matrix = self.compute_chamfer_distance_matrix_maptr_official(
            pred_vectors, gt_vectors, linewidth=2.0,
            pred_geometries=pred_geometries, gt_geometries=gt_geometries)
        
        # Find best matching GT for each prediction
        matrix_max = cd_matrix.max(axis=1)
        matrix_argmax = cd_matrix.argmax(axis=1)
        
        # Sort by confidence (descending)
        sort_inds = np.argsort(-pred_scores)
        
        # Track matched GTs
        gt_covered = np.zeros(num_gts, dtype=bool)
        
        # Greedy matching
        for i in sort_inds:
            if matrix_max[i] >= threshold:
                matched_gt = matrix_argmax[i]
                if not gt_covered[matched_gt]:
                    gt_covered[matched_gt] = True
                    tp[i] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        return tp, fp
    
    def compute_ap_area_based(self,
                              recalls: np.ndarray,
                              precisions: np.ndarray) -> float:
        """
        Compute Average Precision using area under PR curve.
        """
        mrec = np.concatenate([[0], recalls, [1]])
        mpre = np.concatenate([[0], precisions, [0]])
        
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
        
        return float(ap)
    
    def compute_ap_for_class(self,
                            pred_vectors_list: List[np.ndarray],
                            pred_scores_list: List[np.ndarray],
                            gt_vectors_list: List[np.ndarray],
                            threshold: float,
                            class_name: str = "",
                            pred_geoms_list: List = None,
                            gt_geoms_list: List = None) -> Tuple[float, float]:
        """
        Compute AP and average CD for a single class at given threshold.
        
        Args:
            pred_geoms_list: Optional pre-computed prediction geometries (avoids redundant buffering)
            gt_geoms_list: Optional pre-computed GT geometries (avoids redundant buffering)
        """
        num_gts = sum(len(gts) for gts in gt_vectors_list)
        
        if num_gts == 0:
            return 0.0, float('inf')
        
        all_tp = []
        all_fp = []
        all_scores = []
        chamfer_distances_per_sample = []
        
        # Use pre-computed geometries if provided, otherwise compute them
        if pred_geoms_list is None or gt_geoms_list is None:
            pred_geoms_list = []
            gt_geoms_list = []
            
            desc = f"Pre-computing geometries ({class_name})" if class_name else "Pre-computing geometries"
            for pred_vecs, gt_vecs in tqdm(zip(pred_vectors_list, gt_vectors_list), 
                                           total=len(pred_vectors_list),
                                           desc=desc,
                                           leave=False,
                                           disable=len(pred_vectors_list) < 100):
                if len(pred_vecs) > 0:
                    pred_geoms = self.precompute_shapely_geometries(pred_vecs, linewidth=2.0)
                else:
                    pred_geoms = []
                
                if len(gt_vecs) > 0:
                    gt_geoms = self.precompute_shapely_geometries(gt_vecs, linewidth=2.0)
                else:
                    gt_geoms = []
                
                pred_geoms_list.append(pred_geoms)
                gt_geoms_list.append(gt_geoms)
        
        # Match predictions using pre-computed geometries
        iterator = zip(pred_vectors_list, pred_scores_list, gt_vectors_list, 
                      pred_geoms_list, gt_geoms_list)
        
        for pred_vecs, pred_scores, gt_vecs, pred_geoms, gt_geoms in iterator:
            if len(pred_vecs) == 0:
                continue
            
            if len(gt_vecs) == 0:
                all_tp.append(np.zeros(len(pred_vecs), dtype=np.float32))
                all_fp.append(np.ones(len(pred_vecs), dtype=np.float32))
                all_scores.append(pred_scores)
                continue
            
            # Match predictions to GT (using pre-computed geometries)
            tp, fp = self.match_predictions_to_gt_maptr_official(
                pred_vecs, pred_scores, gt_vecs, threshold,
                pred_geometries=pred_geoms, gt_geometries=gt_geoms)
            
            all_tp.append(tp)
            all_fp.append(fp)
            all_scores.append(pred_scores)
            
            # Compute chamfer distance
            cd_sample = self.compute_chamfer_distance_torch(pred_vecs, gt_vecs)
            chamfer_distances_per_sample.append(cd_sample)
        
        if len(all_tp) == 0:
            return 0.0, float('inf')
        
        # Concatenate all predictions
        all_tp = np.concatenate(all_tp)
        all_fp = np.concatenate(all_fp)
        all_scores = np.concatenate(all_scores)
        
        # Sort by confidence
        sort_inds = np.argsort(-all_scores)
        tp = all_tp[sort_inds]
        fp = all_fp[sort_inds]
        
        # Compute cumulative TP/FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Compute precision and recall
        eps = np.finfo(np.float32).eps
        recalls = tp_cumsum / np.maximum(num_gts, eps)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        
        # Compute AP
        ap = self.compute_ap_area_based(recalls, precisions)
        
        # Average CD
        avg_cd = np.mean(chamfer_distances_per_sample) if chamfer_distances_per_sample else float('inf')
        
        return ap, avg_cd
    
    def evaluate(self) -> Dict:
        """
        Compute final metrics across all cameras and classes.
        """
        results = {}
        class_names = ['divider', 'ped_crossing', 'boundary']
        
        for camera_name in self.camera_names:
            camera_results = {}
            all_aps = []
            
            camera_preds = self.predictions_per_camera[camera_name]
            camera_gts = self.ground_truths_per_camera[camera_name]
            
            # PRE-COMPUTE GEOMETRIES ONCE PER CAMERA (not per class!)
            # This avoids redundant buffering: 6 cameras × 3 classes × 3 thresholds = 54 runs
            # Optimized to: 6 cameras = 6 runs (9x speedup!)
            # NOW PARALLELIZED: Use multiprocessing for additional 8x speedup
            print(f"Pre-computing geometries for {camera_name}...")
            
            # Prepare data for parallel buffering
            buffer_tasks = [(pred_data['vectors'], gt_data['vectors']) 
                           for pred_data, gt_data in zip(camera_preds, camera_gts)]
            
            # Use multiprocessing to buffer geometries in parallel
            num_workers = self.num_workers  # Use configured num_workers
            if num_workers > 1:
                # Clamp to cpu_count() to avoid excessive overhead if user requests too many
                num_workers = min(num_workers, cpu_count())
            
            if num_workers > 1:
                print(f"  Using {num_workers} parallel workers for geometry buffering...")
                with Pool(processes=num_workers) as pool:
                    geom_results = list(tqdm(
                        pool.starmap(buffer_geometries_worker, buffer_tasks),
                        total=len(buffer_tasks),
                        desc=f"Buffering geometries ({camera_name})",
                        leave=False
                    ))
                all_pred_geoms = [r[0] for r in geom_results]
                all_gt_geoms = [r[1] for r in geom_results]
            else:
                # Serial fallback
                all_pred_geoms = []
                all_gt_geoms = []
                for pred_data, gt_data in tqdm(zip(camera_preds, camera_gts),
                                              total=len(camera_preds),
                                              desc=f"Buffering geometries ({camera_name})",
                                              leave=False):
                    if len(pred_data['vectors']) > 0:
                        pred_geoms = self.precompute_shapely_geometries(pred_data['vectors'], linewidth=2.0)
                    else:
                        pred_geoms = []
                    
                    if len(gt_data['vectors']) > 0:
                        gt_geoms = self.precompute_shapely_geometries(gt_data['vectors'], linewidth=2.0)
                    else:
                        gt_geoms = []
                    
                    all_pred_geoms.append(pred_geoms)
                    all_gt_geoms.append(gt_geoms)
            
            # Evaluate each class
            for class_id, class_name in enumerate(class_names):
                class_results = {}
                
                # Extract predictions and GT for this class
                pred_vectors_list = []
                pred_scores_list = []
                gt_vectors_list = []
                
                # Also extract the corresponding pre-computed geometries for this class
                pred_geoms_for_class = []
                gt_geoms_for_class = []
                
                for pred_data, gt_data, pred_geoms, gt_geoms in zip(camera_preds, camera_gts, 
                                                                     all_pred_geoms, all_gt_geoms):
                    pred_mask = pred_data['labels'] == class_id
                    gt_mask = gt_data['labels'] == class_id
                    
                    pred_vectors_list.append(pred_data['vectors'][pred_mask])
                    pred_scores_list.append(pred_data['scores'][pred_mask])
                    gt_vectors_list.append(gt_data['vectors'][gt_mask])
                    
                    # Extract geometries for this class using the same mask
                    if len(pred_geoms) > 0:
                        pred_geoms_for_class.append([pred_geoms[i] for i in range(len(pred_geoms)) if pred_mask[i]])
                    else:
                        pred_geoms_for_class.append([])
                    
                    if len(gt_geoms) > 0:
                        gt_geoms_for_class.append([gt_geoms[i] for i in range(len(gt_geoms)) if gt_mask[i]])
                    else:
                        gt_geoms_for_class.append([])
                
                # Compute AP at each threshold
                avg_cd = None
                for threshold in self.thresholds_chamfer:
                    ap, cd = self.compute_ap_for_class(
                        pred_vectors_list, pred_scores_list, gt_vectors_list, 
                        threshold, class_name=class_name,
                        pred_geoms_list=pred_geoms_for_class,
                        gt_geoms_list=gt_geoms_for_class)
                    
                    class_results[f'AP@{threshold}m'] = ap
                    all_aps.append(ap)
                    
                    if avg_cd is None:
                        avg_cd = cd
                
                class_results['avg_chamfer_distance'] = avg_cd if avg_cd is not None else float('inf')
                
                camera_results[class_name] = class_results
            
            # Compute mAP
            camera_results['mAP'] = np.mean(all_aps) if all_aps else 0.0
            
            results[camera_name] = camera_results
        
        # Compute average across cameras if multiple
        if len(self.camera_names) > 1:
            avg_results = {}
            
            for class_name in class_names:
                class_avg = {}
                for threshold in self.thresholds_chamfer:
                    threshold_key = f'AP@{threshold}m'
                    aps = [results[cam][class_name][threshold_key] 
                           for cam in self.camera_names 
                           if class_name in results[cam]]
                    class_avg[threshold_key] = np.mean(aps) if aps else 0.0
                
                cds = [results[cam][class_name]['avg_chamfer_distance'] 
                      for cam in self.camera_names 
                      if class_name in results[cam] and results[cam][class_name]['avg_chamfer_distance'] != float('inf')]
                class_avg['avg_chamfer_distance'] = np.mean(cds) if cds else float('inf')
                
                avg_results[class_name] = class_avg
            
            all_camera_maps = [results[cam]['mAP'] for cam in self.camera_names]
            avg_results['mAP'] = np.mean(all_camera_maps) if all_camera_maps else 0.0
            
            results['AVERAGE'] = avg_results
        
        return results


# ==================== MAIN UNIFIED WORKFLOW ====================
def main():
    parser = argparse.ArgumentParser(
        description='Unified GeMap Evaluation with Noise: Inference + Camera-Specific FOV Clipping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline (no noise)
  python %(prog)s --cameras CAM_FRONT

  # Translation noise (following MapTR paper Table 11)
  python %(prog)s --cameras all --noise-trans-std 0.05
  python %(prog)s --cameras all --noise-trans-std 0.1
  python %(prog)s --cameras all --noise-trans-std 0.5
  python %(prog)s --cameras all --noise-trans-std 1.0
  
  # Rotation noise (following MapTR paper Table 12)
  python %(prog)s --cameras all --noise-rot-std 0.005
  python %(prog)s --cameras all --noise-rot-std 0.01
  python %(prog)s --cameras all --noise-rot-std 0.02
  python %(prog)s --cameras all --noise-rot-std 0.05
  
  # Combined noise
  python %(prog)s --cameras all --noise-trans-std 0.1 --noise-rot-std 0.01
  
  # Custom camera subset with noise
  python %(prog)s --cameras CAM_FRONT CAM_BACK --noise-trans-std 0.1
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
                       help='Camera names to use (CAM_FRONT, CAM_BACK, etc.) or "all" for all 6 cameras')
    
    # Noise parameters (following MapTR robustness experiments)
    parser.add_argument('--noise-trans-std', type=float, default=0.0,
                       help='Standard deviation of Gaussian noise for translation (meters). e.g. 0.05, 0.1, 0.5, 1.0')
    parser.add_argument('--noise-rot-std', type=float, default=0.0,
                       help='Standard deviation of Gaussian noise for rotation (radians). e.g. 0.005, 0.01, 0.02, 0.05')
    parser.add_argument('--noise-seed', type=int, default=42,
                       help='Random seed for noise generation (default: 42 for reproducibility)')
    
    # Evaluation arguments
    parser.add_argument('--nuscenes-path', type=str,
                       default=None,
                       help='Path to NuScenes dataset (default: use path from config file)')
    parser.add_argument('--samples-pkl', type=str,
                       default=None,
                       help='Path to samples pickle file (default: use path from config file)')
    parser.add_argument('--pc-range', type=float, nargs=6,
                       default=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                       help='Point cloud range')
    parser.add_argument('--num-sample-pts', type=int, default=100,
                       help='Number of points to resample vectors to (default: 100)')
    
    # Output arguments
    parser.add_argument('--predictions-pkl', type=str, default=None,
                       help='Output/input pickle file for predictions (auto-generated if not specified)')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Output JSON file for results (auto-generated if not specified)')
    
    # Control flow
    parser.add_argument('--skip-inference', action='store_true',
                       help='Skip inference step (use existing predictions)')
    parser.add_argument('--apply-clipping', action='store_true',
                       help='Apply camera FOV clipping (default: True)')
    parser.add_argument('--no-clipping', dest='apply_clipping', action='store_false',
                       help='Disable FOV clipping (full BEV evaluation)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help=f'Number of parallel workers for evaluation (default: 8, set to 1 for serial, max: {cpu_count()})')
    parser.set_defaults(apply_clipping=True)
    
    args = parser.parse_args()
    
    # Parse camera configuration
    camera_indices = parse_camera_config(args.cameras)
    camera_names = [name for name, idx in CAMERA_MAP.items() if idx in camera_indices]
    
    # Generate default filenames if not specified
    camera_suffix = '_'.join([name.replace('CAM_', '').lower() for name in camera_names])
    noise_suffix = ""
    if args.noise_trans_std > 0:
        noise_suffix += f"_trans{args.noise_trans_std:.3f}"
    if args.noise_rot_std > 0:
        noise_suffix += f"_rot{args.noise_rot_std:.3f}"
    if not noise_suffix:
        noise_suffix = "_clean"
    if args.predictions_pkl is None:
        args.predictions_pkl = f'gemap_predictions_{camera_suffix}{noise_suffix}.pkl'
    if args.output_json is None:
        args.output_json = f'evaluation_results_{camera_suffix}{noise_suffix}.json'
    
    print("\n" + "="*80)
    print("UNIFIED GeMap EVALUATION PIPELINE (with NOISE)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  NuScenes path: {args.nuscenes_path or 'from config file'}")
    print(f"  Samples pickle: {args.samples_pkl or 'from config file'}")
    print(f"  Camera configuration: {camera_names} ({len(camera_names)}/6 cameras)")
    print(f"  Noise Trans std: {args.noise_trans_std} meters")
    print(f"  Noise Rot std: {args.noise_rot_std} radians")
    print(f"  Noise seed: {args.noise_seed}")
    print(f"  FOV clipping: {'ENABLED' if args.apply_clipping else 'DISABLED (full BEV)'}")
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
            nuscenes_path=args.nuscenes_path,
            noise_trans_std=args.noise_trans_std,
            noise_rot_std=args.noise_rot_std,
            noise_seed=args.noise_seed
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
    
    # Get actual samples path (from args or infer from config)
    if args.samples_pkl:
        samples_path = args.samples_pkl
    else:
        # Load config to get the samples path
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
    
    # STEP 3: Evaluate
    print("\n" + "="*80)
    print("STEP 3: Evaluating with Camera-Specific FOV Clipping")
    print("="*80)
    
    # Get actual NuScenes path for evaluation
    if args.nuscenes_path:
        nuscenes_eval_path = args.nuscenes_path
    else:
        # Load config to get the data root
        cfg = Config.fromfile(args.config)
        nuscenes_eval_path = cfg.data.test.data_root
    
    # Create evaluator
    evaluator = CameraSpecificEvaluator(
        nuscenes_data_path=nuscenes_eval_path,
        pc_range=args.pc_range,
        num_sample_pts=args.num_sample_pts,
        thresholds_chamfer=[0.5, 1.0, 1.5],
        camera_names=camera_names,
        num_workers=max(1, args.num_workers)  # Pass configured worker count
    )
    
    print(f"\nEvaluator configuration:")
    print(f"  PC range: {args.pc_range}")
    print(f"  Patch size: {evaluator.patch_size}")
    print(f"  Sample points per vector: {args.num_sample_pts} (MapTR standard)")
    print(f"  Chamfer thresholds: {evaluator.thresholds_chamfer} meters")
    print(f"  Evaluation method: Official MapTR (STRtree, linewidth=2m)")
    
    # Accumulate predictions and GT
    mode_str = "camera-specific FOV clipping" if args.apply_clipping else "full BEV (no clipping)"
    
    # Determine number of workers (default is 8)
    num_workers = max(1, args.num_workers)  # At least 1 worker
    
    print(f"\nProcessing {len(samples)} samples with {mode_str}...")
    print(f"Using {num_workers} parallel workers (CPU count: {cpu_count()})")
    
    # Prepare data for parallel processing
    samples_with_preds = []
    for sample_info in samples:
        sample_token = sample_info['token']
        if sample_token in predictions_by_token:
            samples_with_preds.append((sample_info, predictions_by_token[sample_token]))
    
    print(f"Matched {len(samples_with_preds)} samples with predictions")
    
    if num_workers == 1:
        # Serial processing (for debugging or when multiprocessing has issues)
        print("Running in SERIAL mode...")
        results_list = []
        for sample_info, pred_data in tqdm(samples_with_preds, desc="Processing samples"):
            result = process_sample_worker(
                sample_info, pred_data, nuscenes_eval_path, args.pc_range,
                args.num_sample_pts, camera_names, args.apply_clipping
            )
            results_list.append(result)
    else:
        # Parallel processing
        print("Running in PARALLEL mode...")
        worker_fn = partial(
            process_sample_worker,
            nuscenes_data_path=nuscenes_eval_path,
            pc_range=args.pc_range,
            num_sample_pts=args.num_sample_pts,
            camera_names=camera_names,
            apply_clipping=args.apply_clipping
        )
        
        with Pool(processes=num_workers) as pool:
            # Use imap for progress bar
            results_list = list(tqdm(
                pool.starmap(worker_fn, samples_with_preds),
                total=len(samples_with_preds),
                desc="Processing samples"
            ))
    
    # Merge results into evaluator
    print("\nMerging results from parallel workers...")
    for result in results_list:
        for camera_name in camera_names:
            camera_result = result[camera_name]
            
            # Only append if there was actual data for this camera in the sample
            if camera_result['pred'] is not None:
                evaluator.predictions_per_camera[camera_name].append(camera_result['pred'])
            if camera_result['gt'] is not None:
                evaluator.ground_truths_per_camera[camera_name].append(camera_result['gt'])
        
        evaluator.num_samples_processed += 1  
    # Compute metrics
    print("\nComputing metrics...")
    results = evaluator.evaluate()
    
    # STEP 4: Print and save results
    print("\n" + "="*80)
    print("STEP 4: EVALUATION RESULTS")
    print("="*80)
    
    class_names = ['divider', 'ped_crossing', 'boundary']
    
    # Print average first if multiple cameras
    if 'AVERAGE' in results:
        camera_results = results['AVERAGE']
        print(f"\nAVERAGE (across {len(camera_names)} cameras):")
        print(f"  mAP (all classes & thresholds): {camera_results['mAP']:.4f}")
        print()
        for class_name in class_names:
            if class_name not in camera_results:
                continue
            class_results = camera_results[class_name]
            print(f"  {class_name}:")
            for threshold in evaluator.thresholds_chamfer:
                ap = class_results[f'AP@{threshold}m']
                print(f"    AP@{threshold}m: {ap:.4f}")
            cd = class_results['avg_chamfer_distance']
            cd_str = f"{cd:.4f}m" if cd != float('inf') else "N/A"
            print(f"    Avg CD: {cd_str}")
        print("\n" + "-"*80)
    
    # Print per-camera results
    for camera_name, camera_results in results.items():
        if camera_name == 'AVERAGE':
            continue
        
        print(f"\n{camera_name}:")
        
        if 'mAP' in camera_results:
            print(f"  mAP (all classes & thresholds): {camera_results['mAP']:.4f}")
        
        print()
        for class_name in class_names:
            if class_name not in camera_results:
                continue
            class_results = camera_results[class_name]
            print(f"  {class_name}:")
            for threshold in evaluator.thresholds_chamfer:
                ap = class_results[f'AP@{threshold}m']
                print(f"    AP@{threshold}m: {ap:.4f}")
            cd = class_results['avg_chamfer_distance']
            cd_str = f"{cd:.4f}m" if cd != float('inf') else "N/A"
            print(f"    Avg CD: {cd_str}")
    
    # Save results
    print(f"\n\nSaving results to {args.output_json}...")
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
