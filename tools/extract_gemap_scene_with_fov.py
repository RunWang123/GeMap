"""
Extract GeMap Scene Predictions with Camera FOV Clipping

This script runs GeMap inference with camera-specific FOV clipping to match
the VGGT evaluation protocol:
1. Run inference on specific camera view(s) only (e.g., CAM_FRONT)
2. Apply FOV clipping to filter predictions visible in each camera
3. Transform to global coordinates
4. Merge predictions from all timestamps to construct scene map

This enables fair comparison with VGGT which uses camera-specific inputs and FOV clipping.

Usage:
    # Single camera (CAM_FRONT only)
    python extract_gemap_scene_with_fov.py \\
        --config /home/runw/Project/GeMap/projects/configs/gemap/gemap_simple_r50_110ep.py \\
        --checkpoint /home/runw/Project/GeMap/ckpts/gemap_simple_r50_110ep.pth \\
        --nuscenes_path /home/runw/Project/data/mini/nuscenes \\
        --output_dir output/gemap_scene_fov \\
        --version v1.0-mini \\
        --scene_idx 0 \\
        --camera CAM_FRONT
    
    # Multiple cameras
    python extract_gemap_scene_with_fov.py \\
        --config ... \\
        --camera CAM_FRONT,CAM_BACK
"""

import argparse
import mmcv
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# Ensure GeMap repo root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings('ignore')

# GeMap imports
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

# NuScenes imports
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
except ImportError:
    print("Error: nuscenes-devkit not available. Please install it.")
    sys.exit(1)

# Import FOV clipping utilities (from local tools which we fixed)
# maptr_tools_path = Path(__file__).resolve().parents[2] / 'MapTR' / 'tools'
# if str(maptr_tools_path) not in sys.path:
#     sys.path.insert(0, str(maptr_tools_path))

from camera_fov_utils import CameraFOVClipper


def load_gemap_model(config_path: str, checkpoint_path: str, device: str = 'cuda'):
    """Load GeMap model"""
    print(f"Loading GeMap config from: {config_path}")
    cfg = Config.fromfile(config_path)
    
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
            importlib.import_module(_module_path)
    
    # Build model
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Handle FP16
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    # Move to device and set eval mode
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    
    return model, cfg


def run_inference_with_fov_clipping(
    model,
    cfg,
    nusc: NuScenes,
    scene_idx: int,
    camera_names: List[str],
    confidence_threshold: float = 0.3,
    device: str = 'cuda',
) -> Dict:
    """
    Run GeMap inference on scene with FOV clipping per camera.
    
    Returns predictions in GLOBAL coordinates after FOV filtering.
    """
    scene = nusc.scene[scene_idx]
    scene_token = scene['token']
    scene_name = f"scene-{scene_token[:4]}"  # Use token like gamma script
    
    print(f"\nProcessing scene {scene_idx}: {scene_name} (token: {scene_token})")
    print(f"  Cameras: {', '.join(camera_names)}")
    
    # Get all samples in scene
    sample_token = scene['first_sample_token']
    samples = []
    while sample_token:
        sample = nusc.get('sample', sample_token)
        samples.append(sample)
        sample_token = sample['next']
    
    print(f"  Timestamps: {len(samples)}")
    
    # Get PC range from config
    pc_range = cfg.point_cloud_range
    
    # Initialize FOV clipper
    fov_clipper = CameraFOVClipper()
    
    # Storage for FOV-clipped predictions
    all_predictions = []
    all_labels = []
    all_scores = []
    all_timestamps = []
    all_cameras = []
    
    # Build dataset (prefer val, fallback to train)
    cfg.data.test.test_mode = True
    cfg.data.test.data_root = nusc.dataroot
    
    # Fix annotation file path - check if file exists, otherwise try alternative locations
    ann_file = cfg.data.test.ann_file
    if not os.path.isabs(ann_file) or not os.path.exists(ann_file):
        # Try common locations for mini dataset
        possible_paths = [
            os.path.join(nusc.dataroot, 'gemap', 'nuscenes_map_infos_temporal_val.pkl'),
            os.path.join(nusc.dataroot, 'nuscenes_map_infos_temporal_val.pkl'),
            os.path.join(nusc.dataroot, 'nuscenes_infos_temporal_val.pkl'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"  Using annotation file: {path}")
                cfg.data.test.ann_file = path
                break
        else:
            print(f"  Warning: Could not find annotation file. Tried:")
            for path in possible_paths:
                print(f"    - {path}")
            print(f"  Will attempt to use original path: {ann_file}")
    
    dataset_val = build_dataset(cfg.data.test)
    
    import copy as _copy
    train_like_test_cfg = _copy.deepcopy(cfg.data.test)
    train_like_test_cfg['ann_file'] = cfg.data.train['ann_file']
    
    # Fix train annotation file path too
    ann_file_train = train_like_test_cfg['ann_file']
    if not os.path.isabs(ann_file_train) or not os.path.exists(ann_file_train):
        possible_paths_train = [
            os.path.join(nusc.dataroot, 'gemap', 'nuscenes_map_infos_temporal_train.pkl'),
            os.path.join(nusc.dataroot, 'nuscenes_map_infos_temporal_train.pkl'),
            os.path.join(nusc.dataroot, 'nuscenes_infos_temporal_train.pkl'),
        ]
        for path in possible_paths_train:
            if os.path.exists(path):
                train_like_test_cfg['ann_file'] = path
                break
    
    dataset_train = build_dataset(train_like_test_cfg)
    
    # Collect sample indices for this scene
    def _collect_indices(ds):
        indices = []
        for idx, data_info in enumerate(getattr(ds, 'data_infos', [])):
            if data_info.get('scene_token', None) == scene_token:
                indices.append(idx)
        return indices
    
    scene_sample_indices = _collect_indices(dataset_val)
    selected_split = 'val'
    selected_dataset = dataset_val
    if len(scene_sample_indices) == 0:
        scene_sample_indices = _collect_indices(dataset_train)
        selected_split = 'train'
        selected_dataset = dataset_train
    
    print(f"  Found {len(scene_sample_indices)} samples in {selected_split} dataset")
    
    if len(scene_sample_indices) == 0:
        print("  Warning: No samples found for this scene")
        return {
            'predictions': [],
            'labels': [],
            'scores': [],
            'timestamps': [],
            'cameras': [],
            'scene_token': scene_token,
            'scene_name': scene_name,
        }
    
    # Build dataloader
    scene_dataset = torch.utils.data.Subset(selected_dataset, scene_sample_indices)
    data_loader = build_dataloader(
        scene_dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    
    # Run inference on each timestamp
    with torch.no_grad():
        for ts_idx, data in enumerate(tqdm(data_loader, desc="Inference + FOV clipping")):
            sample = samples[ts_idx]
            
            # Zero out all camera views except the specified ones
            # NuScenes camera order: CAM_FRONT (0), CAM_FRONT_RIGHT (1), CAM_FRONT_LEFT (2),
            #                        CAM_BACK (3), CAM_BACK_LEFT (4), CAM_BACK_RIGHT (5)
            camera_map = {
                'CAM_FRONT': 0,
                'CAM_FRONT_RIGHT': 1,
                'CAM_FRONT_LEFT': 2,
                'CAM_BACK': 3,
                'CAM_BACK_LEFT': 4,
                'CAM_BACK_RIGHT': 5
            }
            
            # Get target camera indices
            target_camera_indices = [camera_map[cam] for cam in camera_names if cam in camera_map]
            
            # Zero out unwanted camera inputs
            if 'img' in data and data['img'][0] is not None:
                imgs = data['img'][0].data[0]  # Shape: [B, N_views, C, H, W] or [N_views, C, H, W]
                
                if ts_idx == 0:
                    print(f"  Image tensor shape: {imgs.shape}")
                    print(f"  Using camera indices {target_camera_indices} ({', '.join(camera_names)})")
                
                # Zero out all views except target cameras
                if len(imgs.shape) == 5:  # [B, N_views, C, H, W]
                    mask = torch.zeros_like(imgs)
                    for cam_idx in target_camera_indices:
                        mask[:, cam_idx, :, :, :] = 1
                    data['img'][0].data[0] = imgs * mask
                elif len(imgs.shape) == 4:  # [N_views, C, H, W]
                    mask = torch.zeros_like(imgs)
                    for cam_idx in target_camera_indices:
                        mask[cam_idx, :, :, :] = 1
                    data['img'][0].data[0] = imgs * mask
            
            # Forward pass (with zeroed camera inputs)
            result = model(return_loss=False, rescale=True, **data)
            
            # Extract predictions (in lidar-centric coordinates)
            # GeMap uses same format as MapTR
            result_dic = result[0]['pts_bbox']
            pts_3d = result_dic['pts_3d']
            scores_3d = result_dic['scores_3d']
            labels_3d = result_dic['labels_3d']
            
            # Filter by confidence
            keep = scores_3d > confidence_threshold
            if keep.sum() == 0:
                continue
            
            pred_pts_lidar = pts_3d[keep].cpu().numpy()
            pred_labels = labels_3d[keep].cpu().numpy()
            pred_scores = scores_3d[keep].cpu().numpy()
            
            # Process each camera
            for camera_name in camera_names:
                # Get camera calibration for FOV clipping
                sd_token = sample['data'][camera_name]
                sd = nusc.get('sample_data', sd_token)
                ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
                cs_record = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
                
                # Build BEV-aligned camera extrinsics (EXACT logic from camera_fov_utils.py)
                # This is CRITICAL for correct FOV clipping!
                lidar_sd_token = sample['data']['LIDAR_TOP']
                lidar_sd = nusc.get('sample_data', lidar_sd_token)
                lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
                lidar_ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
                
                # Camera2ego transform
                cam2ego = np.eye(4)
                cam2ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
                cam2ego[:3, 3] = cs_record['translation']
                
                # Ego2global transform
                ego2global = np.eye(4)
                ego2global[:3, :3] = Quaternion(lidar_ego_pose['rotation']).rotation_matrix
                ego2global[:3, 3] = lidar_ego_pose['translation']
                
                # Lidar2ego transform
                lidar2ego = np.eye(4)
                lidar2ego[:3, :3] = Quaternion(lidar_cs['rotation']).rotation_matrix
                lidar2ego[:3, 3] = lidar_cs['translation']
                
                # Compute composed transforms
                lidar2global = ego2global @ lidar2ego
                cam2global = ego2global @ cam2ego
                
                # Get lidar2global rotation angle for BEV alignment
                lidar2global_rotation = Quaternion(matrix=lidar2global)
                patch_angle_deg = quaternion_yaw(lidar2global_rotation) / np.pi * 180
                patch_angle_rad = np.radians(patch_angle_deg)
                
                # Create BEV alignment rotation matrix
                cos_a = np.cos(-patch_angle_rad)
                sin_a = np.sin(-patch_angle_rad)
                rotation_matrix_bev = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
                
                # Apply BEV alignment to camera extrinsics
                cam_extrinsics_bev = np.eye(4)
                cam_extrinsics_bev[:3, :3] = rotation_matrix_bev @ cam2global[:3, :3]
                cam_extrinsics_bev[:3, 3] = rotation_matrix_bev @ (cam2global[:3, 3] - lidar2global[:3, 3])
                
                # Camera intrinsics
                intrinsics = np.array(cs_record['camera_intrinsic'])
                
                # Prepare predictions in FOV clipper format (List[List[np.ndarray]])
                # Each prediction is a separate "instance" group
                vectors_for_fov = [[pred] for pred in pred_pts_lidar]
                labels_for_fov = pred_labels.tolist()
                
                # Apply FOV clipping using BEV-aligned extrinsics (CRITICAL!)
                fov_clipped_vecs, fov_clipped_labels, _ = fov_clipper.crop_vectors_to_fov(
                    vectors=vectors_for_fov,
                    labels=labels_for_fov,
                    extrinsics=cam_extrinsics_bev,  # Use BEV-aligned extrinsics!
                    intrinsics=intrinsics
                )
                
                if len(fov_clipped_vecs) == 0:
                    continue
                
                # Flatten back to single vectors (since each group has only 1 vector)
                fov_clipped_preds = [vecs[0] for vecs in fov_clipped_vecs if len(vecs) > 0 and len(vecs[0]) >= 2]
                fov_clipped_labels_flat = [label for vecs, label in zip(fov_clipped_vecs, fov_clipped_labels) if len(vecs) > 0 and len(vecs[0]) >= 2]
                
                # Get corresponding scores (match by index in original predictions)
                fov_clipped_scores = []
                for i, (vecs, label) in enumerate(zip(fov_clipped_vecs, fov_clipped_labels)):
                    if len(vecs) > 0 and len(vecs[0]) >= 2:
                        fov_clipped_scores.append(pred_scores[i])
                
                if len(fov_clipped_preds) == 0:
                    continue
                
                # Transform FOV-clipped predictions from lidar to global coordinates
                lidar_ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
                ego2global = np.eye(4)
                ego2global[:3, :3] = Quaternion(lidar_ego_pose['rotation']).rotation_matrix
                ego2global[:3, 3] = lidar_ego_pose['translation']
                
                lidar2global = ego2global @ lidar2ego
                
                # Transform each prediction to global
                for pred, label, score in zip(fov_clipped_preds, fov_clipped_labels_flat, fov_clipped_scores):
                    pred_global = np.zeros_like(pred)
                    for i, pt in enumerate(pred):
                        pt_3d = np.array([pt[0], pt[1], 0.0, 1.0])
                        pt_global = lidar2global @ pt_3d
                        pred_global[i] = pt_global[:2]
                    
                    all_predictions.append(pred_global)
                    all_labels.append(int(label))
                    all_scores.append(float(score))
                    all_timestamps.append(ts_idx)
                    all_cameras.append(camera_name)
    
    print(f"  Collected {len(all_predictions)} FOV-clipped predictions")
    
    # Count per class
    label_counts = {0: 0, 1: 0, 2: 0}
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"  Class distribution:")
    print(f"    Divider: {label_counts.get(0, 0)}")
    print(f"    Ped Crossing: {label_counts.get(1, 0)}")
    print(f"    Boundary: {label_counts.get(2, 0)}")
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'scores': all_scores,
        'timestamps': all_timestamps,
        'cameras': all_cameras,
        'scene_token': scene_token,
        'scene_name': scene_name,
        'camera_names': camera_names,
    }


def visualize_scene_predictions(
    pred_data: Dict,
    output_path: Path,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    figsize: Tuple[int, int] = (20, 20),
):
    """Visualize FOV-clipped scene predictions"""
    predictions = pred_data['predictions']
    labels = pred_data['labels']
    
    if len(predictions) == 0:
        print(f"  Warning: No predictions to visualize")
        return
    
    class_colors = {0: 'orange', 1: 'blue', 2: 'green'}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute bounds
    if xlim is None or ylim is None:
        all_coords = np.concatenate([p.reshape(-1, 2) for p in predictions])
        x_min, y_min = all_coords.min(axis=0)
        x_max, y_max = all_coords.max(axis=0)
        
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        
        if xlim is None:
            xlim = (x_min - x_margin, x_max + x_margin)
        if ylim is None:
            ylim = (y_min - y_margin, y_max + y_margin)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Plot predictions
    for pred, label in zip(predictions, labels):
        color = class_colors.get(label, 'gray')
        ax.plot(pred[:, 0], pred[:, 1], color=color, linewidth=1.5, alpha=0.7)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract GeMap scene predictions with FOV clipping')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--nuscenes_path', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-mini',
                       choices=['v1.0-trainval', 'v1.0-test', 'v1.0-mini'])
    parser.add_argument('--split', type=str, default='val', nargs='?', const='all',
                       choices=['train', 'val', 'all', 'none', 'None'],
                       help='Filter scenes by split (train/val). Default is val. If "all" (or flag without arg), process all scenes.')
    parser.add_argument('--scene_idx', type=int, help='Specific scene index')
    parser.add_argument('--num_scenes', type=int, help='Number of scenes to process')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--camera', type=str, default='CAM_FRONT',
                       help='Camera(s) to use, comma-separated (e.g., CAM_FRONT,CAM_BACK)')
    parser.add_argument('--confidence_threshold', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Parse camera names
    camera_names = [cam.strip() for cam in args.camera.split(',')]
    print(f"Using cameras: {', '.join(camera_names)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, cfg = load_gemap_model(args.config, args.checkpoint, args.device)
    
    # Load NuScenes
    print(f"\nLoading nuScenes {args.version} from {args.nuscenes_path}...")
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_path, verbose=False)
    print(f"Loaded {len(nusc.scene)} scenes")
    
    # Load scene ordering from pickle file to match Sigma dataset ordering
    import pickle
    if args.split is not None and args.split.lower() not in ['all', 'none']:
        pkl_file = os.path.join(args.nuscenes_path, f'nuscenes_infos_temporal_{args.split}.pkl')
    else:
        # Default to val split for consistency with Sigma
        pkl_file = os.path.join(args.nuscenes_path, 'nuscenes_infos_temporal_val.pkl')
    
    if os.path.exists(pkl_file):
        print(f"Loading scene ordering from: {pkl_file}")
        with open(pkl_file, 'rb') as f:
            pkl_data = pickle.load(f)
        
        # Extract UNIQUE scene tokens in ORDER they appear in pickle file
        # This matches how Sigma's NuScenesMapTRDataset builds dataset.samples
        seen_scene_tokens = []
        scene_token_to_nusc_idx = {}
        
        for sample_info in pkl_data['infos']:
            scene_token = sample_info['scene_token']
            if scene_token not in seen_scene_tokens:
                seen_scene_tokens.append(scene_token)
                # Find this scene's index in nusc.scene
                for i, scene in enumerate(nusc.scene):
                    if scene['token'] == scene_token:
                        scene_token_to_nusc_idx[scene_token] = i
                        break
        
        # Create filtered_indices using pickle file scene order
        filtered_indices = [scene_token_to_nusc_idx[token] for token in seen_scene_tokens 
                           if token in scene_token_to_nusc_idx]
        
        print(f"Using {len(filtered_indices)} scenes from pickle file (matches Sigma dataset ordering)")
        print(f"First 5 scene indices: {filtered_indices[:5]}")
    else:
        print(f"Warning: Pickle file not found: {pkl_file}")
        print(f"Falling back to nuScenes scene order (may not match Sigma!)")
        filtered_indices = list(range(len(nusc.scene)))
    
    # Determine scenes to process
    if args.scene_idx is not None:
        scene_indices = [args.scene_idx]
    elif args.num_scenes is not None:
        scene_indices = filtered_indices[:args.num_scenes]
    else:
        scene_indices = filtered_indices
        print(f"Processing all {len(scene_indices)} scenes")
    
    # Process each scene
    for dataset_idx, scene_idx in enumerate(scene_indices):
        print(f"\n{'='*80}")
        print(f"Processing dataset index {dataset_idx} (nusc.scene[{scene_idx}])")
        print(f"{'='*80}")
        
        # Run inference with FOV clipping
        pred_data = run_inference_with_fov_clipping(
            model=model,
            cfg=cfg,
            nusc=nusc,
            scene_idx=scene_idx,
            camera_names=camera_names,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
        )
        
        if len(pred_data['predictions']) == 0:
            print(f"  Skipping scene {scene_idx} - no predictions after FOV clipping")
            continue
        
        # Create scene output directory (use dataset_idx to match Sigma naming)
        # This ensures scene_0000, scene_0001, etc. match the pickle file order
        scene_name = pred_data['scene_name']
        camera_suffix = '_'.join(camera_names)
        scene_output_dir = output_dir / f"scene_{dataset_idx:04d}_{scene_name}_{camera_suffix}"
        scene_output_dir.mkdir(exist_ok=True)
        
        # Save predictions
        save_data = {
            'predictions': np.array([p for p in pred_data['predictions']], dtype=object),
            'labels': np.array(pred_data['labels']),
            'scores': np.array(pred_data['scores']),
            'timestamps': np.array(pred_data['timestamps']),
            'cameras': pred_data['cameras'],
            'scene_token': pred_data['scene_token'],
            'scene_name': scene_name,
            'camera_names': camera_names,
        }
        np.save(scene_output_dir / 'gemap_fov_predictions.npy', save_data, allow_pickle=True)
        print(f"  Saved to: {scene_output_dir / 'gemap_fov_predictions.npy'}")
        
        # Visualize
        visualize_scene_predictions(
            pred_data=pred_data,
            output_path=scene_output_dir / 'gemap_fov_pred_map.png',
        )
    
    print(f"\n{'='*80}")
    print(f"Done! Processed {len(scene_indices)} scenes")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
