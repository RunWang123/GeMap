#!/usr/bin/env python3
"""
Run GeMap inference and save predictions for evaluation.
Based on MapTR's save_maptr_predictions.py but adapted for GeMap.
"""

import argparse
import mmcv
import os
import torch
import warnings
import numpy as np
import pickle
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import get_root_logger
import os.path as osp
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Run GeMap inference and save predictions')
    parser.add_argument('--config', type=str, 
                       default='/home/runw/Project/GeMap/projects/configs/gemap/gemap_simple_r50_110ep.py',
                       help='test config file path')
    parser.add_argument('--checkpoint', type=str,
                       default='/home/runw/Project/GeMap/ckpts/gemap_simple_r50_110ep.pth',
                       help='checkpoint file')
    parser.add_argument('--output-pkl', type=str, default='gemap_predictions.pkl',
                       help='Output pickle file for predictions')
    parser.add_argument('--score-thresh', type=float, default=0.0,
                       help='Score threshold for predictions (default: 0.0 = keep ALL for proper evaluation, range: 0.0-1.0)')
    parser.add_argument('--front-camera-only', action='store_true',
                       help='Zero out all camera views except CAM_FRONT to simulate corrupted cameras')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

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
    
    logger.info(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    logger.info('Model loaded and ready')

    # Storage for predictions
    predictions = {}
    
    # Run inference
    logger.info('Running inference...')
    if args.front_camera_only:
        logger.info('NOTE: Zeroing out all camera views EXCEPT CAM_FRONT to simulate corrupted cameras')
    prog_bar = mmcv.ProgressBar(len(dataset))
    
    for i, data in enumerate(data_loader):
        try:
            img_metas = data['img_metas'][0].data[0]
            
            # Get sample token - use the actual NuScenes sample_token if available
            # Debug: print available keys on first sample
            if i == 0:
                logger.info(f"DEBUG: img_metas[0] keys: {list(img_metas[0].keys())}")
            
            if 'sample_idx' in img_metas[0]:
                sample_token = img_metas[0]['sample_idx']  # This might be the token
            else:
                # Fallback to pts_filename
                pts_filename = img_metas[0]['pts_filename']
                sample_token = osp.basename(pts_filename).replace('__LIDAR_TOP__', '_').split('.')[0]
            
            # Zero out all camera views except CAM_FRONT (index 0) if requested
            # NuScenes camera order: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, 
            #                        CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
            if args.front_camera_only and 'img' in data and data['img'][0] is not None:
                imgs = data['img'][0].data[0]  # Shape: [B, N_views, C, H, W] or [N_views, C, H, W]
                if i == 0:
                    logger.info(f"DEBUG: Image tensor shape: {imgs.shape}")
                
                # Zero out views 1-5 (keep only view 0 = CAM_FRONT)
                if len(imgs.shape) == 5:  # [B, N_views, C, H, W]
                    imgs[:, 1:, :, :, :] = 0
                elif len(imgs.shape) == 4:  # [N_views, C, H, W]
                    imgs[1:, :, :, :] = 0
                else:
                    logger.warning(f"Unexpected image tensor shape: {imgs.shape}")
            
            # Run inference
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            
            # Extract predictions from GeMap result format (same as MapTR)
            result_dic = result[0]['pts_bbox']
            pred_boxes = result_dic['boxes_3d']       # bbox: xmin, ymin, xmax, ymax
            pred_scores = result_dic['scores_3d']     # confidence scores
            pred_labels = result_dic['labels_3d']     # class labels
            pred_vectors = result_dic['pts_3d']       # polyline points
            
            # Filter by score threshold
            keep = pred_scores > args.score_thresh
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
    logger.info(f'Saving {len(predictions)} predictions to {args.output_pkl}...')
    with open(args.output_pkl, 'wb') as f:
        pickle.dump(predictions, f)
    
    logger.info('Done! Predictions saved.')
    logger.info(f'\nTo evaluate, run:')
    logger.info(f'python tools/evaluate_with_fov_clipping_standalone.py \\')
    logger.info(f'  --nuscenes-path /path/to/nuscenes \\')
    logger.info(f'  --samples-pkl /path/to/nuscenes_infos_temporal_val.pkl \\')
    logger.info(f'  --predictions-pkl {args.output_pkl} \\')
    logger.info(f'  --output-json evaluation_results.json')


if __name__ == '__main__':
    main()

