#!/usr/bin/env python3
"""
Visualize GeMap predictions with GT for debugging.
Uses EXACT same logic as training_6pv_enhance for camera FOV clipping and rotation.
Supports both BEV and camera-specific visualizations.

All FOV clipping and rotation logic is imported from camera_fov_utils.py
to ensure 100% identical code with evaluation script.
"""

import sys
from pathlib import Path
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from tqdm import tqdm

# Import shared camera FOV utilities (ensures identical logic with evaluation)
from camera_fov_utils import (
    VectorizedLocalMap,
    CameraFOVClipper,
    extract_gt_vectors,
    extract_gt_with_fov_clipping,
    process_predictions_with_fov_clipping
)


# ==================== VISUALIZATION FUNCTIONS ====================
def visualize_sample(
    sample_info: Dict,
    pred_vectors: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    gt_data: Dict,
    save_path: str,
    pc_range: list,
    confidence_threshold: float = 0.0
):
    """
    Visualize GT and predictions side by side.
    EXACT same style as inference.py / visualize_clean_frame_prediction
    """
    # Colors matching inference style: orange, blue, green
    colors_plt = ['orange', 'b', 'g']
    class_names = ['divider', 'ped_crossing', 'boundary']
    
    # Create figure with 2 panels: GT and Predictions
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # BEV coordinate range
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    
    # ==================== Panel 1: Ground Truth ====================
    ax_gt = axes[0]
    ax_gt.set_xlim(x_min, x_max)
    ax_gt.set_ylim(y_min, y_max)
    ax_gt.set_aspect('equal')
    ax_gt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    num_gt_vecs = len(gt_data['vectors'])
    total_gt_pts = sum(len(v) for v in gt_data['vectors'])
    ax_gt.set_title(f'Ground Truth\n{num_gt_vecs} vectors', fontsize=12, fontweight='bold')
    ax_gt.set_xlabel('X (meters)', fontsize=10)
    ax_gt.set_ylabel('Y (meters)', fontsize=10)
    ax_gt.tick_params(labelsize=8)
    
    # Add coordinate axes
    ax_gt.axhline(y=0, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=1)
    ax_gt.axvline(x=0, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=1)
    
    # Plot GT vectors - GROUP BY CLASS for consistent drawing order
    for class_id in range(len(colors_plt)):  # Draw class 0, then 1, then 2
        for vector, label in zip(gt_data['vectors'], gt_data['labels']):
            if int(label) != class_id:
                continue
            if len(vector) >= 2:
                color = colors_plt[int(label)]
                # Plot as solid line
                ax_gt.plot(vector[:, 0], vector[:, 1], color=color, linewidth=2.5, 
                          alpha=0.8, linestyle='-', solid_capstyle='round')
                # Plot all points along the vector
                ax_gt.scatter(vector[:, 0], vector[:, 1], color=color, s=15, alpha=0.6,
                             edgecolors='white', linewidths=0.5, zorder=4)
                # Mark start point (circular for GT, larger and more prominent)
                ax_gt.scatter(vector[0, 0], vector[0, 1], color=color, s=40, marker='o',
                             edgecolors='white', linewidths=1.5, zorder=5)
    
    # ==================== Panel 2: Predictions ====================
    ax_pred = axes[1]
    ax_pred.set_xlim(x_min, x_max)
    ax_pred.set_ylim(y_min, y_max)
    ax_pred.set_aspect('equal')
    ax_pred.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Filter predictions by confidence
    if isinstance(pred_vectors, list):
        # Handle list of vectors (from FOV clipping)
        confident_vectors = []
        confident_labels = []
        confident_scores = []
        for vec, label, score in zip(pred_vectors, pred_labels, pred_scores):
            if score >= confidence_threshold:
                confident_vectors.append(vec)
                confident_labels.append(label)
                confident_scores.append(score)
        num_confident = len(confident_vectors)
    else:
        # Handle numpy array
        confident_mask = pred_scores >= confidence_threshold
        confident_vectors = pred_vectors[confident_mask]
        confident_labels = pred_labels[confident_mask]
        confident_scores = pred_scores[confident_mask]
        num_confident = confident_mask.sum()
    
    ax_pred.set_title(f'Model Predictions\n{num_confident} confident (>{confidence_threshold})', 
                     fontsize=12, fontweight='bold')
    ax_pred.set_xlabel('X (meters)', fontsize=10)
    ax_pred.set_ylabel('Y (meters)', fontsize=10)
    ax_pred.tick_params(labelsize=8)
    
    # Add coordinate axes
    ax_pred.axhline(y=0, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=1)
    ax_pred.axvline(x=0, color='black', linewidth=1.5, linestyle='-', alpha=0.6, zorder=1)
    
    # Plot prediction vectors - GROUP BY CLASS for consistent drawing order
    for class_id in range(len(colors_plt)):  # Draw class 0, then 1, then 2
        for vector, label, score in zip(confident_vectors, confident_labels, confident_scores):
            if int(label) != class_id:
                continue
            if len(vector) >= 2:
                color = colors_plt[int(label)]
                # Plot as solid line, alpha based on confidence
                ax_pred.plot(vector[:, 0], vector[:, 1], color=color, linewidth=2.5,
                           alpha=min(0.9, score + 0.2), linestyle='-', solid_capstyle='round')
                # Plot all points along the predicted vector
                ax_pred.scatter(vector[:, 0], vector[:, 1], color=color, s=15,
                               alpha=min(0.6, score + 0.1), edgecolors='white', 
                               linewidths=0.5, zorder=4)
                # Mark start point (square marker for predictions, larger)
                ax_pred.scatter(vector[0, 0], vector[0, 1], color=color, s=40, marker='s',
                               edgecolors='white', linewidths=1.5, 
                               alpha=min(0.9, score + 0.2), zorder=5)
    
    # Add legend positioned at bottom center to avoid overlap
    legend_elements = [
        plt.Line2D([0], [0], color=colors_plt[i], lw=2.5, label=class_names[i])
        for i in range(len(class_names))
    ]
    ax_pred.legend(handles=legend_elements, loc='lower center', fontsize=9,
                  framealpha=0.95, edgecolor='black', fancybox=True, shadow=True,
                  bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize GeMap predictions with GT')
    parser.add_argument('--nuscenes-path', type=str, 
                      default='/home/runw/Project/data/mini/nuscenes',
                      help='Path to nuScenes dataset')
    parser.add_argument('--samples-pkl', type=str,
                      default='/home/runw/Project/data/mini/nuscenes/nuscenes_infos_temporal_val.pkl',
                      help='Path to samples pickle file')
    parser.add_argument('--predictions-pkl', type=str,
                      default='gemap_predictions.pkl',
                      help='Path to predictions pickle file')
    parser.add_argument('--output-dir', type=str, default='vis_maptr_with_points',
                      help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=100,
                      help='Number of samples to visualize')
    parser.add_argument('--confidence-threshold', type=float, default=0.3,
                      help='Confidence threshold for predictions')
    parser.add_argument('--pc-range', type=float, nargs=6,
                      default=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                      help='Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]')
    parser.add_argument('--fov-clip', action='store_true', default=True,
                      help='Enable camera-specific FOV clipping and rotation (default: True, use --no-fov-clip to disable)')
    parser.add_argument('--no-fov-clip', action='store_false', dest='fov_clip',
                      help='Disable camera-specific FOV clipping (use BEV mode)')
    parser.add_argument('--camera', type=str, default='CAM_FRONT',
                      choices=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                              'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                      help='Camera to use for FOV clipping (requires --fov-clip)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load samples
    print(f"Loading samples from {args.samples_pkl}...")
    with open(args.samples_pkl, 'rb') as f:
        samples_data = pickle.load(f)
    samples = samples_data['infos']
    print(f"Loaded {len(samples)} samples")
    
    # Load predictions
    print(f"Loading predictions from {args.predictions_pkl}...")
    with open(args.predictions_pkl, 'rb') as f:
        predictions_data = pickle.load(f)
    
    # Debug: Check prediction structure
    print(f"\nDEBUG: First 3 prediction tokens:")
    for i, token in enumerate(list(predictions_data.keys())[:3]):
        print(f"  Pred token {i}: {token}")
    
    print(f"\nDEBUG: First 3 sample tokens:")
    for i, sample in enumerate(samples[:3]):
        print(f"  Sample token {i}: {sample['token']}")
    
    # Match predictions to samples by token
    predictions_by_token = predictions_data
    print(f"Loaded predictions for {len(predictions_by_token)} samples")
    
    # Visualize samples
    mode_str = f"FOV-clipped ({args.camera})" if args.fov_clip else "BEV"
    print(f"\nVisualizing {args.num_samples} samples in {mode_str} mode...")
    
    visualized_count = 0
    
    for sample_idx, sample_info in enumerate(tqdm(samples[:args.num_samples], desc="Visualizing")):
        try:
            sample_token = sample_info['token']
            
            # Get predictions for this sample
            if sample_token not in predictions_by_token:
                print(f"Warning: No predictions for sample {sample_token}")
                continue
            
            pred_data = predictions_by_token[sample_token]
            pred_vectors = pred_data['vectors']  # [N, num_pts, 2]
            pred_labels = pred_data['labels']    # [N]
            pred_scores = pred_data['scores']    # [N]
            
            # Extract ground truth
            if args.fov_clip:
                # Extract GT with FOV clipping and rotation
                gt_data = extract_gt_with_fov_clipping(
                    sample_info=sample_info,
                    nuscenes_path=args.nuscenes_path,
                    pc_range=args.pc_range,
                    camera_name=args.camera,
                    fixed_num=20
                )
                
                # Process predictions with same FOV clipping and rotation
                pred_vectors_processed, pred_labels_processed, pred_scores_processed = \
                    process_predictions_with_fov_clipping(
                        pred_vectors=pred_vectors,
                        pred_labels=pred_labels,
                        pred_scores=pred_scores,
                        sample_info=sample_info,
                        nuscenes_path=args.nuscenes_path,
                        pc_range=args.pc_range,
                        camera_name=args.camera
                    )
                
                # Keep as list for visualization
                pred_vectors = pred_vectors_processed
                pred_labels = np.array(pred_labels_processed)
                pred_scores = np.array(pred_scores_processed)
            else:
                # Standard BEV visualization (no FOV clipping)
                gt_data = extract_gt_vectors(
                    sample_info=sample_info,
                    nuscenes_path=args.nuscenes_path,
                    pc_range=args.pc_range,
                    fixed_num=20
                )
            
            # Save visualization
            save_filename = f"sample_{sample_idx:03d}_{sample_token}"
            if args.fov_clip:
                save_filename += f"_{args.camera}"
            save_filename += ".png"
            save_path = output_dir / save_filename
            
            visualize_sample(
                sample_info=sample_info,
                pred_vectors=pred_vectors,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                gt_data=gt_data,
                save_path=str(save_path),
                pc_range=args.pc_range,
                confidence_threshold=args.confidence_threshold
            )
            
            visualized_count += 1
            
        except Exception as e:
            print(f"Error visualizing sample {sample_idx} ({sample_info.get('token', 'unknown')}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Visualized {visualized_count} samples")
    print(f"✓ Saved to: {output_dir}")


if __name__ == '__main__':
    main()
