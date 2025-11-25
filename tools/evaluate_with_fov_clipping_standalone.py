#!/usr/bin/env python3
"""
Evaluate GeMap predictions with camera-specific FOV clipping and rotation.
STANDALONE VERSION - Uses EXACT MapTR official evaluation code

This script applies camera-specific preprocessing to both GT and predictions:
1. Clip both GT and predictions to each camera's FOV
2. Rotate to camera-centric coordinates (camera forward = +Y)
3. Compute metrics using EXACT MapTR official method:
   - STRtree spatial indexing for Chamfer distance computation
   - Only computes CD for intersecting buffered geometries (linewidth=2m)
   - Negative CD convention (higher = better)
   - Greedy confidence-sorted matching

All FOV clipping logic imported from camera_fov_utils.py.
All evaluation logic matches MapTR's tpfp.py and tpfp_chamfer.py exactly.
"""

import sys
from pathlib import Path
import argparse
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import json

# Import NuScenes and geometry utilities
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely.geometry import LineString, Point, CAP_STYLE, JOIN_STYLE
from shapely.strtree import STRtree
from scipy.spatial.transform import Rotation
from scipy.spatial import distance

# Import shared camera FOV utilities (ensures identical logic with visualization)
from camera_fov_utils import (
    VectorizedLocalMap,
    CameraFOVClipper,
    extract_gt_vectors,
    extract_gt_with_fov_clipping,
    process_predictions_with_fov_clipping
)

# Add GeMap project path for evaluation utilities
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import MapTR's Chamfer distance implementation (if available)
try:
    from projects.mmdet3d_plugin.datasets.map_utils.tpfp_chamfer import custom_polyline_score
except ImportError:
    print("Warning: Could not import MapTR's custom_polyline_score. Using fallback implementation.")
    custom_polyline_score = None


# ==================== EVALUATION-SPECIFIC CODE ====================
class CameraSpecificEvaluator:
    """
    Evaluator using EXACT MapTR official evaluation method.
    
    Applies camera-specific FOV clipping and rotation to both GT and predictions,
    then evaluates using MapTR's official matching algorithm.
    
    MapTR NEGATIVE Chamfer Distance Convention:
    ==========================================
    - All Chamfer distances stored as NEGATIVE values
    - Example CD matrix: [[-0.3, -1.5], [-2.0, -0.8]]
    - Higher (less negative) = better match: -0.3 > -1.5 ✓
    - Non-intersecting pairs: -100.0 (guaranteed to not match)
    
    Why Negative?
    - Allows max() to find best match: max([-0.3, -1.5]) = -0.3
    - Consistent with score-based ranking (higher = better)
    - Threshold comparison: cd >= -threshold (e.g., -0.3 >= -0.5 ✓)
    
    Threshold Convention:
    ====================
    - User provides POSITIVE thresholds (e.g., [0.5, 1.0, 1.5] meters)
    - Internally converted to NEGATIVE (e.g., [-0.5, -1.0, -1.5])
    - Matching: if cd_score >= negative_threshold
    
    Example:
    --------
    CD = 0.35m (actual distance) -> stored as -0.35
    Threshold = 0.5m -> converted to -0.5
    Match check: -0.35 >= -0.5 ? YES ✓ (close enough)
    
    CD = 2.0m (actual distance) -> stored as -2.0
    Threshold = 0.5m -> converted to -0.5
    Match check: -2.0 >= -0.5 ? NO ✗ (too far)
    """
    
    def __init__(
        self,
        nuscenes_data_path: str,
        pc_range: List[float] = None,
        num_sample_pts: int = 100,
        thresholds_chamfer: List[float] = None,
        camera_names: List[str] = None
    ):
        """
        Args:
            nuscenes_data_path: Path to NuScenes dataset
            pc_range: BEV range [-x, -y, -z, x, y, z]
            num_sample_pts: Number of points to resample vectors to (MUST match training: 100)
            thresholds_chamfer: Chamfer distance thresholds (MapTR uses [0.5, 1.0, 1.5])
            camera_names: List of camera names to evaluate (default: ['CAM_FRONT'] only)
        """
        self.nuscenes_data_path = nuscenes_data_path
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.num_sample_pts: int = num_sample_pts
        self.thresholds_chamfer = thresholds_chamfer or [0.5, 1.0, 1.5]
        self.camera_names = camera_names or ['CAM_FRONT']
        
        # Calculate patch size from pc_range
        self.patch_size = (self.pc_range[4] - self.pc_range[1], self.pc_range[3] - self.pc_range[0])
        
        # Accumulators for per-camera metrics
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
            # Handle degenerate cases
            if num_sample > len(vector):
                padding = np.zeros((num_sample - len(vector), 2))
                return np.vstack([vector, padding])
            return vector
        
        # Create LineString and sample evenly
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
        Uses shared extract_gt_with_fov_clipping() for 100% identical logic with visualization.
        
        Args:
            apply_clipping: If True, apply FOV clipping; if False, skip clipping
        
        Returns:
            vectors: (N, num_pts, 2) array of GT vectors
            labels: (N,) array of GT labels
        """
        # Use shared function for GT extraction + optional FOV clipping + rotation
        gt_data = extract_gt_with_fov_clipping(
            sample_info=sample_info,
            nuscenes_path=self.nuscenes_data_path,
            pc_range=self.pc_range,
            camera_name=camera_name,
            fixed_num=20,  # Initial MapTR resampling to 20 points
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
        Apply optional FOV clipping AND camera-centric rotation to prediction vectors.
        Uses shared process_predictions_with_fov_clipping() for 100% identical logic with visualization.
        
        Args:
            apply_clipping: If True, apply FOV clipping; if False, skip clipping
        
        Returns:
            vectors: (N, num_pts, 2) optionally FOV-clipped and rotated prediction vectors
            labels: (N,) prediction labels
            scores: (N,) prediction scores (tracked through processing)
        """
        if len(pred_vectors) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Use shared function for optional FOV clipping + rotation
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
                                                        linewidth: float = 2.0) -> np.ndarray:
        """
        Compute Chamfer Distance matrix using EXACT MapTR official method.
        EXACT copy from MapTR's tpfp_chamfer.py:custom_polyline_score()
        
        Key Features (EXACT MapTR):
        1. STRtree spatial indexing - only computes CD for geometrically plausible pairs
        2. Buffer each line with linewidth (default: 2m) for intersection checking
        3. Returns NEGATIVE CD values (higher = better match)
           - Good match: -0.3 (close)
           - Poor match: -2.0 (far)
           - Non-intersecting: -100.0 (guaranteed to not match)
        
        Why Negative?
        - Allows using max() to find best match: max([-0.3, -1.5]) = -0.3 ✓
        - Consistent with score-based matching (higher = better)
        
        Args:
            pred_vectors: [num_preds, num_pts, 2] predicted vectors in meters
            gt_vectors: [num_gts, num_pts, 2] ground truth vectors in meters
            linewidth: Buffer width in meters for spatial filtering (MapTR default: 2.0)
            
        Returns:
            cd_matrix: [num_preds, num_gts] NEGATIVE Chamfer distance matrix
                      Values: typically -0.1 to -5.0 for matches, -100.0 for non-intersecting
        """
        num_preds = len(pred_vectors)
        num_gts = len(gt_vectors)
        
        if num_preds == 0 or num_gts == 0:
            return np.full((num_preds, num_gts), -100.0)
        
        # Create buffered shapely geometries (EXACT MapTR method)
        pred_lines_shapely = [
            LineString(pred_vectors[i]).buffer(
                linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
            for i in range(num_preds)
        ]
        gt_lines_shapely = [
            LineString(gt_vectors[i]).buffer(
                linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
            for i in range(num_gts)
        ]
        
        # Construct STRtree for spatial indexing (EXACT MapTR method)
        # O(log n) spatial queries instead of O(n²) brute force
        tree = STRtree(pred_lines_shapely)
        
        # Initialize with -100.0 for non-intersecting pairs (EXACT MapTR convention)
        # -100.0 is so negative that it will never match any threshold
        cd_matrix = np.full((num_preds, num_gts), -100.0)
        
        # Compute CD only for intersecting buffered geometries (EXACT MapTR method)
        # This skips ~80-95% of computations for typical scenes
        # Note: STRtree.query() behavior differs between Shapely versions:
        #   - Shapely 1.x: returns geometry objects (need to match by intersection)
        #   - Shapely 2.x: returns indices (can use directly)
        for i, gt_line in enumerate(gt_lines_shapely):
            query_result = tree.query(gt_line)
            
            # Check if query returns indices (Shapely 2.x) or geometries (Shapely 1.x)
            if len(query_result) > 0 and isinstance(query_result[0], (int, np.integer)):
                # Shapely 2.x: query returns indices
                for pred_idx in query_result:
                    pred_line = pred_lines_shapely[pred_idx]
                    
                    if pred_line.intersects(gt_line):
                        # Compute Chamfer Distance (positive distance in meters)
                        dist_mat = distance.cdist(
                            pred_vectors[pred_idx], gt_vectors[i], 'euclidean')
                        valid_ab = dist_mat.min(axis=1).mean()  # pred -> gt (e.g., 0.3m)
                        valid_ba = dist_mat.min(axis=0).mean()  # gt -> pred (e.g., 0.4m)
                        
                        # Store NEGATIVE CD (EXACT MapTR convention)
                        cd_matrix[pred_idx, i] = -(valid_ab + valid_ba) / 2.0
            else:
                # Shapely 1.x: query returns geometries, need to find matching index
                for pred_idx in range(num_preds):
                    pred_line = pred_lines_shapely[pred_idx]
                    
                    if pred_line.intersects(gt_line):
                        # Compute Chamfer Distance (positive distance in meters)
                        dist_mat = distance.cdist(
                            pred_vectors[pred_idx], gt_vectors[i], 'euclidean')
                        valid_ab = dist_mat.min(axis=1).mean()  # pred -> gt (e.g., 0.3m)
                        valid_ba = dist_mat.min(axis=0).mean()  # gt -> pred (e.g., 0.4m)
                        
                        # Store NEGATIVE CD (EXACT MapTR convention)
                        cd_matrix[pred_idx, i] = -(valid_ab + valid_ba) / 2.0
        
        return cd_matrix
    
    def compute_chamfer_distance_torch(self,
                                       pred_vectors: np.ndarray,
                                       gt_vectors: np.ndarray) -> float:
        """
        Compute Chamfer Distance for monitoring (returns POSITIVE distance for readability).
        This is ONLY used for logging/monitoring, NOT for matching.
        
        Args:
            pred_vectors: [N_pred, num_pts, 2] predicted vectors
            gt_vectors: [N_gt, num_pts, 2] ground truth vectors
        Returns:
            chamfer_dist: POSITIVE Chamfer distance in meters (for logging)
        """
        if len(pred_vectors) == 0 or len(gt_vectors) == 0:
            return float('inf')
        
        # Flatten points
        pred_points = pred_vectors.reshape(-1, 2)
        gt_points = gt_vectors.reshape(-1, 2)
        
        # Compute pairwise distances
        dist_matrix = distance.cdist(pred_points, gt_points, 'euclidean')
        
        # Forward & Backward Chamfer (EXACT MapTR computation)
        valid_ab = dist_matrix.min(axis=1).mean()  # pred -> gt
        valid_ba = dist_matrix.min(axis=0).mean()  # gt -> pred
        
        # Return POSITIVE for logging (NOT used in matching)
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
        
        Args:
            apply_clipping: If True, apply FOV clipping to both GT and predictions
        """
        for camera_name in self.camera_names:
            # Process GT with optional FOV clipping (uses shared function)
            gt_vectors, gt_labels = self.process_gt_with_fov_clipping(
                sample_info, camera_name, apply_clipping=apply_clipping)
            
            # Process predictions with optional FOV clipping (uses shared function)
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
                                               threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match predictions to GT using MapTR's EXACT OFFICIAL method.
        EXACT copy from MapTR's tpfp.py:custom_tpfp_gen()
        
        MapTR Convention:
        - Chamfer distances are NEGATIVE (higher = better match, e.g., -0.5 > -1.0)
        - Threshold is provided as POSITIVE (e.g., 0.5m) but converted to NEGATIVE
        - Matching: matrix_max[i] >= threshold (e.g., -0.3 >= -0.5 ✓ match)
        
        Args:
            pred_vectors: [N_pred, num_pts, 2] predicted vectors
            pred_scores: [N_pred] confidence scores  
            gt_vectors: [N_gt, num_pts, 2] ground truth vectors
            threshold: Distance threshold in meters (POSITIVE, e.g., 0.5, 1.0, 1.5)
        Returns:
            tp: [N_pred] binary array (1 = true positive, 0 = false positive)
            fp: [N_pred] binary array (1 = false positive, 0 = true positive)
        """
        num_preds = len(pred_vectors)
        num_gts = len(gt_vectors)
        
        # Initialize tp and fp arrays (EXACT MapTR)
        tp = np.zeros(num_preds, dtype=np.float32)
        fp = np.zeros(num_preds, dtype=np.float32)
        
        # If no GT, all predictions are FP (EXACT MapTR)
        if num_gts == 0:
            fp[:] = 1
            return tp, fp
        
        # If no predictions, return empty (EXACT MapTR)
        if num_preds == 0:
            return tp, fp
        
        # Convert threshold to NEGATIVE (EXACT MapTR convention)
        # e.g., 0.5 -> -0.5, so we can compare with negative CD values
        if threshold > 0:
            threshold = -threshold
        
        # Compute CD matrix using EXACT MapTR method (STRtree filtered)
        # Returns NEGATIVE CD values: -0.3, -1.5, etc.
        # Non-intersecting pairs: -100.0 (guaranteed not to match)
        cd_matrix = self.compute_chamfer_distance_matrix_maptr_official(
            pred_vectors, gt_vectors, linewidth=2.0)
        
        # For each prediction, find best matching GT
        # max() finds HIGHEST (least negative) CD = best match
        # e.g., [-0.3, -1.5, -2.0] -> max = -0.3
        matrix_max = cd_matrix.max(axis=1)  # [num_preds]
        matrix_argmax = cd_matrix.argmax(axis=1)  # [num_preds]
        
        # Sort predictions by CONFIDENCE (descending) - EXACT MapTR
        sort_inds = np.argsort(-pred_scores)
        
        # Track which GTs have been matched - EXACT MapTR
        gt_covered = np.zeros(num_gts, dtype=bool)
        
        # Greedy matching in CONFIDENCE order (EXACT MapTR method)
        # Process high-confidence predictions first
        for i in sort_inds:
            # Check if CD is good enough: e.g., -0.3 >= -0.5 ✓
            if matrix_max[i] >= threshold:
                matched_gt = matrix_argmax[i]
                if not gt_covered[matched_gt]:
                    # Valid match: CD good enough and GT not taken
                    gt_covered[matched_gt] = True
                    tp[i] = 1
                else:
                    # GT already matched by higher-confidence prediction
                    fp[i] = 1
            else:
                # CD too poor: e.g., -2.0 >= -0.5 ✗ (too far)
                fp[i] = 1
        
        return tp, fp
    
    def compute_ap_area_based(self,
                              recalls: np.ndarray,
                              precisions: np.ndarray) -> float:
        """
        Compute Average Precision using area under PR curve (MapTR official method).
        
        Args:
            recalls: [N] recall values
            precisions: [N] precision values
        Returns:
            ap: Average Precision
        """
        # Add boundary values
        mrec = np.concatenate([[0], recalls, [1]])
        mpre = np.concatenate([[0], precisions, [0]])
        
        # Make precision monotonically decreasing
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        # Compute area where recall changes
        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
        
        return float(ap)
    
    def compute_ap_for_class(self,
                            pred_vectors_list: List[np.ndarray],
                            pred_scores_list: List[np.ndarray],
                            gt_vectors_list: List[np.ndarray],
                            threshold: float) -> Tuple[float, float]:
        """
        Compute AP and average CD for a single class at given threshold.
        Uses EXACT MapTR official matching method.
        
        Args:
            pred_vectors_list: List of [N_pred, num_pts, 2] arrays for each sample
            pred_scores_list: List of [N_pred] confidence scores for each sample
            gt_vectors_list: List of [N_gt, num_pts, 2] arrays for each sample
            threshold: Distance threshold in meters (POSITIVE, e.g., 0.5, 1.0, 1.5)
                      Automatically converted to NEGATIVE internally
        Returns:
            ap: Average Precision using area-based method (EXACT MapTR)
            avg_cd: Average Chamfer Distance (POSITIVE for logging)
        """
        # Count total GTs
        num_gts = sum(len(gts) for gts in gt_vectors_list)
        
        if num_gts == 0:
            return 0.0, float('inf')
        
        # Compute TP/FP for each sample
        all_tp = []
        all_fp = []
        all_scores = []
        chamfer_distances_per_sample = []
        
        for pred_vecs, pred_scores, gt_vecs in zip(pred_vectors_list, pred_scores_list, gt_vectors_list):
            if len(pred_vecs) == 0:
                continue
            
            if len(gt_vecs) == 0:
                # No GT: all predictions are FP
                all_tp.append(np.zeros(len(pred_vecs), dtype=np.float32))
                all_fp.append(np.ones(len(pred_vecs), dtype=np.float32))
                all_scores.append(pred_scores)
                continue
            
            # Match predictions to GT (MapTR official method)
            tp, fp = self.match_predictions_to_gt_maptr_official(
                pred_vecs, pred_scores, gt_vecs, threshold)
            
            all_tp.append(tp)
            all_fp.append(fp)
            all_scores.append(pred_scores)
            
            # Compute chamfer distance for this sample
            cd_sample = self.compute_chamfer_distance_torch(pred_vecs, gt_vecs)
            chamfer_distances_per_sample.append(cd_sample)
        
        if len(all_tp) == 0:
            return 0.0, float('inf')
        
        # Concatenate all predictions globally
        all_tp = np.concatenate(all_tp)
        all_fp = np.concatenate(all_fp)
        all_scores = np.concatenate(all_scores)
        
        # Sort by confidence (descending)
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
        
        # Compute area-based AP (MapTR official)
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
            
            # Get predictions and GT for this camera
            camera_preds = self.predictions_per_camera[camera_name]
            camera_gts = self.ground_truths_per_camera[camera_name]
            
            # Evaluate each class
            for class_id, class_name in enumerate(class_names):
                class_results = {}
                
                # Extract predictions and GT for this class
                pred_vectors_list = []
                pred_scores_list = []
                gt_vectors_list = []
                
                for pred_data, gt_data in zip(camera_preds, camera_gts):
                    # Filter by class
                    pred_mask = pred_data['labels'] == class_id
                    gt_mask = gt_data['labels'] == class_id
                    
                    pred_vectors_list.append(pred_data['vectors'][pred_mask])
                    pred_scores_list.append(pred_data['scores'][pred_mask])
                    gt_vectors_list.append(gt_data['vectors'][gt_mask])
                
                # Compute AP at each threshold
                avg_cd = None
                for threshold in self.thresholds_chamfer:
                    ap, cd = self.compute_ap_for_class(
                        pred_vectors_list, pred_scores_list, gt_vectors_list, threshold)
                    
                    class_results[f'AP@{threshold}m'] = ap
                    all_aps.append(ap)
                    
                    # Store CD from first threshold computation
                    if avg_cd is None:
                        avg_cd = cd
                
                class_results['avg_chamfer_distance'] = avg_cd if avg_cd is not None else float('inf')
                
                camera_results[class_name] = class_results
            
            # Compute mAP across all classes and thresholds
            camera_results['mAP'] = np.mean(all_aps) if all_aps else 0.0
            
            results[camera_name] = camera_results
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate MapTR with camera-specific FOV clipping')
    parser.add_argument('--nuscenes-path', type=str, 
                       default='/home/runw/Project/data/mini/nuscenes',
                       help='Path to NuScenes dataset')
    parser.add_argument('--samples-pkl', type=str, 
                       default='/home/runw/Project/data/mini/nuscenes/nuscenes_infos_temporal_val.pkl',
                      help='Path to samples pickle file')
    parser.add_argument('--predictions-pkl', type=str, 
                       default='gemap_predictions.pkl',
                       help='Path to predictions pickle file')
    parser.add_argument('--output-json', type=str,
                      default='evaluation_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--pc-range', type=float, nargs=6,
                      default=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                      help='Point cloud range')
    parser.add_argument('--cameras', type=str, nargs='+',
                      default=['CAM_FRONT'],
                      help='Camera names to evaluate')
    parser.add_argument('--num-sample-pts', type=int, default=100,
                      help='Number of points to resample vectors to (default: 100)')
    parser.add_argument('--apply-clipping', action='store_true',
                      help='Apply camera FOV clipping to GT and predictions (default: False for full BEV evaluation)')
    parser.add_argument('--no-clipping', dest='apply_clipping', action='store_false',
                      help='Disable FOV clipping (full BEV evaluation)')
    parser.set_defaults(apply_clipping=True)
    
    args = parser.parse_args()
    
    print("="*80)
    print("Evaluate GeMap with Camera-Specific FOV Clipping")
    print("Using EXACT Official MapTR Evaluation Code")
    print("="*80)
    print(f"\nNuScenes path: {args.nuscenes_path}")
    print(f"Samples pickle: {args.samples_pkl}")
    print(f"Predictions pickle: {args.predictions_pkl}")
    print(f"Output: {args.output_json}")
    
    # Load samples
    print(f"\nLoading samples from {args.samples_pkl}...")
    with open(args.samples_pkl, 'rb') as f:
        samples_data = pickle.load(f)
    samples = samples_data['infos']
    print(f"Loaded {len(samples)} samples")
    
    # Load predictions
    print(f"\nLoading predictions from {args.predictions_pkl}...")
    with open(args.predictions_pkl, 'rb') as f:
        predictions_data = pickle.load(f)
    predictions_by_token = predictions_data
    print(f"Loaded predictions for {len(predictions_by_token)} samples")
    
    # Create evaluator
    evaluator = CameraSpecificEvaluator(
        nuscenes_data_path=args.nuscenes_path,
        pc_range=args.pc_range,
        num_sample_pts=args.num_sample_pts,
        thresholds_chamfer=[0.5, 1.0, 1.5],
        camera_names=args.cameras
    )
    
    print(f"Initialized evaluator with:")
    print(f"  - PC range: {args.pc_range}")
    print(f"  - Patch size: {evaluator.patch_size}")
    print(f"  - Cameras: {args.cameras}")
    print(f"  - Sample points per vector: {args.num_sample_pts} (MapTR standard)")
    print(f"  - Chamfer thresholds: {evaluator.thresholds_chamfer} meters (MapTR standard)")
    print(f"  - FOV clipping: {'ENABLED' if args.apply_clipping else 'DISABLED (full BEV)'}")
    print(f"  - Evaluation method: Official MapTR (STRtree spatial filtering, linewidth=2m)")
    
    # Evaluate all samples
    print(f"\nEvaluating {len(samples)} samples across {len(args.cameras)} cameras...")
    mode_str = "camera-specific FOV clipping" if args.apply_clipping else "full BEV (no clipping)"
    print(f"Accumulating predictions and GT with {mode_str}...")
    
    for sample_info in tqdm(samples, desc="Processing samples"):
        sample_token = sample_info['token']
        
        if sample_token not in predictions_by_token:
            continue
        
        pred_data = predictions_by_token[sample_token]
        evaluator.accumulate_sample(
            sample_info=sample_info,
            pred_vectors=pred_data['vectors'],
            pred_labels=pred_data['labels'],
            pred_scores=pred_data['scores'],
            apply_clipping=args.apply_clipping
        )
    
    # Compute metrics
    print("\nComputing metrics...")
    results = evaluator.evaluate()
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    class_names = ['divider', 'ped_crossing', 'boundary']
    
    for camera_name, camera_results in results.items():
        print(f"\n{camera_name}:")
        
        # Print mAP first
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
    print(f"\nSaving results to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Evaluation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
