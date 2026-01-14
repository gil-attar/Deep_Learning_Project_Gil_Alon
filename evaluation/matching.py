"""
IoU computation and greedy box matching for detection evaluation.
"""

from typing import List, Dict, Tuple
import numpy as np


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2] in xyxy format
        box2: [x1, y1, x2, y2] in xyxy format

    Returns:
        IoU score (0.0 to 1.0)
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Compute intersection area
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Compute union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def greedy_match_boxes(
    pred_boxes: List[Dict],
    gt_boxes: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy matching of predicted boxes to ground truth boxes.

    Matching strategy:
    - Compute IoU between all pred-GT pairs
    - Sort pairs by IoU descending
    - Greedily assign matches (one-to-one)
    - Match is valid if IoU >= iou_threshold

    Args:
        pred_boxes: List of predicted detections with 'bbox' key
        gt_boxes: List of ground truth boxes with 'bbox_xyxy' key
        iou_threshold: Minimum IoU for a valid match

    Returns:
        matches: List of (pred_idx, gt_idx) tuples for matched pairs
        unmatched_preds: List of pred indices without a match (FP)
        unmatched_gts: List of GT indices without a match (FN)
    """
    if len(pred_boxes) == 0:
        # No predictions: all GTs are unmatched (FN)
        return [], [], list(range(len(gt_boxes)))

    if len(gt_boxes) == 0:
        # No ground truth: all predictions are unmatched (FP)
        return [], list(range(len(pred_boxes))), []

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        pred_bbox = pred.get('bbox', pred.get('bbox_xyxy'))
        for j, gt in enumerate(gt_boxes):
            gt_bbox = gt.get('bbox_xyxy', gt.get('bbox'))
            iou_matrix[i, j] = compute_iou(pred_bbox, gt_bbox)

    # Get all candidate pairs with IoU >= threshold
    candidates = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                candidates.append((i, j, iou_matrix[i, j]))

    # Sort by IoU descending (greedy: prefer highest IoU matches)
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Greedily assign matches (one-to-one)
    matched_preds = set()
    matched_gts = set()
    matches = []

    for pred_idx, gt_idx, iou in candidates:
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matches.append((pred_idx, gt_idx))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)

    # Find unmatched predictions and ground truths
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


def match_predictions_to_ground_truth(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.0
) -> Dict:
    """
    Match predictions to ground truth across all images.

    Matching is done per-image, per-class for accurate TP/FP/FN counting.

    Args:
        predictions: List of per-image predictions
        ground_truths: List of per-image ground truths
        iou_threshold: Minimum IoU for a valid match
        conf_threshold: Minimum confidence to consider a prediction

    Returns:
        Dictionary with:
        - matches: List of (image_id, pred_det, gt_obj) tuples for TPs
        - false_positives: List of (image_id, pred_det) tuples for FPs
        - false_negatives: List of (image_id, gt_obj) tuples for FNs
    """
    # Create image_id lookup for ground truth
    gt_by_image = {img['image_id']: img for img in ground_truths}

    matches = []
    false_positives = []
    false_negatives = []

    for pred_img in predictions:
        image_id = pred_img['image_id']
        pred_dets = pred_img.get('detections', [])

        # Filter by confidence threshold
        pred_dets = [d for d in pred_dets if d.get('confidence', 0) >= conf_threshold]

        # Get corresponding ground truth
        gt_img = gt_by_image.get(image_id)
        if gt_img is None:
            # No ground truth for this image: all predictions are FP
            for det in pred_dets:
                false_positives.append((image_id, det))
            continue

        gt_objs = gt_img.get('ground_truth', [])

        # Group predictions and GTs by class (matching must be per-class)
        pred_by_class = {}
        for det in pred_dets:
            class_id = det['class_id']
            if class_id not in pred_by_class:
                pred_by_class[class_id] = []
            pred_by_class[class_id].append(det)

        gt_by_class = {}
        for obj in gt_objs:
            class_id = obj['class_id']
            if class_id not in gt_by_class:
                gt_by_class[class_id] = []
            gt_by_class[class_id].append(obj)

        # Match per class
        all_classes = set(pred_by_class.keys()) | set(gt_by_class.keys())

        for class_id in all_classes:
            class_preds = pred_by_class.get(class_id, [])
            class_gts = gt_by_class.get(class_id, [])

            # Greedy matching
            class_matches, class_fp_indices, class_fn_indices = greedy_match_boxes(
                class_preds, class_gts, iou_threshold
            )

            # Record matches (TP)
            for pred_idx, gt_idx in class_matches:
                matches.append((image_id, class_preds[pred_idx], class_gts[gt_idx]))

            # Record false positives
            for pred_idx in class_fp_indices:
                false_positives.append((image_id, class_preds[pred_idx]))

            # Record false negatives
            for gt_idx in class_fn_indices:
                false_negatives.append((image_id, class_gts[gt_idx]))

    return {
        'matches': matches,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
