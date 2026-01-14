"""
Core evaluation metrics for object detection.

Implements three main evaluation functions:
1. eval_detection_prf_at_iou: Precision/Recall/F1 at multiple confidence thresholds
2. eval_per_class_metrics_and_confusions: Per-class metrics + confusion matrix
3. eval_counting_quality: Counting accuracy (MAE) for duplicate objects
"""

from typing import List, Dict
from collections import defaultdict
import numpy as np
from .matching import match_predictions_to_ground_truth


def eval_detection_prf_at_iou(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    conf_thresholds: List[float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate Precision, Recall, F1 at multiple confidence thresholds.

    This metric evaluates box-level correctness:
    - TP: Predicted box matched to GT box with IoU >= iou_threshold
    - FP: Predicted box not matched to any GT
    - FN: GT box not matched to any prediction

    Matching strategy:
    - One-to-one greedy matching per image per class
    - Matches sorted by IoU descending

    Args:
        predictions: List of per-image predictions from load_predictions()
        ground_truths: List of per-image ground truths from load_ground_truth()
        iou_threshold: Minimum IoU for a valid match (default: 0.5)
        conf_thresholds: List of confidence thresholds to evaluate
                        (default: [0.0, 0.1, 0.2, ..., 0.9])

    Returns:
        Dictionary mapping conf_threshold (str) to metrics:
        {
            "0.0": {"precision": 0.45, "recall": 0.92, "f1": 0.60, "tp": 850, "fp": 1040, "fn": 70},
            "0.5": {"precision": 0.82, "recall": 0.75, "f1": 0.78, "tp": 643, "fp": 141, "fn": 213},
            ...
        }

    Example:
        >>> results = eval_detection_prf_at_iou(preds, gts, iou_threshold=0.5)
        >>> print(f"At conf=0.5: F1={results['0.5']['f1']:.3f}")
    """
    if conf_thresholds is None:
        conf_thresholds = [round(x * 0.1, 1) for x in range(10)]  # 0.0 to 0.9

    results = {}

    for conf_thr in conf_thresholds:
        # Match predictions to ground truth at this confidence threshold
        matching_result = match_predictions_to_ground_truth(
            predictions, ground_truths, iou_threshold, conf_thr
        )

        tp = len(matching_result['matches'])
        fp = len(matching_result['false_positives'])
        fn = len(matching_result['false_negatives'])

        # Compute P/R/F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[str(conf_thr)] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    return results


def eval_per_class_metrics_and_confusions(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
    class_names: Dict[int, str] = None
) -> Dict:
    """
    Evaluate per-class Precision/Recall/F1 and build confusion matrix.

    Confusion matrix:
    - Only includes MATCHED pairs (IoU >= iou_threshold)
    - Rows = true class, Cols = predicted class
    - Unmatched GT → FN (not in confusion matrix)
    - Unmatched pred → FP (not in confusion matrix)

    Args:
        predictions: List of per-image predictions
        ground_truths: List of per-image ground truths
        iou_threshold: Minimum IoU for a valid match (default: 0.5)
        conf_threshold: Confidence threshold for predictions (default: 0.5)
        class_names: Dict mapping class_id to class_name (optional)

    Returns:
        {
            "per_class": {
                "Asparagus": {
                    "precision": 0.80,
                    "recall": 0.70,
                    "f1": 0.75,
                    "support": 13,  # num GT objects
                    "tp": 9,
                    "fp": 2,
                    "fn": 4
                },
                ...
            },
            "confusion_matrix": [[...], ...],  # rows=true, cols=pred
            "class_order": ["Asparagus", "Avocado", ...],
            "top_confusions": [
                {"true_class": "Garlic", "pred_class": "Onion", "count": 5},
                ...
            ]
        }

    Example:
        >>> results = eval_per_class_metrics_and_confusions(preds, gts, conf_threshold=0.5)
        >>> print(f"Asparagus F1: {results['per_class']['Asparagus']['f1']:.3f}")
    """
    # Match predictions to ground truth
    matching_result = match_predictions_to_ground_truth(
        predictions, ground_truths, iou_threshold, conf_threshold
    )

    # Auto-detect class names if not provided
    if class_names is None:
        class_names = {}
        for gt_img in ground_truths:
            for obj in gt_img.get('ground_truth', []):
                class_id = obj['class_id']
                class_name = obj.get('class_name', f'class_{class_id}')
                class_names[class_id] = class_name

    # Count TP, FP, FN per class
    tp_per_class = defaultdict(int)
    fp_per_class = defaultdict(int)
    fn_per_class = defaultdict(int)
    support_per_class = defaultdict(int)

    # Count support (total GT objects per class)
    for gt_img in ground_truths:
        for obj in gt_img.get('ground_truth', []):
            class_id = obj['class_id']
            support_per_class[class_id] += 1

    # Count TP (matches)
    for image_id, pred_det, gt_obj in matching_result['matches']:
        class_id = gt_obj['class_id']
        tp_per_class[class_id] += 1

    # Count FP (unmatched predictions)
    for image_id, pred_det in matching_result['false_positives']:
        class_id = pred_det['class_id']
        fp_per_class[class_id] += 1

    # Count FN (unmatched ground truths)
    for image_id, gt_obj in matching_result['false_negatives']:
        class_id = gt_obj['class_id']
        fn_per_class[class_id] += 1

    # Compute per-class P/R/F1
    per_class_metrics = {}
    all_class_ids = sorted(set(tp_per_class.keys()) | set(fp_per_class.keys()) | set(fn_per_class.keys()) | set(support_per_class.keys()))

    for class_id in all_class_ids:
        tp = tp_per_class[class_id]
        fp = fp_per_class[class_id]
        fn = fn_per_class[class_id]
        support = support_per_class[class_id]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        class_name = class_names.get(class_id, f'class_{class_id}')

        per_class_metrics[class_name] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    # Build confusion matrix (only from matched pairs)
    confusion_pairs = defaultdict(int)
    for image_id, pred_det, gt_obj in matching_result['matches']:
        true_class = gt_obj['class_id']
        pred_class = pred_det['class_id']
        confusion_pairs[(true_class, pred_class)] += 1

    # Convert to matrix format
    class_order = [class_names.get(cid, f'class_{cid}') for cid in sorted(class_names.keys())]
    class_id_to_idx = {cid: idx for idx, cid in enumerate(sorted(class_names.keys()))}

    num_classes = len(class_order)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for (true_id, pred_id), count in confusion_pairs.items():
        if true_id in class_id_to_idx and pred_id in class_id_to_idx:
            true_idx = class_id_to_idx[true_id]
            pred_idx = class_id_to_idx[pred_id]
            confusion_matrix[true_idx, pred_idx] = count

    # Extract top confusions (off-diagonal elements)
    confusions_list = []
    for (true_id, pred_id), count in confusion_pairs.items():
        if true_id != pred_id:  # Only off-diagonal (true confusions)
            confusions_list.append({
                'true_class': class_names.get(true_id, f'class_{true_id}'),
                'pred_class': class_names.get(pred_id, f'class_{pred_id}'),
                'count': count
            })

    # Sort by count descending
    confusions_list.sort(key=lambda x: x['count'], reverse=True)

    return {
        'per_class': per_class_metrics,
        'confusion_matrix': confusion_matrix.tolist(),
        'class_order': class_order,
        'top_confusions': confusions_list[:20]  # Top 20 confusions
    }


def eval_counting_quality(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
    class_names: Dict[int, str] = None
) -> Dict:
    """
    Evaluate counting accuracy for images with duplicate objects.

    This metric is important for cases like "2 carrots, 3 potatoes".
    We evaluate whether the model detects ALL objects, not just one.

    Counting methods:
    - "matched_only": pred_count = number of TPs (matched detections)
      → Robust to FP spam, counts only correctly detected objects
    - "all_predictions": pred_count = all boxes above conf_threshold
      → May be inflated by false positives, but simpler

    Both methods are computed and returned.

    Args:
        predictions: List of per-image predictions
        ground_truths: List of per-image ground truths
        iou_threshold: Minimum IoU for a valid match (default: 0.5)
        conf_threshold: Confidence threshold for predictions (default: 0.5)
        class_names: Dict mapping class_id to class_name (optional)

    Returns:
        {
            "conf_threshold": 0.5,
            "matched_only": {
                "global_mae": 0.28,  # mean MAE per image
                "per_class_mae": {"Capsicum": 0.15, ...}
            },
            "all_predictions": {
                "global_mae": 0.45,
                "per_class_mae": {"Capsicum": 0.22, ...}
            }
        }

    MAE (Mean Absolute Error) = average of |pred_count - gt_count| per image

    Example:
        >>> results = eval_counting_quality(preds, gts, conf_threshold=0.5)
        >>> print(f"Matched-only MAE: {results['matched_only']['global_mae']:.2f}")
    """
    # Auto-detect class names if not provided
    if class_names is None:
        class_names = {}
        for gt_img in ground_truths:
            for obj in gt_img.get('ground_truth', []):
                class_id = obj['class_id']
                class_name = obj.get('class_name', f'class_{class_id}')
                class_names[class_id] = class_name

    # Match predictions to ground truth
    matching_result = match_predictions_to_ground_truth(
        predictions, ground_truths, iou_threshold, conf_threshold
    )

    # Build lookup: image_id -> ground truth
    gt_by_image = {img['image_id']: img for img in ground_truths}

    # Build lookup: image_id -> predictions
    pred_by_image = {img['image_id']: img for img in predictions}

    # Compute per-image count errors for both methods
    per_image_errors_matched = []
    per_image_errors_all = []

    # Track per-class MAE
    class_errors_matched = defaultdict(list)
    class_errors_all = defaultdict(list)

    for image_id in gt_by_image.keys():
        gt_img = gt_by_image[image_id]
        pred_img = pred_by_image.get(image_id, {'detections': []})

        # Count GT objects per class
        gt_counts = defaultdict(int)
        for obj in gt_img.get('ground_truth', []):
            class_id = obj['class_id']
            gt_counts[class_id] += 1

        # Count predictions per class (all method)
        pred_counts_all = defaultdict(int)
        for det in pred_img.get('detections', []):
            if det.get('confidence', 0) >= conf_threshold:
                class_id = det['class_id']
                pred_counts_all[class_id] += 1

        # Count matched predictions per class (matched_only method)
        pred_counts_matched = defaultdict(int)
        for img_id, pred_det, gt_obj in matching_result['matches']:
            if img_id == image_id:
                class_id = gt_obj['class_id']
                pred_counts_matched[class_id] += 1

        # Compute MAE for this image (sum over all classes)
        all_classes = set(gt_counts.keys()) | set(pred_counts_all.keys()) | set(pred_counts_matched.keys())

        image_mae_matched = 0
        image_mae_all = 0

        for class_id in all_classes:
            gt_count = gt_counts[class_id]
            pred_count_matched = pred_counts_matched[class_id]
            pred_count_all = pred_counts_all[class_id]

            error_matched = abs(pred_count_matched - gt_count)
            error_all = abs(pred_count_all - gt_count)

            image_mae_matched += error_matched
            image_mae_all += error_all

            # Track per-class errors
            class_errors_matched[class_id].append(error_matched)
            class_errors_all[class_id].append(error_all)

        per_image_errors_matched.append(image_mae_matched)
        per_image_errors_all.append(image_mae_all)

    # Compute global MAE (mean over images)
    global_mae_matched = np.mean(per_image_errors_matched) if per_image_errors_matched else 0.0
    global_mae_all = np.mean(per_image_errors_all) if per_image_errors_all else 0.0

    # Compute per-class MAE
    per_class_mae_matched = {}
    per_class_mae_all = {}

    for class_id in class_errors_matched.keys():
        class_name = class_names.get(class_id, f'class_{class_id}')
        per_class_mae_matched[class_name] = round(np.mean(class_errors_matched[class_id]), 4)

    for class_id in class_errors_all.keys():
        class_name = class_names.get(class_id, f'class_{class_id}')
        per_class_mae_all[class_name] = round(np.mean(class_errors_all[class_id]), 4)

    return {
        'conf_threshold': conf_threshold,
        'matched_only': {
            'global_mae': round(global_mae_matched, 4),
            'per_class_mae': per_class_mae_matched
        },
        'all_predictions': {
            'global_mae': round(global_mae_all, 4),
            'per_class_mae': per_class_mae_all
        }
    }
