"""
Visualization functions for detection evaluation results.
"""

from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend for server environments
matplotlib.use('Agg')


def plot_threshold_sweep(
    threshold_sweep_results: Dict[str, Dict],
    output_path: str,
    title: str = "Precision/Recall/F1 vs Confidence Threshold"
):
    """
    Plot P/R/F1 curves vs confidence threshold.

    Args:
        threshold_sweep_results: Dict from eval_detection_prf_at_iou()
        output_path: Path to save figure
        title: Plot title

    Example:
        >>> results = eval_detection_prf_at_iou(preds, gts)
        >>> plot_threshold_sweep(results, "figures/threshold_sweep.png")
    """
    thresholds = sorted([float(k) for k in threshold_sweep_results.keys()])
    precisions = [threshold_sweep_results[str(t)]['precision'] for t in thresholds]
    recalls = [threshold_sweep_results[str(t)]['recall'] for t in thresholds]
    f1s = [threshold_sweep_results[str(t)]['f1'] for t in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, marker='o', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, marker='s', label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, marker='^', label='F1 Score', linewidth=2, linestyle='--')

    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([min(thresholds), max(thresholds)])
    plt.ylim([0, 1.05])

    # Add best F1 annotation
    best_f1_idx = np.argmax(f1s)
    best_thr = thresholds[best_f1_idx]
    best_f1 = f1s[best_f1_idx]
    plt.axvline(best_thr, color='red', linestyle=':', alpha=0.5, label=f'Best F1 @ {best_thr}')
    plt.annotate(
        f'Best F1={best_f1:.3f}\n@ conf={best_thr}',
        xy=(best_thr, best_f1),
        xytext=(best_thr + 0.1, best_f1 - 0.15),
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved threshold sweep plot: {output_path}")


def plot_per_class_f1(
    per_class_results: Dict[str, Dict],
    output_path: str,
    title: str = "Per-Class F1 Score",
    top_k: Optional[int] = None
):
    """
    Plot horizontal bar chart of per-class F1 scores.

    Args:
        per_class_results: Dict from eval_per_class_metrics_and_confusions()['per_class']
        output_path: Path to save figure
        title: Plot title
        top_k: Show only top-K classes (optional, default: all classes)

    Example:
        >>> results = eval_per_class_metrics_and_confusions(preds, gts)
        >>> plot_per_class_f1(results['per_class'], "figures/per_class_f1.png")
    """
    # Extract class names and F1 scores
    classes = []
    f1_scores = []
    supports = []

    for class_name, metrics in per_class_results.items():
        classes.append(class_name)
        f1_scores.append(metrics['f1'])
        supports.append(metrics['support'])

    # Sort by F1 score descending
    sorted_indices = np.argsort(f1_scores)[::-1]
    classes = [classes[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    supports = [supports[i] for i in sorted_indices]

    # Limit to top-K if specified
    if top_k is not None and top_k < len(classes):
        classes = classes[:top_k]
        f1_scores = f1_scores[:top_k]
        supports = supports[:top_k]

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(classes) * 0.3)))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
    bars = ax.barh(classes, f1_scores, color=colors, edgecolor='black', linewidth=0.5)

    # Add F1 values at end of bars
    for i, (f1, support) in enumerate(zip(f1_scores, supports)):
        ax.text(f1 + 0.02, i, f'{f1:.3f} (n={support})', va='center', fontsize=9)

    ax.set_xlabel('F1 Score (Percentage)', fontsize=12)
    ax.set_ylabel('Class (Number of Instances)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.1])
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved per-class F1 plot: {output_path}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_order: list,
    output_path: str,
    title: str = "Confusion Matrix",
    figsize: tuple = (12, 10),
    cmap: str = 'Blues'
):
    """
    Plot confusion matrix as a heatmap.

    Args:
        confusion_matrix: 2D array (rows=true, cols=pred)
        class_order: List of class names in order
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size (width, height)
        cmap: Colormap name

    Example:
        >>> results = eval_per_class_metrics_and_confusions(preds, gts)
        >>> plot_confusion_matrix(
        ...     results['confusion_matrix'],
        ...     results['class_order'],
        ...     "figures/confusion_matrix.png"
        ... )
    """
    cm = np.array(confusion_matrix)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(cm, cmap=cmap, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20, fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(class_order)))
    ax.set_yticks(np.arange(len(class_order)))
    ax.set_xticklabels(class_order, rotation=90, fontsize=9)
    ax.set_yticklabels(class_order, fontsize=9)

    # Labels
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add text annotations (only for non-zero cells to avoid clutter)
    max_val = cm.max()
    for i in range(len(class_order)):
        for j in range(len(class_order)):
            count = cm[i, j]
            if count > 0:
                # Use white text for dark cells, black for light cells
                text_color = 'white' if count > max_val / 2 else 'black'
                ax.text(j, i, str(count), ha='center', va='center',
                       color=text_color, fontsize=8)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved confusion matrix plot: {output_path}")


def plot_count_mae_comparison(
    counting_results: Dict,
    output_path: str,
    title: str = "Count MAE: Matched-Only vs All-Predictions"
):
    """
    Plot comparison of count MAE between two counting methods.

    Args:
        counting_results: Dict from eval_counting_quality()
        output_path: Path to save figure
        title: Plot title

    Example:
        >>> results = eval_counting_quality(preds, gts)
        >>> plot_count_mae_comparison(results, "figures/count_mae.png")
    """
    matched_mae = counting_results['matched_only']['global_mae']
    all_mae = counting_results['all_predictions']['global_mae']

    methods = ['Matched-Only\n(TP counts)', 'All-Predictions\n(all boxes)']
    maes = [matched_mae, all_mae]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(methods, maes, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
               f'{mae:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(maes) * 1.2])
    ax.grid(axis='y', alpha=0.3)

    # Add explanation text
    explanation = (
        "Matched-Only: Counts only correctly detected objects (TPs)\n"
        "All-Predictions: Counts all predicted boxes above threshold"
    )
    ax.text(0.5, 0.95, explanation, transform=ax.transAxes,
           fontsize=9, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved count MAE comparison plot: {output_path}")


def plot_all_metrics(
    threshold_sweep: Dict,
    per_class_results: Dict,
    confusion_data: Dict,
    counting_results: Dict,
    output_dir: str,
    run_name: str = "evaluation"
):
    """
    Generate all evaluation plots in one call.

    Args:
        threshold_sweep: Results from eval_detection_prf_at_iou()
        per_class_results: Results from eval_per_class_metrics_and_confusions()
        confusion_data: Dict with 'confusion_matrix' and 'class_order' keys
        counting_results: Results from eval_counting_quality()
        output_dir: Directory to save all plots
        run_name: Name to include in titles

    Example:
        >>> from evaluation.metrics import *
        >>> prf = eval_detection_prf_at_iou(preds, gts)
        >>> per_class = eval_per_class_metrics_and_confusions(preds, gts)
        >>> counting = eval_counting_quality(preds, gts)
        >>> plot_all_metrics(prf, per_class['per_class'], per_class, counting, "results/")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Threshold sweep
    plot_threshold_sweep(
        threshold_sweep,
        output_dir / "threshold_sweep.png",
        title=f"{run_name}: P/R/F1 vs Confidence"
    )

    # 2. Per-class F1
    plot_per_class_f1(
        per_class_results,
        output_dir / "per_class_f1.png",
        title=f"{run_name}: Per-Class F1 Score"
    )

    # 3. Confusion matrix
    plot_confusion_matrix(
        np.array(confusion_data['confusion_matrix']),
        confusion_data['class_order'],
        output_dir / "confusion_matrix.png",
        title=f"{run_name}: Confusion Matrix"
    )

    # 4. Count MAE comparison
    plot_count_mae_comparison(
        counting_results,
        output_dir / "count_mae_comparison.png",
        title=f"{run_name}: Count MAE Comparison"
    )

    print(f"\n✓ All plots saved to: {output_dir}/")
