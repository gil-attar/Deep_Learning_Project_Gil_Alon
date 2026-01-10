# Experimental Protocol
This document defines the experimental settings conducted in our project.
All experiments, metrics, datasets, and evaluation procedures described below are fixed
before results are analyzed.

## Overview and Motivation
The general goal of the experiments is to perform a controlled and reproducible comparison between
a CNN-based detector (YOLOv8) and a Transformer-based detector (RT-DETR), and to evaluate
their behavior under increasing levels of visual occlusion, which we found to be the most related
augmentation to true usage reality of our models.

The experiments defined here are designed to:
- Establish fair baseline performance under identical conditions
- Measure robustness degradation as occlusion severity increases
- Empirically justify confidence-based filtering (Logic Gate)
- Validate downstream recipe generation robustness

All experiments are executed under shared constraints.

## Shared Experimental Constraints
The following constraints apply to **all experiments (E1–E4)**:

- **Test Set**:  
  All evaluations use the same frozen test set defined in  
  `data/processed/evaluation/test_index.json`

- **Occlusion Difficulty Definition**:  
  Difficulty levels (Easy / Medium / Hard) are precomputed and fixed in  
  `data/processed/evaluation/difficulty_summary.csv`

- **Image Resolution**:  
  Identical image resolution is used for YOLOv8 and RT-DETR during evaluation

- **Confidence Threshold (Baseline)**:  
  A fixed confidence threshold is used for baseline evaluation (no per-model tuning)

- **Hardware / Environment**:  
  All inference runs are performed on the same execution environment (Google Colab GPU)

- **No Post-hoc Changes**:  
  No experiment definitions, metrics, or dataset splits are modified after results are generated

## Experiments:
## E1: Baseline YOLOv8 vs RT-DETR Training on raw Dataset
* Inputs:
    - Test images listed in `test_index.json`
    - Ground-truth annotations from the processed dataset
    - No difficulty-based filtering applied

* Models:
    - YOLOv8 (CNN-based detector)
    - RT-DETR (Transformer-based detector)

* Metrics:
    - mAP@50
    - Precision
    - Recall
    - FPS (averaged over the full test set)

* Output artifacts:
    Stored under:
        - `evaluation/metrics/baseline_yolo_*.json`
        - `evaluation/metrics/baseline_rtdetr_*.json`

## E2: YOLOv8 vs RT-DETR Training on Dataset with occlusions, Evaluation on Difficulty Level (Easy/ Medium / Hard)
* Inputs:
    - Test images grouped by difficulty level (Easy / Medium / Hard)
    - Difficulty assignment from `difficulty_summary.csv`
    - Same test index as Experiment E1

* Models:
    - YOLOv8
    - RT-DETR  
    (models trained identically to Experiment E1)

* Metrics
    - Recall (primary metric)
    - Precision
    - False Negative (FN) count

* Output artifacts
    Stored under:
        - `evaluation/metrics/yolo_by_difficulty.json`
        - `evaluation/metrics/rtdetr_by_difficulty.json`
        - Difficulty-based comparison plots under `evaluation/plots/

## E3: Confidence ThreshHold Sweep (Logic Gate Analysis)
* Inputs:
    - Full test set (`test_index.json`)
    - Detection outputs across a sweep of confidence thresholds

* Model:
    - YOLOv8
    - RT-DETR

* Metrics:
For each confidence threshold:
    - Precision
    - Recall
    - False positives per image

* Output artifacts:
    Stored under:
        - `evaluation/plots/confidence_precision_curve.png`
        - `evaluation/plots/confidence_recall_curve.png`

## E4: Recipe Hallucination Robustness Comparison (Raw vs Filtered) 

* Input:
    - Detection outputs from Experiment E3
    - Two ingredient sets per image:
        - Raw detections
        - Logic-Gate-filtered detections

* Models:
    - Same detection models as previous experiments
    - Recipe generation pipeline (LLM-based), unchanged between conditions

* Metrics:
    - Binary or ordinal cookability score
    - Presence of hallucinated ingredients

* Output artifacts
    Stored under:
        - `recipes/raw_outputs.json`
        - `recipes/filtered_outputs.json`
        - `recipes/evaluation_scores.csv`
