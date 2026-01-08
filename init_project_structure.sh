#!/usr/bin/env bash
set -e

echo "Initializing project directory structure..."

# Data
mkdir -p data/raw
mkdir -p data/processed

# Experiments
mkdir -p experiments/yolo
mkdir -p experiments/rtdetr

# Evaluation
mkdir -p evaluation/metrics
mkdir -p evaluation/plots

# Logic gate and recipes
mkdir -p logic_gate
mkdir -p recipes

# Notebooks and scripts
mkdir -p notebooks
mkdir -p scripts

echo "Project structure created successfully."
