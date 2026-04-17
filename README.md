# Multi-scale Dual-Path Framework for Lung Nodule Classification

[![Kaggle](https://img.shields.io/badge/Data-Kaggle-blue)](https://www.kaggle.com/datasets/zhangweiled/lidcidri)
[![UBC](https://img.shields.io/badge/Affiliation-UBC-gold)](https://bme.ubc.ca/)

This project develops a deep learning system for lung nodule classification. It features a custom V4 Dual-Path Architecture and a Hierarchical Sampling strategy to improve diagnostic accuracy on the LIDC-IDRI dataset.

## 📖 Key Features
- **Evolutionary Design**: Transitioned from baseline CNNs to a multi-scale V4 architecture to capture both local textures and global anatomical context.
- **Smart Data Refinement**: Optimized a 15,548-slice dataset from 70,000+ raw images by prioritizing high-consensus (3-4 votes) and hard-negative (2 votes) samples.
- **Clinical Validation**: Achieved **79.95% accuracy** using strict patient-wise partitioning to ensure real-world generalizability.

## 📁 Project Structure
- **Code/**: Contains the full model evolution (V1-V5), data preprocessing scripts, and evaluation tools.
- **Plots/**: Visualizations of training performance, loss curves, and model comparison metrics.

## 📊 Dataset
This project uses the [Kaggle LIDC-IDRI 2D Slices](https://www.kaggle.com/datasets/zhangweiled/lidcidri).   
*Note: Due to size constraints, the image data is not included in this repository. Please download it from Kaggle and update the `DATA_ROOT` path in `Code/data_utils.py` before running.*

## 🛠️ Quick Start
1. **Configure Paths**: Set your local dataset path in `Code/data_utils.py`.
2. **Train**: Run `python Code/train_v4.py`.
3. **Evaluate**: Run `python Code/evaluate_v4_tta.py` for final performance metrics.

## 📈 Results
The final **V4 Dual-Path model** significantly reduces false positives by leveraging its global-view path, especially in cases where vascular bifurcations mimic nodule textures.

---
**Author**: Xiaoxi Gao 
