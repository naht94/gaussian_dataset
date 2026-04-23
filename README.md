# [Research] Preprocessing Strategies for 3D Gaussian Splatting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19694064.svg)](https://doi.org/10.5281/zenodo.19694064)

This repository contains the official implementation and benchmarking scripts for the paper:
> **"A Study on Preprocessing Strategies for RGB-Based 3DGS"**
> Hyeong-Taek Kwon, Moonsoo Kang and SeongKi Kim
> *The Visual Computer*, Springer Nature, 2026. (Revised Submission)

---

## 📌 Project Overview
This study systematically analyzes how different feature extraction and matching strategies impact the reconstruction quality of 3D Gaussian Splatting (3DGS). We provide a standardized pipeline to reproduce the 12 preprocessing combinations (from SIFT to LightGlue) evaluated in our survey.

## 📌 Paper Information

- **Journal**: The Visual Computer

- **Manuscript ID**: 40b24de7-0c46-4abd-95f6-277bae217144

- **Status**: Revised version

- **Corresponding Author**: SeongKi Kim (skkim@chosun.ac.kr)

## 🛠 Author's Contribution for Reproducibility
While this framework is built upon the **hloc (Hierarchical Localization)** toolbox, we have developed specific components to ensure research transparency:
- **`run_all_experiments.py`**: A dedicated automation script that executes all 12 experimental matrices to generate 3DGS-ready sparse point clouds.
- **`datasets/survey_data/`**: Includes the multi-view image dataset (28 frames) used in our comparative analysis.

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository with submodules
git clone --recursive https://github.com/naht94/gaussian_splatting_preprocessing_dataset.git
cd gaussian_splatting_preprocessing_dataset

# Install dependencies
pip install -r requirements.txt
python setup.py install
```

2. Running Experiments
To reproduce the SfM initialization results for all 12 combinations:
```bash
python run_all_experiments.py
```

Outputs (sparse PLY files and COLMAP databases) will be saved in the outputs/survey_benchmark/ directory.

📄 License
This project is released underthe MIT License. 

🔁 Reproducibility

All experiments reported in the paper can be reproduced using the provided scripts. Random seeds and configuration files are included to ensure reproducibility.

📜 Citation

If you use this code or our survey data in your research,please cite: 

```bibtex
@article{Kwon2026VisualComputer,
  title = {A Study on Preprocessing Strategies for RGB-Based 3DGS},
  author = {Hyeong-Taek Kwon and Moonsoo Kang and SeongKi Kim},
  journal = {The Visual Computer},
  year = {2026},
  publisher = {Springer Nature},
  note = {Revised Submission}
}
```

Note: This project is based on the hloc toolbox. We provide our custom pipeline on top of it.
