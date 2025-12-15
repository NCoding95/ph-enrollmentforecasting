# Regional Enrollment Forecasting and Volatility Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)

> **A Machine Learning Approach for Adaptive Educational Resource Planning in the Philippines**

## 📋 Overview

This repository contains the complete implementation of a dual-method framework for:
1. **Forecasting** regional K-12 enrollment using machine learning regression models
2. **Classifying** regional enrollment volatility to enable differentiated resource planning

The research analyzes Philippine Department of Education (DepEd) enrollment data from 2010-2021 across all 17 regions to support evidence-based educational resource allocation.

## 📖 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@inproceedings{bulio2025enrollment,
  title={Regional Enrollment Forecasting and Volatility Classification: A Machine Learning Approach for Adaptive Educational Resource Planning in the Philippines},
  author={Bulio, Nice O. and Cardaño, Mc Sergel},
  booktitle={Proceedings of the 3rd Kalimuhagan Multidisciplinary Research Congress},
  year={2025},
  organization={South Philippine Adventist College},
  address={Camanchiles, Matanao, Davao del Sur, Philippines}
}
```

**APA Format:**
> Bulio, N. O., & Cardaño, M. S. (2025). Regional enrollment forecasting and volatility classification: A machine learning approach for adaptive educational resource planning in the Philippines. In *Proceedings of the 3rd Kalimuhagan Multidisciplinary Research Congress*. South Philippine Adventist College.

## 🎯 Research Questions

| RQ | Focus | Description |
|----|-------|-------------|
| **RQ1** | Prediction | Which ML model best predicts regional K-12 enrollment? |
| **RQ2** | Classification | Which regions are stable vs. volatile in enrollment patterns? |
| **RQ3** | Application | How can predictions improve resource distribution? |

## 🗂️ Repository Structure

```
enrollment-forecasting/
│
├── 📁 data/
│   └── philippines_enrollment.csv      # Dataset (see Data Source below)
│
├── 📁 src/
│   └── mini.py                         # Main analysis pipeline
│
├── 📁 outputs/
│   ├── 01_enrollment_distribution.png  # Distribution visualization
│   ├── 02_model_performance.png        # Model comparison chart
│   ├── 03_predicted_vs_actual.png      # Prediction scatter plot
│   ├── 04_feature_importance.png       # Random Forest features
│   ├── 05_regional_volatility.png      # Volatility classification
│   └── regional_volatility_classification.csv  # Results data
│
├── 📁 docs/
│   └── KALIMUHAGAN_CARDAÑO_BULIO_2025.pptx               # Conference presentation
│
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
└── README.md                           # This file
```

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/enrollment-forecasting.git
cd enrollment-forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0
openpyxl>=3.0.0
```

## 🚀 Quick Start

### Running the Analysis

```bash
# Navigate to source directory
cd src

# Run the complete pipeline
python mini.ipyynb
```

### Expected Output

The script will:
1. Load and preprocess the enrollment dataset
2. Train four ML models (Linear Regression, Decision Tree, Random Forest, SVR)
3. Evaluate models using R², MAE, and RMSE
4. Classify regions as "Stable" or "Volatile" using CV analysis
5. Generate visualizations and save results

## 📊 Methodology

### Data Pipeline

```
INPUT                    PROCESS                      OUTPUT
─────────────────────────────────────────────────────────────────
Historical Data    →    ML Models:                →  Enrollment
(2010-2021)             • Linear Regression           Forecasts
17 Regions              • Decision Tree
Public/Private          • Random Forest           →  Volatility
K-12 Grade Levels       • SVR                         Classes
                                                      (Stable/Volatile)
561 observations   →    CV Analysis:              →  Planning
57 features             75th %ile threshold           Recommendations
```

### Train-Test Split (Walk-Forward Validation)

| Set | Years | Samples | Purpose |
|-----|-------|---------|---------|
| **Training** | 2010-2019 | 448 (80%) | Model learning |
| **Testing** | 2020-2021 | 113 (20%) | Validation (pandemic robustness) |

### Volatility Classification

Regions are classified using **Coefficient of Variation (CV)**:

```
CV = (Standard Deviation / Mean) × 100
```

- **Threshold**: 75th percentile of all regional CV values
- **Volatile**: CV > threshold (top 25% most variable)
- **Stable**: CV ≤ threshold (bottom 75%)

## 📈 Key Results

### Model Performance

### Model performance (hold-out test)
| Model                    | R² (Train) | R² (Test) | MAE (students) | RMSE (students) | MAPE (%) |
|-------------------------:|:----------:|:---------:|---------------:|----------------:|---------:|
| Random Forest            | 0.9937     | **0.9874**| **46,312.61**  | **78,753.36**   | 39.28    |
| Decision Tree            | 0.9930     | 0.9867    | 46,444.54      | 80,905.23       | 28.38    |
| Linear Regression        | 0.8456     | 0.8362    | 196,320.79     | 284,212.39      | 2038.43  |
| Support Vector Regression| -0.3018    | -0.3254   | 486,184.01     | 808,461.41      | 2401.08  |

**Best model:** **Random Forest** — R² = **0.9874** (98.74% variance explained).  
**Practical error (Random Forest):** MAE ≈ **±46,313 students** (≈**8.9%** of the average enrollment = 520,536).

### Volatility Classification

| Classification | Count | Regions |
|----------------|-------|---------|
| **Volatile** | 4 | NCR, CALABARZON, Central Luzon, Central Visayas |
| **Stable** | 13 | All other regions |

**Volatility Ratio**: Volatile regions exhibit **1.53×** higher enrollment variability than stable regions.

## 📁 Data Source

**Dataset**: Philippines School Enrollment Data  
**Source**: Kaggle (Raiblaze, 2023)  
**URL**: https://www.kaggle.com/datasets/raiblaze/philippines-school-enrollment-data  
**Validation**: Cross-referenced with DepEd official records (±0.2% accuracy)

### Dataset Specifications

| Attribute | Details |
|-----------|---------|
| Time Period | 2010-2021 (11 academic years) |
| Geographic Coverage | All 17 Philippine regions |
| Sectors | Public and Private |
| Grade Levels | Kindergarten through Grade 12 + SHS strands |
| Total Observations | 561 records |
| Features | 57 (after one-hot encoding) |

## 🔬 Replication Guide

To replicate this study:

1. **Download the dataset** from the Kaggle link above
2. **Place the CSV file** in the `data/` directory as `philippines_enrollment.csv`
3. **Run the analysis pipeline**: `python src/mini.py`
4. **View outputs** in the `outputs/` directory

### Customization

Modify these parameters in `mini.py` to experiment:

```python
# Model hyperparameters
DecisionTreeRegressor(max_depth=10, min_samples_split=5)
RandomForestRegressor(n_estimators=100, max_depth=10)
SVR(kernel='rbf', C=1000, epsilon=0.1)

# Volatility threshold (default: 75th percentile)
threshold_75 = regional_stats['CV'].quantile(0.75)
```

## ⚠️ Limitations

1. **Temporal Overfitting**: The R² = 1.00 for Linear Regression suggests the model learns year-specific patterns due to one-hot encoded year features. For forecasting beyond the dataset years, consider removing year features or using time-series models.

2. **Data Scope**: Analysis limited to 2010-2021; post-pandemic enrollment patterns may differ.

3. **Aggregation Level**: Regional-level analysis; school-level or district-level forecasting would require additional data.

4. **External Factors**: Model does not incorporate external variables (economic indicators, migration data, policy changes) that may affect enrollment.

## 🔮 Future Work

- [ ] Incorporate external socioeconomic indicators
- [ ] Develop school-level forecasting models
- [ ] Create early warning indicators for enrollment surges
- [ ] Build real-time forecasting dashboard for DepEd
- [ ] Extend methodology to other ASEAN countries

## 👥 Authors

**Nice O. Bulio**  
South Philippine Adventist College  
📧 ncbulio95@gmail.com

**Mc Sergel Cardaño**  
South Philippine Adventist College  
📧 kentgats@gmail.com

## 🙏 Acknowledgments

- **South Philippine Adventist College** - Institutional support
- **3rd Kalimuhagan Multidisciplinary Research Congress** - Presentation venue
- **Raiblaze (Kaggle)** - Dataset compilation
- **Department of Education (DepEd)** - Original enrollment data

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Nice O. Bulio & Mc Sergel Cardaño

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📬 Contact

For questions, collaborations, or feedback:

- **Issues**: Open an issue on this repository
- **Email**: [ncbulio95@gmail.com]
- **Institution**: South Philippine Adventist College, Camanchiles, Matanao, Davao del Sur, Philippines

---

<p align="center">
  <i>Advancing educational planning through data-driven research</i>
  <br><br>
  <b>🎓 Enlighten. Empower. Engage. 🎓</b>
</p>
