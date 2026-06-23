# High-Dimensional Tabular Regression Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Feature_Synthesis-orange)
![CatBoost & XGBoost](https://img.shields.io/badge/CatBoost_%7C_XGBoost-Weighted_Ensemble-red)
![R² Score](https://img.shields.io/badge/Cross--Val_R²-0.5791-brightgreen)

An end-to-end Machine Learning pipeline engineered to predict testing times for custom vehicle configurations on a sparse, anonymized 378-feature manufacturing dataset. 

## 📊 Performance Benchmark

| Model Iteration | Validation Strategy | $R^2$ Score |
| :--- | :--- | :--- |
| Baseline (Simple Linear) | Standard Train/Test | `0.4012` |
| Tuned XGBoost | 5-Fold CV | `0.5582` |
| Tuned CatBoost | 5-Fold CV | `0.5645` |
| **Final Weighted Blend** | **5-Fold Out-of-Fold** | **`0.5791`** |

## 📂 Project Architecture

* `mercedes_manufacturing_regression.ipynb`: The master execution pipeline. Handles parallel dimensionality reduction, custom leakage-free encoding, and the weighted ensembling.
* `archive/`
  * `v1_linear_baseline.ipynb`: Historical sandbox demonstrating the initial $R^2 = 0.40$ baseline and raw feature distributions.
  * `v2_bayesian_opt.ipynb`: Experimental hyperparameter search space tests utilizing `BayesianOptimization`.

## 🧠 Core Pipeline Architecture

1. **Dual Dimensionality Reduction (`PCA` + `FastICA`)** The dataset contains 300+ highly sparse binary flags. Principal Component Analysis and Independent Component Analysis are executed in parallel to collapse the sparse space into 10 dense, high-variance synthetic features.
2. **Leakage-Free Mean Target Encoding** High-cardinality categorical variables (such as `X0`) are transformed using a custom 5-Fold Out-of-Fold (OOF) mean encoder, strictly preventing target leakage into the validation folds.
3. **Static Ensemble Blending** Balances the aggressive, depth-wise split finding of `XGBoost` against the balanced, oblivious trees of `CatBoost` via a static weighted regressor:
   $$\hat{y} = (0.6 \times \text{CatBoost}) + (0.4 \times \text{XGBoost})$$

## 🚀 Quick Start

#### 1. Environment Setup
```bash
git clone https://github.com/pineapple-bits/mercedes-regression-pipeline.git
cd mercedes-regression-pipeline
pip install -r requirements.txt
```
#### 2. Dataset Setup
Due to Kaggle’s redistribution rules, the raw dataset cannot be hosted inside this repository. You can obtain the files via one of two ways:

Download `train.csv` and `test.csv` from the [Kaggle Competition Page](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/data) *(Requires Kaggle login)*.


Once downloaded, place both files inside a `data/` folder at the root...
