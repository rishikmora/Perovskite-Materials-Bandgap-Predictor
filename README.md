# Band Gap Prediction Using Machine Learning

## 📌 Project Overview

This project analyzes material data to:
- **Classify** materials as insulators or non-insulators based on PBE band gap.
- **Predict** the exact band gap of identified insulators.

It combines classification using **Random Forest** and regression using **XGBoost**.

---

## 📁 Files

- `02f0e97d-c8f4-4988-8cca-5a44b8922f1f.py`: Main script for data preprocessing, model training, and evaluation.
- `dataset_excavate.xlsx`: Excel dataset containing chemical compositions and band gap values.
- `7d419f2f-ca80-45ed-8df7-db836b5fb481.csv`: Final prediction results (only for insulators).

---

## 🧪 Result Format

The output CSV (`7d419f2f-ca80-45ed-8df7-db836b5fb481.csv`) contains:

| Functional Group | Predicted Insulator | Predicted Band Gap |
|------------------|---------------------|---------------------|
| –OH              | Insulator           | 1.42                |
| –Cl              | Insulator           | 0.87                |
| ...              | ...                 | ...                 |

Only records classified as **insulators** are included, along with their predicted band gaps.

---

## 🛠 How to Run

1. **Install required libraries**:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost openpyxl
