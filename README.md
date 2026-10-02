# Travel Mode Choice in Victoria

A data science project investigating how household demographics, individual characteristics, and journey distance are associated with travel mode for work and education journeys in Victoria.

## Overview

This project integrates four Victorian Integrated Survey of Travel and Activity (VISTA) 2023–2024 datasets and compares two classification models:

- K-Nearest Neighbours (KNN)
- Decision Tree

The target variable contains three travel mode groups: **Private**, **Public**, and **Active**.

The workflow covers data-quality investigation, preprocessing, feature engineering, normalised mutual information (NMI), hyperparameter tuning, and model evaluation.

## Data

The analysis requires four CSV files:

- `household_vista_2023_2024.csv`
- `person_vista_2023_2024.csv`
- `journey_to_work_vista_2023_2024.csv`
- `journey_to_education_vista_2023_2024.csv`

Source: [Victorian Integrated Survey of Travel and Activity](https://discover.data.vic.gov.au/dataset/victorian-integrated-survey-of-travel-and-activity-vista)

Place these files in `data/raw/` before running the analysis. Raw datasets are excluded from the repository.

## Repository Structure

| Location | Description |
|---|---|
| `src/main.ipynb` | Main notebook for preprocessing, feature analysis, model training, and evaluation |
| `src/preprocessing_helpers.py` | Functions for grouping travel modes and constructing features |
| `src/ml_helpers.py` | Cross-validation and classification evaluation functions |
| `scripts/data_investigation.py` | Exploratory checks of distributions, missing values, and unusual observations |
| `scripts/income_diagnostics.py` | Investigation of missing household income using person-level records |
| `data/raw/` | Original VISTA CSV files, stored locally |
| `data/processed/` | Generated processed dataset |
| `reports/` | Project report and selected figures |

## Methodology

### Data Preprocessing

Household, person, and journey datasets are linked using household and person identifiers (`hhid` and `persid`).

Eight predictors are constructed:

- Household income
- Household size
- Bicycle ownership
- Household vehicle access
- Residential region
- Age group
- Employment status
- Journey distance group

The feature named `car_ownership` combines household vehicle availability with the presence of at least one member holding a full or green probationary licence, following the original project definition.

Travel modes are grouped into Private, Public, and Active. Unrecognised modes are excluded.

### Feature Analysis

Normalised mutual information measures the association between each predictor and travel mode. A reduced-feature experiment excludes household size and bicycle ownership.

### Model Training

KNN and decision tree models are evaluated under three configurations:

1. Random 80/20 training and test split
2. Stratified 80/20 training and test split
3. Stratified split with a reduced feature set

Five-fold cross-validation is used to tune the number of neighbours for KNN and the maximum depth for decision trees.

### Evaluation

Model performance is assessed using:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Confusion matrices


## Getting Started

### Install Dependencies

The project uses Python, pandas, NumPy, Matplotlib, seaborn, and scikit-learn.

Install the libraries and Jupyter:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

1. Download or clone the repository.
2. Place the four VISTA CSV files in `data/raw/`.
3. Open `src/main.ipynb` in Jupyter or VS Code.
4. Select a Python kernel with the required libraries installed.
5. Restart the kernel and run all cells in order.

Run the notebook from the repository root or `src/` so its path setup can locate the data.

The notebook saves the processed dataset to `data/processed/preprocessed.csv` and displays NMI results, tuning plots, confusion matrices, and evaluation metrics.

### Run Optional Diagnostics

From the repository root:

```bash
python scripts/data_investigation.py
python scripts/income_diagnostics.py
```

## Limitations

- Private travel dominates the dataset, so accuracy and weighted metrics can conceal poor minority-class performance.
- Encoding and NMI analysis are performed before splitting in the retained workflow.
- Different test splits are used across experiments, limiting direct comparisons of sampling and feature-selection strategies.
- Journeys from the same person or household may appear in both training and test sets.
- The original missing-income treatment and distance-bin definitions remain in the preprocessing helpers.
- The predictors do not capture all behavioural, environmental, and transport-access factors associated with travel mode.

Future improvements include training-only preprocessing, group-aware evaluation, a majority-class baseline, and macro and per-class performance metrics.

## Contributors and Code Updates

Original analysis and code:
- Ha Phuong Nguyen
- Yiwen Zhang
- Chi Ian Chan

Code adapted and updated by **Ha Phuong Nguyen**
The original preprocessing and modelling methodology is otherwise retained.
