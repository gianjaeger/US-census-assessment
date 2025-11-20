# US Census Income Analysis

## Problem statement

The assessment provides a sample dataset from the US Census archive containing detailed, but anonymized, information for ~300,000 individuals. Specifically, it contains information on 43 features/variables that I group into the categories: "Demographic, labour market and household", "Financial" and "Geographic". See attached PDF write-up for a full beak-down.

The objective is to identify "characteristics that are associated with a person making more or less than $50,000 per year".

## Repository structure

```text
.
├── Analysis_1.ipynb              # Data cleaning and statistical analysis
├── Analysis_2.ipynb              # Machine learning analysis
│
├── Data/
│   ├── Raw/
│   ├── Cleaned/
│   ├── census_income_metadata.txt
│   └── census_income_additional_info.pdf
│
├── interpretability/
│   ├── visualize_categorical.py
│   ├── visualize_pdp.py
│   └── visualize_tree.py
│
├── models/
│   ├── logistic_regression.py
│   └── random_forest.py
│
├── requirements.txt
└── README.md
```

## Analysis overview

The project is structured into two notebooks:

**Analysis_1.ipynb (Data cleaning and statistical analysis)**
Covers data cleaning, exploratory data analysis, and hypothesis testing. Includes basic descriptive statistics, visual exploration, and bivariate tests for both numerical and categorical variables.

**Analysis_2.ipynb (Machine learning analysis)**
Builds predictive models to capture non-linear relationships. Implements a Random Forest and a Logistic Regression model, with appropriate preprocessing, hyperparameter tuning, and an interpretability analysis.

## Data availability and reproducibility

The original data was provided as two CSV files (`census_income_learn.csv` and `census_income_test.csv`). Since these exceed GitHub's 100 MB per-fule limit, they were converted to Feather files.

The Feather files are stored in the **Raw data** folder. Cleaned, preprocessed and labeled versions, as well as the full dataset with all responses can be found in the **Cleaned data** folder. These datasets are used from Notebook 1, Section 2 (Exploratory Data Analysis) onwards.

## Notes

- The analysis uses relative paths, so the notebooks have to be ran from the repository root directory
- The `instance_weight` column is excluded from modeling as per metadata recommendations
