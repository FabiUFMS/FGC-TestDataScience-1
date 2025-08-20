# CleanDataset Class

## Overview

The `dataset.py`  is responsible for cleaning and preprocessing customer churn datasets.  


---

## Class: `CleanDataset`

Cleans and preprocesses a dataset loaded from a CSV or provided as a pandas DataFrame.  
The process prepares the data for exploratory analysis and machine learning models.

---

## Cleaning Process Flowchart

                                ┌───────────────────────┐
                                │ Load Raw DataFrame  │
                                └──────────┬────────────┘
                                          │
                                          ▼
                                ┌───────────────────────┐
                                │ Lowercase Columns   │
                                │ & String Values     │
                                └──────────┬────────────┘
                                          │
                                          ▼
                                ┌───────────────────────┐
                                │ Drop 'customerid'   │
                                └──────────┬────────────┘
                                          │
                                          ▼
                            ┌───────────────────────────────┐
                            │ Convert 'totalcharges'     │
                            │ → numeric (fill 0 if NaN & │
                            │ tenure == 0)               │
                            └──────────────┬────────────────┘
                                          │
                                          ▼
                                ┌───────────────────────┐
                                │ Map 'seniorcitizen' │
                                │ 0 → no, 1 → yes     │
                                └──────────┬────────────┘
                                          │
                                          ▼
                             ┌────────────────────────────┐
                             │ Simplify 'paymentmethod'│
                             │ & 'contract' values     │
                             └─────────────┬──────────────┘
                                          │
                                          ▼
                                ┌───────────────────────┐
                                │ Save Clean Dataset  │
                                │ churn_clean_data.csv│
                                └───────────────────────┘


---

## Output File

Location: `PROCESSED_DATA_DIR/churn_clean_data.csv`  

---
## Next Improvements
Perhaps I should consolidate the functions of dataset.py and features.py, considering that they both practically fulfill the same objective, which is to process the dataset.