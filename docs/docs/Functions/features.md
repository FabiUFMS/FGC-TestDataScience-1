#  FeatureEng Class


The `features.py` performs feature engineering on a customer dataset, preparing it for machine learning models.  
It supports label encoding, one-hot encoding, and mapping of specific categorical features to numerical values.

---

##  Class: `FeatureEng`

Encodes categorical features using label encoding for binary/ordinal variables and one-hot encoding for multi-class variables, and other features.

---


## 🔄 Feature Engineering Process Flowchart

                                                  ┌───────────────┐
                                                  │Input DataFrame│
                                                  └───────┬───────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │ Map 'contract' to │
                                                │ numeric values    │
                                                └─────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │ Label Encode      │
                                                │ label_columns     │
                                                └─────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │ One-Hot Encode    │
                                                │ one_hot_columns   │
                                                └─────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │Concatenate encoded│
                                                │ columns & drop    │
                                                │ originals         │
                                                └─────────┬─────────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │ Output Engineered │
                                                │ DataFrame         │
                                                └───────────────────┘
