## FGC-TestDataScience-1

## Telco Customer Churn Prediction

This project is a **Data Science classification case** for *CLASIFICACIÓN* challenge.
The goal is to **predict customer churn** (whether a customer will leave the company) based on their demographic characteristics and*contracted services.

The dataset used is the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/).

---

## ⚙️ Installation & Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/FabiUFMS/FGC-TestDataScience-1
   ```
2. **Create the virtual environment:**
	```powershell
	python -m venv 0_venv_class
	```

3. **Activate the virtual environment:**
	```powershell
	.\0_venv_class\Scripts\Activate
	```

	After activation, you will see the environment name at the beginning of the command line.

4. **Install the dependencies:**
	```powershell
	pip install -r requirements.txt
	```

5. **Run the main script**

   ```bash
   python main.py
   ```

6. **Check results**
   The trained model and evaluation metrics will be stored in the `models/` folder.
<br>

7. **When finished, deactivate the environment:**
	```powershell
	deactivate
	```


---


## 📂 Project Structure

```
├── data
│   ├── processed    # Processed/cleaned data
│   └── raw          # Original raw data
│
├── functions        # Isolated functions used across the project
│
├── models           # Model metrics and trained model
│
├── notebooks        # Ordered notebooks: EDA, feature engineering, model testing
│   ├── data         # Data outputs generated from notebooks
│   └── html         # Notebooks exported as HTML
│
└── reports
    └── figures      # Visualizations and plots
```

---

## 📊 Project Workflow

1. **Exploratory Data Analysis (EDA)** – Understanding churn patterns.
2. **Feature Engineering** – Data cleaning, encoding, and feature selection.
3. **Model Training & Evaluation** – Testing different machine learning models.
4. **Reporting** – Storing results, metrics, and visualizations.

---
