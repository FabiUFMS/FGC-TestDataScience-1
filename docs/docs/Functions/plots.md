
# PlotData Class


The `plots.py` provides methods to visualize customer churn data. It supports generating:

- Churn rate by tenure with a logarithmic fit for no-churn customers.
- Churn distributions for categorical variables using multiple pie charts and a comparison bar chart.
- Overall churn distribution with pie and bar charts.


---

## Output Files

| Function                       | Output File                        |
| ------------------------------ | ---------------------------------- |
| `plot_rate_tenure_log_fit`     | `churn_rate_by_tenure_log_fit.png` |
| `plot_cat_vs_churn_multi_pies` | `{column}_churn_pie_bar.png`       |
| `plot_churn_distribution`      | `churn_distribution.png`           |
