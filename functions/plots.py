
### Imports ###
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
from config import PROCESSED_DATA_DIR, FIGURES_DIR
matplotlib.use('Agg')



class PlotData:
    def __init__(self, 
                 input_path: Path, 
                 output_path: Path):
        
        self.input_path = input_path
        self.output_path = output_path


    def plot_rate_tenure_log_fit(self):
        """
        Plots the churn and no-churn rates by customer tenure with a logarithmic fit for the no-churn rate.
        This method reads customer data from the input CSV file specified by `self.input_path`, calculates the churn and no-churn rates for each tenure value, and fits a logarithmic curve to the no-churn rate. The resulting plot displays stacked bar charts for churn and no-churn rates by tenure, along with the fitted logarithmic curve for the no-churn rate. The plot is saved as a PNG file to the path specified by `self.output_path`.
        The logarithmic fit uses the function: y = a + b * ln(x), where x is tenure.
        Saves:
            churn_rate_by_tenure_log_fit.png: The generated plot showing churn rates and the logarithmic fit.
        """
    
        data = pd.read_csv(self.input_path)
        # Convert 'churn' to binary values
        data['churn_binary'] = data['churn'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
        
        # Group by 'tenure' and calculate churn rates
        churn_by_tenure = data.groupby('tenure')['churn_binary'].agg(['count', 'sum']).reset_index()
        
        # Calculate churn and no-churn rates
        churn_by_tenure['churn_rate'] = (churn_by_tenure['sum'] / churn_by_tenure['count']) * 100
        churn_by_tenure['no_churn_rate'] = 100 - churn_by_tenure['churn_rate']
        
        # Logarithmic fit function
        def log_func(x, a, b):
            return a + b * np.log(x)
        
        # Fit the logarithmic function to the no-churn rate
        filtered_data = churn_by_tenure[churn_by_tenure['tenure'] > 0].copy()
        params, _ = curve_fit(log_func, filtered_data['tenure'], filtered_data['no_churn_rate'])
        x_values = np.linspace(1, filtered_data['tenure'].max(), 100)
        y_values = log_func(x_values, *params)
        

        # Create the stacked bar chart
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.bar(churn_by_tenure['tenure'], 
               churn_by_tenure['no_churn_rate'], 
                label='No Churn', 
                color='#4E79A7', 
                alpha=0.8)
        
        ax.bar(churn_by_tenure['tenure'], 
               churn_by_tenure['churn_rate'], 
                bottom=churn_by_tenure['no_churn_rate'], 
                label='Churn', 
                color='#F28E2B', 
                alpha=0.8)
        
        # Plot the logarithmic fit line
        ax.plot(x_values, 
                y_values, 
                'g-', 
                label=f'Log Fit (No Churn): y = {params[0]:.2f} + {params[1]:.2f}*ln(x)', 
                linewidth=2)
        ax.set_xlabel('Tenure (months)')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Churn Rate by Tenure (Monthly) with Logarithmic Fit (No Churn)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        output_file = self.output_path / "churn_rate_by_tenure_log_fit.png"
        fig.savefig(output_file, bbox_inches='tight')
        
        plt.close(fig)


    def plot_cat_vs_churn_multi_pies(self, column: str):
        """
        Plots multiple pie charts showing the churn distribution for each category of a given categorical variable,
        and a bar chart comparing churn counts across categories, centered next to the pie charts.

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame containing the data, including the 'churn' column and the categorical variable of interest.
        column : str
            The name of the categorical column to analyze.

        Returns
        -------
        None. Displays the charts directly.
        """
        
        data = pd.read_csv(self.input_path)
        
        colors_pie = ['#4E79A7', '#F28E2B']
        colors_bar = ['#59A14F', '#E15759']

        # Crosstabulate the data
        crosstab = pd.crosstab(data[column], 
                            data['churn'])
        n_cats = len(crosstab)
        n_cols = 2
        n_rows = (n_cats + n_cols - 1) // n_cols

        # Create the figure and axes
        fig, axes = plt.subplots(n_rows, 
                                 n_cols + 1, 
                                 figsize=(6*(n_cols+1), 5*n_rows))
        axes = np.array(axes).reshape(n_rows, 
                                      n_cols + 1)

        # --- Pie Charts --- #
        for idx, cat in enumerate(crosstab.index):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            ax.pie(
                crosstab.loc[cat],
                labels=crosstab.columns,
                autopct='%1.1f%%',
                colors=colors_pie,
                startangle=90,
                wedgeprops={'edgecolor': 'white'}
            )
            ax.set_title(f'{column.title()} = {cat}\nChurn Distribution')

        # Remove axes not used for pie charts
        for idx in range(n_cats, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        # --- Bar Chart --- #
        # Centralize the bar chart in the last column of the middle row
        bar_row = n_rows // 2 if n_rows > 1 else 0
        bar_ax = axes[bar_row, -1]

        crosstab.plot(
            kind='bar',
            stacked=False,
            color=colors_bar,
            edgecolor='black',
            ax=bar_ax
        )

        # Set the title and labels for the bar chart
        bar_ax.set_title(f'{column.title()} vs Churn')
        bar_ax.set_xlabel(column.title())
        bar_ax.set_ylabel('Number of Customers')

        # Add legend to the bar chart
        bar_ax.legend(
            title='Churn', 
            loc='upper left', 
            bbox_to_anchor=(1.05, 1), 
            borderaxespad=0.
            )

        # Bar chart annotations
        for p in bar_ax.patches:
            height = p.get_height()
            if height > 0:
                bar_ax.annotate(f'{int(height)}',
                                (p.get_x() + p.get_width() / 2, height),
                                ha='center', 
                                va='bottom', 
                                fontsize=10, 
                                color='black')

        # Remove axes not used for the bar chart
        for r in range(n_rows):
            if r != bar_row:
                axes[r, -1].axis('off')

        # Adjust layout 
        plt.subplots_adjust(wspace=0.3, 
                            hspace=0.4, 
                            right=0.85)

        # Save the figure
        fig.savefig(self.output_path / f"{column}_churn_pie_bar.png", bbox_inches='tight')
        
        plt.close(fig)


    def plot_churn_distribution(self):
            """
            Generates and saves a figure showing the distribution of the 'churn' variable in the dataset.
            This method reads the input CSV file specified by `self.input_path`, computes the count of each class in the 'churn' column,
            and creates a side-by-side pie chart and bar chart visualizing the churn distribution. The resulting figure is saved as
            'churn_distribution.png' in the directory specified by `self.output_path`.
            The pie chart displays the percentage of each churn class, while the bar chart shows the absolute number of customers for each class,
            with value annotations.
            Raises:
                FileNotFoundError: If the input CSV file does not exist.
                KeyError: If the 'churn' column is not present in the dataset.
                Exception: For other errors during file reading or plotting.
            """
            # Read the data
            data = pd.read_csv(self.input_path)

            # Count the occurrences of each churn class
            count_churn = data['churn'].value_counts().reset_index()
            count_churn.columns = ['Churn', 'Count']

            # Colors for the pie and bar charts
            colors_pie = ['#4E79A7', '#F28E2B']
            colors_bar = ['#59A14F', '#E15759']

            # Define the style for the plots
            sns.set_style("whitegrid")
            plt.rcParams.update({
                'font.size': 12, 
                'axes.titlesize': 14, 
                'axes.labelsize': 13
                })

            # Figure setup
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # --- Pie Chart --- #
            axes[0].pie(
                count_churn['Count'],
                labels=count_churn['Churn'],
                autopct='%1.1f%%',
                colors=colors_pie,
                startangle=90,
                wedgeprops={'edgecolor': 'white'}
            )
            axes[0].set_title('Churn Distribution')

            # --- Bar Chart --- #
            sns.barplot(
                x='Churn',
                y='Count',
                data=count_churn,
                palette=colors_bar,
                ax=axes[1]
            )

            # Add value annotations to the bar chart
            for p in axes[1].patches:
                height = p.get_height()
                axes[1].annotate(
                    f'{int(height)}',
                    (p.get_x() + p.get_width() / 2, height),
                    ha='center', 
                    va='bottom', 
                    fontsize=12, 
                    color='black'
                )

            axes[1].set_ylabel('Number of Customers')
            axes[1].set_xlabel('Churn')
            axes[1].set_title('Number of Customers by Churn')

            plt.tight_layout()

            # Save the figure
            output_file = self.output_path / "churn_distribution.png"
            fig.savefig(output_file, bbox_inches='tight')
            plt.close(fig)


def main(
    input_path: Path = PROCESSED_DATA_DIR / "churn_clean_data.csv",
    output_path: Path = FIGURES_DIR 
):
    plot_data = PlotData(input_path, output_path)
    
    cat_cols = ['gender', 'seniorcitizen', 'partner', 'dependents',
                'phoneservice', 'multiplelines', 'internetservice',
                'onlinesecurity', 'onlinebackup', 'deviceprotection',
                'techsupport', 'streamingtv', 'streamingmovies', 'contract',
                'paperlessbilling', 'paymentmethod']
    
    for col in cat_cols:
         plot_data.plot_cat_vs_churn_multi_pies(col)
        
    plot_data.plot_churn_distribution()
    plot_data.plot_rate_tenure_log_fit()
    
    

if __name__ == "__main__":
    main()
