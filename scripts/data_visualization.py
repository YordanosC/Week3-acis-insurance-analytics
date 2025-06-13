import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataVisualizer class with a dataset.
        """
        self.data = data
        sns.set(style="whitegrid")

    def plot_outliers_boxplot(self, cols):
        """
        Plots box plots to detect outliers in numerical columns and saves to file.
        """
        plt.figure(figsize=(12, 4))
        for i, col in enumerate(cols, 1):
            plt.subplot(1, len(cols), i)
            sns.boxplot(y=self.data[col], color='lightblue')
            plt.title(f'Box Plot of {col}')
        plt.tight_layout()
        plt.savefig("Data/visualizations/outliers_boxplot.png", bbox_inches='tight', dpi=300)
        plt.close()

    def plot_correlation_heatmap(self, cols):
        """
        Creates a correlation heatmap for key numerical columns.
        """
        plt.figure(figsize=(10, 8))
        corr_matrix = self.data[cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig("Data/visualizations/correlation_heatmap.png", bbox_inches='tight', dpi=300)
        plt.close()

    def plot_violin_premium_by_cover(self, x_col, y_col):
        """
        Creates a violin plot showing the distribution of TotalPremium by CoverType.
        """
        plt.figure(figsize=(12, 6))
        sns.violinplot(x=x_col, y=y_col, data=self.data, palette='muted', inner='quartile')
        plt.title('Distribution of TotalPremium by CoverType')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("Data/visualizations/premium_by_cover.png", bbox_inches='tight', dpi=300)
        plt.close()

    def plot_geographical_trends(self, cover_types):
        """
        Creates geographical trend plots.
        """
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        filtered_data = self.data[self.data['CoverType'].isin(cover_types)]

        sns.countplot(x='Province', hue='CoverType', data=filtered_data, palette='Set3', ax=axs[0, 0])
        axs[0, 0].set_title('Distribution of Cover Types by Province')
        axs[0, 0].tick_params(axis='x', rotation=45)

        car_make_counts = self.data.groupby('Province')['make'].count().reset_index()
        sns.barplot(x='Province', y='make', data=car_make_counts, ax=axs[0, 1])
        axs[0, 1].set_title('Car Make Distribution by Province')
        axs[0, 1].tick_params(axis='x', rotation=45)

        sns.boxplot(x='Province', y='TotalPremium', data=self.data, ax=axs[1, 0])
        axs[1, 0].set_title('Premium Distribution by Province')
        axs[1, 0].tick_params(axis='x', rotation=45)

        sns.countplot(x='Province', hue='VehicleType', data=self.data, ax=axs[1, 1])
        axs[1, 1].set_title('Vehicle Types by Province')
        axs[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig("Data/visualizations/geographical_trends.png", bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    os.makedirs("Data/visualizations", exist_ok=True)
    df = pd.read_csv("Data/processed/cleaned_data.csv", low_memory=False)
    visualizer = DataVisualizer(df)
    
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    visualizer.plot_outliers_boxplot(numerical_cols)
    visualizer.plot_correlation_heatmap(numerical_cols)
    visualizer.plot_violin_premium_by_cover('CoverType', 'TotalPremium')
    
    common_cover_types = df['CoverType'].value_counts().nlargest(5).index.tolist()
    visualizer.plot_geographical_trends(common_cover_types)