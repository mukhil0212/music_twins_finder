import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EDAVisualizer:
    def __init__(self):
        self.style_config = {
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        }
        plt.rcParams.update(self.style_config)
        
    def create_feature_distributions(self, data: pd.DataFrame,
                                   feature_columns: List[str],
                                   save_path: Optional[str] = None):
        """Create comprehensive distribution plots for all features."""
        n_features = len(feature_columns)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(feature_columns):
            ax = axes[idx]
            
            # Create histogram with KDE
            data[feature].hist(ax=ax, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            data[feature].plot(kind='density', ax=ax, color='red', linewidth=2)
            
            # Add statistics
            mean_val = data[feature].mean()
            median_val = data[feature].median()
            ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            
            ax.set_title(f'Distribution of {feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Feature Distributions', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature distributions to {save_path}")
        
        plt.close()
    
    def create_box_plots_by_cluster(self, data: pd.DataFrame,
                                  feature_columns: List[str],
                                  cluster_column: str = 'cluster',
                                  save_path: Optional[str] = None):
        """Create box plots for features grouped by cluster."""
        n_features = len(feature_columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(feature_columns):
            ax = axes[idx]
            
            # Create box plot
            data.boxplot(column=feature, by=cluster_column, ax=ax)
            ax.set_title(f'{feature} by Cluster')
            ax.set_xlabel('Cluster')
            ax.set_ylabel(feature)
            ax.grid(True, alpha=0.3)
            
            # Remove automatic title
            ax.get_figure().suptitle('')
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Feature Comparison Across Clusters', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved box plots to {save_path}")
        
        plt.close()
    
    def create_violin_plots(self, data: pd.DataFrame,
                          feature_columns: List[str],
                          cluster_column: str = 'cluster',
                          save_path: Optional[str] = None):
        """Create violin plots for detailed distribution comparison."""
        n_features = min(len(feature_columns), 9)  # Limit to 9 for readability
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(feature_columns[:n_features]):
            ax = axes[idx]
            
            # Create violin plot
            sns.violinplot(data=data, x=cluster_column, y=feature, ax=ax, palette='viridis')
            
            # Add mean points
            means = data.groupby(cluster_column)[feature].mean()
            for cluster, mean_val in means.items():
                ax.scatter(cluster, mean_val, color='red', s=100, zorder=10, marker='D')
            
            ax.set_title(f'{feature} Distribution by Cluster')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Detailed Feature Distributions (Violin Plots)', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved violin plots to {save_path}")
        
        plt.close()
    
    def create_correlation_analysis(self, data: pd.DataFrame,
                                  feature_columns: List[str],
                                  save_path: Optional[str] = None):
        """Create correlation analysis visualizations."""
        # Select numeric features
        numeric_data = data[feature_columns]
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1])
        
        # 1. Correlation heatmap
        ax1 = fig.add_subplot(gs[0])
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, ax=ax1,
                   cbar_kws={"shrink": 0.8})
        ax1.set_title('Feature Correlation Matrix', fontsize=14)
        
        # 2. Top correlations bar plot
        ax2 = fig.add_subplot(gs[1])
        
        # Extract upper triangle correlations
        upper_triangle = corr_matrix.where(mask == False)
        correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if not np.isnan(upper_triangle.iloc[i, j]):
                    correlations.append({
                        'feature_pair': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                        'correlation': upper_triangle.iloc[i, j]
                    })
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        top_corrs = correlations[:15]  # Top 15 correlations
        
        # Plot
        pairs = [c['feature_pair'] for c in top_corrs]
        values = [c['correlation'] for c in top_corrs]
        colors = ['red' if v < 0 else 'green' for v in values]
        
        y_pos = np.arange(len(pairs))
        ax2.barh(y_pos, values, color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(pairs)
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_title('Top 15 Feature Correlations', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Correlation strength distribution
        ax3 = fig.add_subplot(gs[2])
        all_corrs = [c['correlation'] for c in correlations]
        ax3.hist(all_corrs, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_xlabel('Correlation Coefficient')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Correlation Distribution', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation analysis to {save_path}")
        
        plt.close()
    
    def create_scatter_plot_matrix(self, data: pd.DataFrame,
                                 feature_columns: List[str],
                                 cluster_column: Optional[str] = None,
                                 save_path: Optional[str] = None):
        """Create scatter plot matrix for top features."""
        # Limit to top 6 features for readability
        n_features = min(len(feature_columns), 6)
        selected_features = feature_columns[:n_features]
        
        if cluster_column and cluster_column in data.columns:
            # Create scatter plot matrix with cluster coloring
            fig = px.scatter_matrix(
                data,
                dimensions=selected_features,
                color=cluster_column,
                title=f"Scatter Plot Matrix - Top {n_features} Features",
                height=900,
                width=900
            )
        else:
            # Create without clustering
            fig = px.scatter_matrix(
                data,
                dimensions=selected_features,
                title=f"Scatter Plot Matrix - Top {n_features} Features",
                height=900,
                width=900
            )
        
        fig.update_traces(diagonal_visible=False, marker=dict(size=5))
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved scatter plot matrix to {save_path}")
        
        return fig
    
    def create_statistical_summary(self, data: pd.DataFrame,
                                 feature_columns: List[str],
                                 cluster_column: Optional[str] = None,
                                 save_path: Optional[str] = None):
        """Create comprehensive statistical summary visualizations."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Feature statistics table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('tight')
        ax1.axis('off')
        
        # Calculate statistics
        stats_df = data[feature_columns].describe().T
        stats_df['skewness'] = data[feature_columns].skew()
        stats_df['kurtosis'] = data[feature_columns].kurtosis()
        
        # Create table
        table_data = stats_df.round(3).values
        col_labels = stats_df.columns.tolist()
        row_labels = stats_df.index.tolist()
        
        table = ax1.table(cellText=table_data,
                         colLabels=col_labels,
                         rowLabels=row_labels,
                         cellLoc='center',
                         loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        ax1.set_title('Feature Statistics Summary', fontsize=14, pad=20)
        
        # 2. Missing values heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        missing_data = data[feature_columns].isnull().sum()
        if missing_data.sum() > 0:
            missing_data.plot(kind='barh', ax=ax2, color='red')
            ax2.set_xlabel('Number of Missing Values')
            ax2.set_title('Missing Values by Feature')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Missing Values Check')
        ax2.grid(True, alpha=0.3)
        
        # 3. Skewness visualization
        ax3 = fig.add_subplot(gs[1, 1])
        skewness = data[feature_columns].skew().sort_values()
        colors = ['red' if abs(s) > 1 else 'yellow' if abs(s) > 0.5 else 'green' for s in skewness]
        skewness.plot(kind='barh', ax=ax3, color=colors)
        ax3.set_xlabel('Skewness')
        ax3.set_title('Feature Skewness')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # 4. Outlier detection
        ax4 = fig.add_subplot(gs[1, 2])
        outlier_counts = {}
        for feature in feature_columns[:10]:  # Top 10 features
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[feature] < (Q1 - 1.5 * IQR)) | 
                       (data[feature] > (Q3 + 1.5 * IQR))).sum()
            outlier_counts[feature] = outliers
        
        pd.Series(outlier_counts).plot(kind='bar', ax=ax4, color='orange')
        ax4.set_ylabel('Number of Outliers')
        ax4.set_title('Outlier Count by Feature (IQR Method)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Normality tests
        ax5 = fig.add_subplot(gs[2, :2])
        normality_results = []
        for feature in feature_columns[:15]:  # Top 15 features
            stat, p_value = stats.normaltest(data[feature].dropna())
            normality_results.append({
                'feature': feature,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            })
        
        norm_df = pd.DataFrame(normality_results)
        colors = ['green' if n else 'red' for n in norm_df['is_normal']]
        
        ax5.bar(range(len(norm_df)), -np.log10(norm_df['p_value']), color=colors, alpha=0.7)
        ax5.axhline(y=-np.log10(0.05), color='black', linestyle='--', 
                   label='p=0.05 threshold')
        ax5.set_xticks(range(len(norm_df)))
        ax5.set_xticklabels(norm_df['feature'], rotation=45, ha='right')
        ax5.set_ylabel('-log10(p-value)')
        ax5.set_title('Normality Test Results (D\'Agostino-Pearson)')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Variance analysis
        ax6 = fig.add_subplot(gs[2, 2])
        variances = data[feature_columns].var().sort_values(ascending=False)[:10]
        variances.plot(kind='barh', ax=ax6, color='purple')
        ax6.set_xlabel('Variance')
        ax6.set_title('Top 10 Features by Variance')
        ax6.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Statistical Analysis Summary', fontsize=18)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved statistical summary to {save_path}")
        
        plt.close()
    
    def create_outlier_detection_plots(self, data: pd.DataFrame,
                                     feature_columns: List[str],
                                     save_path: Optional[str] = None):
        """Create outlier detection visualizations."""
        n_features = min(len(feature_columns), 9)
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        for idx, feature in enumerate(feature_columns[:n_features]):
            ax = axes[idx]
            
            # Calculate outliers using IQR method
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
            normal_data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]
            
            # Create box plot with outliers highlighted
            box_data = [normal_data[feature].dropna(), outliers[feature].dropna()]
            bp = ax.boxplot(box_data, labels=['Normal', 'Outliers'], patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            # Add outlier percentage
            outlier_pct = len(outliers) / len(data) * 100
            ax.text(0.02, 0.98, f'Outliers: {outlier_pct:.1f}%', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(f'{feature} - Outlier Analysis')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Outlier Detection Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved outlier detection plots to {save_path}")
        
        plt.close()
    
    def create_time_series_analysis(self, temporal_data: Dict[str, Dict],
                                  save_path: Optional[str] = None):
        """Create time-based analysis visualizations."""
        # Aggregate temporal patterns
        all_hour_patterns = []
        all_day_patterns = []
        
        for user_data in temporal_data.values():
            if 'hour_distribution' in user_data:
                all_hour_patterns.append(user_data['hour_distribution'])
            if 'day_distribution' in user_data:
                all_day_patterns.append(user_data['day_distribution'])
        
        if not all_hour_patterns:
            logger.warning("No temporal data available")
            return
        
        # Convert to arrays
        hour_matrix = np.array(all_hour_patterns)
        day_matrix = np.array(all_day_patterns)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Average listening pattern by hour
        ax1 = axes[0, 0]
        avg_hour_pattern = np.mean(hour_matrix, axis=0)
        std_hour_pattern = np.std(hour_matrix, axis=0)
        
        hours = range(24)
        ax1.plot(hours, avg_hour_pattern, 'b-', linewidth=2, label='Average')
        ax1.fill_between(hours, 
                        avg_hour_pattern - std_hour_pattern,
                        avg_hour_pattern + std_hour_pattern,
                        alpha=0.3)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Listening Activity')
        ax1.set_title('Daily Listening Pattern')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Peak hour distribution
        ax2 = axes[0, 1]
        peak_hours = np.argmax(hour_matrix, axis=1)
        ax2.hist(peak_hours, bins=24, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Number of Users')
        ax2.set_title('Distribution of Peak Listening Hours')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Day of week patterns
        ax3 = axes[1, 0]
        avg_day_pattern = np.mean(day_matrix, axis=0)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        bars = ax3.bar(days, avg_day_pattern, color='orange', alpha=0.7)
        
        # Highlight weekend
        bars[5].set_color('red')
        bars[6].set_color('red')
        
        ax3.set_ylabel('Average Listening Activity')
        ax3.set_title('Weekly Listening Pattern')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Weekend vs Weekday comparison
        ax4 = axes[1, 1]
        weekday_avg = np.mean(day_matrix[:, :5], axis=1)
        weekend_avg = np.mean(day_matrix[:, 5:], axis=1)
        
        ax4.scatter(weekday_avg, weekend_avg, alpha=0.6)
        ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Equal line
        ax4.set_xlabel('Weekday Average')
        ax4.set_ylabel('Weekend Average')
        ax4.set_title('Weekend vs Weekday Listening')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Temporal Listening Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved time series analysis to {save_path}")
        
        plt.close()
    
    def create_feature_importance_analysis(self, feature_importance: pd.DataFrame,
                                         save_path: Optional[str] = None):
        """Create feature importance visualizations."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Bar plot of feature importance
        top_features = feature_importance.nlargest(20, 'importance')
        ax1.barh(top_features['feature'], top_features['importance'], color='teal')
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Top 20 Most Important Features')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Cumulative importance
        sorted_importance = feature_importance.sort_values('importance', ascending=False)
        cumulative_importance = sorted_importance['importance'].cumsum()
        cumulative_importance = cumulative_importance / cumulative_importance.iloc[-1]
        
        ax2.plot(range(len(cumulative_importance)), cumulative_importance, 'b-', linewidth=2)
        ax2.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        ax2.axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
        
        # Find number of features for 80% and 90% importance
        n_80 = (cumulative_importance >= 0.8).argmax()
        n_90 = (cumulative_importance >= 0.9).argmax()
        
        ax2.axvline(x=n_80, color='r', linestyle=':', alpha=0.5)
        ax2.axvline(x=n_90, color='orange', linestyle=':', alpha=0.5)
        
        ax2.text(n_80, 0.5, f'{n_80} features\nfor 80%', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.text(n_90, 0.7, f'{n_90} features\nfor 90%', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance')
        ax2.set_title('Cumulative Feature Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance analysis to {save_path}")
        
        plt.close()