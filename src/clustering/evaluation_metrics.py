import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import pairwise_distances
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusteringEvaluator:
    def __init__(self):
        self.metrics_history = []
        
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate clustering using multiple metrics."""
        n_clusters = len(np.unique(labels))
        
        if n_clusters < 2:
            logger.warning("Only one cluster found. Cannot calculate most metrics.")
            return {
                'n_clusters': n_clusters,
                'silhouette_score': -1,
                'davies_bouldin_index': np.inf,
                'calinski_harabasz_index': 0,
                'within_cluster_sum_of_squares': self._calculate_wcss(X, labels)
            }
        
        metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette_score(X, labels)),
            'davies_bouldin_index': float(davies_bouldin_score(X, labels)),
            'calinski_harabasz_index': float(calinski_harabasz_score(X, labels)),
            'within_cluster_sum_of_squares': float(self._calculate_wcss(X, labels))
        }
        
        # Additional custom metrics
        metrics.update(self._calculate_custom_metrics(X, labels))
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_wcss(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares."""
        wcss = 0
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                wcss += np.sum((cluster_points - centroid) ** 2)
        
        return wcss
    
    def _calculate_custom_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate additional custom clustering metrics."""
        custom_metrics = {}
        
        # Cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))
        
        # Size balance (how evenly distributed are the clusters)
        size_std = np.std(list(cluster_sizes.values()))
        size_mean = np.mean(list(cluster_sizes.values()))
        custom_metrics['size_balance'] = 1 - (size_std / (size_mean + 1e-10))
        
        # Compactness (average within-cluster distance)
        compactness_scores = []
        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                distances = pairwise_distances(cluster_points)
                compactness = np.mean(distances[np.triu_indices_from(distances, k=1)])
                compactness_scores.append(compactness)
        
        custom_metrics['avg_compactness'] = float(np.mean(compactness_scores)) if compactness_scores else 0
        
        # Separation (average between-cluster distance)
        if len(unique_labels) > 1:
            centroids = []
            for label in unique_labels:
                cluster_points = X[labels == label]
                centroids.append(np.mean(cluster_points, axis=0))
            
            centroids = np.array(centroids)
            centroid_distances = pairwise_distances(centroids)
            separation = np.mean(centroid_distances[np.triu_indices_from(centroid_distances, k=1)])
            custom_metrics['avg_separation'] = float(separation)
        else:
            custom_metrics['avg_separation'] = 0
        
        # Dunn Index (ratio of minimum inter-cluster to maximum intra-cluster distance)
        if len(unique_labels) > 1 and compactness_scores:
            custom_metrics['dunn_index'] = custom_metrics['avg_separation'] / (max(compactness_scores) + 1e-10)
        else:
            custom_metrics['dunn_index'] = 0
        
        return custom_metrics
    
    def compare_clusterings(self, X: np.ndarray, 
                          clusterings: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Compare multiple clustering results."""
        import pandas as pd
        
        comparison_results = []
        
        for name, labels in clusterings.items():
            metrics = self.evaluate_clustering(X, labels)
            metrics['method'] = name
            comparison_results.append(metrics)
        
        return pd.DataFrame(comparison_results)
    
    def plot_metrics_comparison(self, comparison_df: pd.DataFrame, 
                              save_path: Optional[str] = None):
        """Plot comparison of clustering metrics."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Select metrics to plot
        metrics_to_plot = [
            'silhouette_score', 
            'davies_bouldin_index',
            'calinski_harabasz_index',
            'size_balance',
            'dunn_index'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            if metric in comparison_df.columns:
                ax = axes[idx]
                
                # Normalize metric for better visualization
                values = comparison_df[metric].values
                if metric == 'davies_bouldin_index':
                    # Lower is better for Davies-Bouldin
                    values = 1 / (values + 1)
                    ylabel = f"1 / ({metric} + 1)"
                else:
                    ylabel = metric.replace('_', ' ').title()
                
                # Create bar plot
                sns.barplot(data=comparison_df, x='method', y=metric, ax=ax)
                ax.set_xlabel('Clustering Method')
                ax.set_ylabel(ylabel)
                ax.set_title(f'{ylabel}')
                ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        if len(metrics_to_plot) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics comparison plot to {save_path}")
        
        plt.close()
    
    def stability_analysis(self, X: np.ndarray, clustering_func, 
                          n_iterations: int = 10, 
                          subsample_ratio: float = 0.8) -> Dict:
        """Analyze clustering stability through subsampling."""
        stability_results = {
            'iteration_metrics': [],
            'label_consistency': []
        }
        
        n_samples = X.shape[0]
        n_subsample = int(n_samples * subsample_ratio)
        
        # Store labels from each iteration
        all_labels = []
        
        for i in range(n_iterations):
            # Subsample data
            indices = np.random.choice(n_samples, n_subsample, replace=False)
            X_subsample = X[indices]
            
            # Perform clustering
            labels = clustering_func(X_subsample)
            all_labels.append((indices, labels))
            
            # Evaluate
            metrics = self.evaluate_clustering(X_subsample, labels)
            stability_results['iteration_metrics'].append(metrics)
        
        # Calculate stability metrics
        metric_names = stability_results['iteration_metrics'][0].keys()
        for metric in metric_names:
            values = [m[metric] for m in stability_results['iteration_metrics']]
            stability_results[f'{metric}_mean'] = float(np.mean(values))
            stability_results[f'{metric}_std'] = float(np.std(values))
        
        # Calculate label consistency (for overlapping samples)
        consistency_scores = []
        for i in range(n_iterations):
            for j in range(i + 1, n_iterations):
                indices_i, labels_i = all_labels[i]
                indices_j, labels_j = all_labels[j]
                
                # Find overlapping samples
                overlap = np.intersect1d(indices_i, indices_j)
                if len(overlap) > 1:
                    # Get labels for overlapping samples
                    labels_i_overlap = labels_i[np.isin(indices_i, overlap)]
                    labels_j_overlap = labels_j[np.isin(indices_j, overlap)]
                    
                    # Calculate adjusted rand index
                    from sklearn.metrics import adjusted_rand_score
                    consistency = adjusted_rand_score(labels_i_overlap, labels_j_overlap)
                    consistency_scores.append(consistency)
        
        stability_results['label_consistency_mean'] = float(np.mean(consistency_scores)) if consistency_scores else 0
        stability_results['label_consistency_std'] = float(np.std(consistency_scores)) if consistency_scores else 0
        
        return stability_results
    
    def cluster_quality_report(self, X: np.ndarray, labels: np.ndarray, 
                             feature_names: List[str]) -> Dict:
        """Generate comprehensive cluster quality report."""
        report = {
            'overall_metrics': self.evaluate_clustering(X, labels),
            'cluster_details': {},
            'recommendations': []
        }
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Analyze each cluster
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_size = np.sum(cluster_mask)
            
            report['cluster_details'][f'cluster_{label}'] = {
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(labels) * 100)
            }
            
            if cluster_size > 1:
                cluster_data = X[cluster_mask]
                
                # Within-cluster cohesion
                distances = pairwise_distances(cluster_data)
                cohesion = np.mean(distances[np.triu_indices_from(distances, k=1)])
                report['cluster_details'][f'cluster_{label}']['cohesion'] = float(cohesion)
                
                # Silhouette score for this cluster
                cluster_silhouette = np.mean(silhouette_score(X, labels, metric='euclidean') * cluster_mask)
                report['cluster_details'][f'cluster_{label}']['silhouette'] = float(cluster_silhouette)
        
        # Generate recommendations
        if report['overall_metrics']['silhouette_score'] < 0.25:
            report['recommendations'].append("Low silhouette score suggests poor cluster separation. Consider different number of clusters.")
        
        if report['overall_metrics']['size_balance'] < 0.5:
            report['recommendations'].append("Clusters are highly imbalanced. Consider different preprocessing or clustering approach.")
        
        if report['overall_metrics']['davies_bouldin_index'] > 2:
            report['recommendations'].append("High Davies-Bouldin index indicates clusters may be overlapping.")
        
        # Check for too small clusters
        min_cluster_size = min(cd['size'] for cd in report['cluster_details'].values())
        if min_cluster_size < len(labels) * 0.05:
            report['recommendations'].append(f"Some clusters are very small (<5% of data). These might be outliers.")
        
        return report