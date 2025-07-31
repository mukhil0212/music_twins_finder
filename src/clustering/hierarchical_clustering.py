import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

from config.spotify_config import SpotifyConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HierarchicalClustering:
    def __init__(self, n_clusters: Optional[int] = None, 
                 linkage_method: str = 'ward',
                 distance_metric: str = 'euclidean'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.model = None
        self.labels = None
        self.linkage_matrix = None
        
    def fit(self, X: np.ndarray, n_clusters: Optional[int] = None) -> 'HierarchicalClustering':
        """Fit hierarchical clustering model."""
        if n_clusters:
            self.n_clusters = n_clusters
        elif self.n_clusters is None:
            # Find optimal number of clusters
            self.n_clusters = self._find_optimal_clusters(X)
            logger.info(f"Using optimal n_clusters={self.n_clusters}")
        
        logger.info(f"Fitting Hierarchical clustering with {self.n_clusters} clusters...")
        
        # Compute linkage matrix
        self.linkage_matrix = linkage(X, method=self.linkage_method, metric=self.distance_metric)
        
        # Fit model
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage_method,
            affinity=self.distance_metric
        )
        
        self.labels = self.model.fit_predict(X)
        
        # Evaluate clustering directly
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        if len(np.unique(self.labels)) > 1:
            metrics = {
                'silhouette_score': silhouette_score(X, self.labels),
                'davies_bouldin_index': davies_bouldin_score(X, self.labels),
                'calinski_harabasz_index': calinski_harabasz_score(X, self.labels)
            }
        else:
            metrics = {
                'silhouette_score': 0.0,
                'davies_bouldin_index': 0.0,
                'calinski_harabasz_index': 0.0
            }

        logger.info(f"Clustering metrics: {metrics}")
        
        return self
    
    def _find_optimal_clusters(self, X: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette analysis."""
        k_range = range(SpotifyConfig.MIN_CLUSTERS, min(SpotifyConfig.MAX_CLUSTERS, len(X) // 2))
        best_score = -1
        best_k = SpotifyConfig.MIN_CLUSTERS
        
        for k in k_range:
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage=self.linkage_method,
                affinity=self.distance_metric
            )
            labels = model.fit_predict(X)
            
            from sklearn.metrics import silhouette_score
            score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # For hierarchical clustering, we need to find nearest cluster
        # This is a simplified approach - for production, use more sophisticated methods
        from sklearn.neighbors import NearestNeighbors
        
        # Get cluster centers
        cluster_centers = []
        for i in range(self.n_clusters):
            cluster_mask = self.labels == i
            if np.any(cluster_mask):
                center = np.mean(X[cluster_mask], axis=0)
                cluster_centers.append(center)
        
        cluster_centers = np.array(cluster_centers)
        
        # Find nearest cluster center for each point
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(cluster_centers)
        _, indices = nn.kneighbors(X)
        
        return indices.flatten()
    
    def fit_predict(self, X: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """Fit model and predict cluster labels."""
        self.fit(X, n_clusters)
        return self.labels
    
    def get_cluster_hierarchy(self, X: np.ndarray, user_ids: List[str]) -> Dict:
        """Get hierarchical structure of clusters."""
        if self.linkage_matrix is None:
            raise ValueError("Model must be fitted first")
        
        hierarchy = {
            'linkage_matrix': self.linkage_matrix.tolist(),
            'cluster_assignments': {},
            'merge_distances': []
        }
        
        # Get cluster assignments at different cut heights
        max_d = np.max(self.linkage_matrix[:, 2])
        for d in np.linspace(0, max_d, 10):
            clusters = fcluster(self.linkage_matrix, d, criterion='distance')
            n_clusters = len(np.unique(clusters))
            hierarchy['cluster_assignments'][f'distance_{d:.2f}'] = {
                'n_clusters': n_clusters,
                'assignments': clusters.tolist()
            }
        
        # Get merge distances
        hierarchy['merge_distances'] = self.linkage_matrix[:, 2].tolist()
        
        return hierarchy
    
    def plot_dendrogram(self, user_ids: Optional[List[str]] = None, 
                       save_path: Optional[str] = None,
                       max_display: int = 50):
        """Plot dendrogram of hierarchical clustering."""
        if self.linkage_matrix is None:
            raise ValueError("Model must be fitted first")
        
        plt.figure(figsize=(15, 8))
        
        # Create dendrogram
        if user_ids and len(user_ids) <= max_display:
            labels = user_ids
        else:
            labels = None
        
        dendrogram_data = dendrogram(
            self.linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=10
        )
        
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage_method} linkage)')
        plt.xlabel('User Index' if labels is None else 'User ID')
        plt.ylabel('Distance')
        
        # Add horizontal line at cut height
        if self.n_clusters:
            # Calculate cut height for n_clusters
            cut_height = self._get_cut_height(self.n_clusters)
            plt.axhline(y=cut_height, color='r', linestyle='--', 
                       label=f'Cut for {self.n_clusters} clusters')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved dendrogram to {save_path}")
        
        plt.close()
    
    def _get_cut_height(self, n_clusters: int) -> float:
        """Calculate the cut height for a given number of clusters."""
        if n_clusters >= len(self.linkage_matrix) + 1:
            return 0
        
        # The cut height is just above the (n-n_clusters)th merge
        return self.linkage_matrix[-(n_clusters-1), 2] + 0.01
    
    def compare_linkage_methods(self, X: np.ndarray) -> Dict:
        """Compare different linkage methods."""
        methods = ['ward', 'complete', 'average', 'single']
        results = {}
        
        for method in methods:
            logger.info(f"Testing {method} linkage...")
            
            # Skip ward if not using euclidean distance
            if method == 'ward' and self.distance_metric != 'euclidean':
                continue
            
            linkage_matrix = linkage(X, method=method, metric=self.distance_metric)
            
            # Test different numbers of clusters
            method_results = {
                'n_clusters': [],
                'silhouette_scores': [],
                'davies_bouldin_scores': []
            }
            
            for k in range(SpotifyConfig.MIN_CLUSTERS, 
                          min(SpotifyConfig.MAX_CLUSTERS, len(X) // 2)):
                labels = fcluster(linkage_matrix, k, criterion='maxclust')
                # Calculate metrics directly
                from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
                labels_indexed = labels - 1  # Convert to 0-indexed

                if len(np.unique(labels_indexed)) > 1:
                    metrics = {
                        'silhouette_score': silhouette_score(X, labels_indexed),
                        'davies_bouldin_index': davies_bouldin_score(X, labels_indexed),
                        'calinski_harabasz_index': calinski_harabasz_score(X, labels_indexed)
                    }
                else:
                    metrics = {
                        'silhouette_score': 0.0,
                        'davies_bouldin_index': 0.0,
                        'calinski_harabasz_index': 0.0
                    }
                
                method_results['n_clusters'].append(k)
                method_results['silhouette_scores'].append(metrics['silhouette_score'])
                method_results['davies_bouldin_scores'].append(metrics['davies_bouldin_index'])
            
            results[method] = method_results
        
        return results
    
    def plot_linkage_comparison(self, comparison_results: Dict, 
                               save_path: Optional[str] = None):
        """Plot comparison of different linkage methods."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Silhouette scores
        for method, results in comparison_results.items():
            ax1.plot(results['n_clusters'], results['silhouette_scores'], 
                    marker='o', label=method)
        
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Silhouette Score by Linkage Method')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Davies-Bouldin scores (lower is better)
        for method, results in comparison_results.items():
            ax2.plot(results['n_clusters'], results['davies_bouldin_scores'], 
                    marker='o', label=method)
        
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Davies-Bouldin Index')
        ax2.set_title('Davies-Bouldin Index by Linkage Method (Lower is Better)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved linkage comparison plot to {save_path}")
        
        plt.close()
    
    def get_cluster_statistics(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """Calculate statistics for each cluster."""
        if self.labels is None:
            raise ValueError("Model must be fitted first")
        
        stats = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels == cluster_id
            cluster_data = X[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            stats[f'cluster_{cluster_id}'] = {
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(self.labels) * 100),
                'feature_means': {},
                'feature_stds': {},
                'cohesion': float(np.mean(pdist(cluster_data))) if len(cluster_data) > 1 else 0
            }
            
            # Calculate feature statistics
            for i, feature in enumerate(feature_names):
                mean_val = float(np.mean(cluster_data[:, i]))
                std_val = float(np.std(cluster_data[:, i]))
                stats[f'cluster_{cluster_id}']['feature_means'][feature] = mean_val
                stats[f'cluster_{cluster_id}']['feature_stds'][feature] = std_val
        
        return stats
    
    def save_model(self, filepath: str):
        """Save the clustering model."""
        import joblib
        model_data = {
            'model': self.model,
            'n_clusters': self.n_clusters,
            'linkage_method': self.linkage_method,
            'distance_metric': self.distance_metric,
            'linkage_matrix': self.linkage_matrix,
            'labels': self.labels
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved clustering model."""
        import joblib
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.n_clusters = model_data['n_clusters']
        self.linkage_method = model_data['linkage_method']
        self.distance_metric = model_data['distance_metric']
        self.linkage_matrix = model_data['linkage_matrix']
        self.labels = model_data['labels']
        logger.info(f"Model loaded from {filepath}")