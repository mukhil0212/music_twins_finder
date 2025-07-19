import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

from config.spotify_config import SpotifyConfig
from .evaluation_metrics import ClusteringEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KMeansClustering:
    def __init__(self, n_clusters: Optional[int] = None, random_state: int = SpotifyConfig.RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_centers = None
        self.labels = None
        self.evaluator = ClusteringEvaluator()
        
    def find_optimal_k(self, X: np.ndarray, k_range: Optional[range] = None) -> Dict:
        """Find optimal number of clusters using elbow method and silhouette score."""
        if k_range is None:
            k_range = range(SpotifyConfig.MIN_CLUSTERS, SpotifyConfig.MAX_CLUSTERS + 1)
        
        logger.info(f"Finding optimal K in range {k_range.start} to {k_range.stop - 1}")
        
        results = {
            'k_values': [],
            'inertias': [],
            'silhouette_scores': [],
            'davies_bouldin_scores': [],
            'calinski_harabasz_scores': []
        }
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            results['k_values'].append(k)
            results['inertias'].append(kmeans.inertia_)
            
            # Calculate evaluation metrics
            metrics = self.evaluator.evaluate_clustering(X, labels)
            results['silhouette_scores'].append(metrics['silhouette_score'])
            results['davies_bouldin_scores'].append(metrics['davies_bouldin_index'])
            results['calinski_harabasz_scores'].append(metrics['calinski_harabasz_index'])
            
            logger.info(f"K={k}: Silhouette={metrics['silhouette_score']:.3f}, "
                       f"DB={metrics['davies_bouldin_index']:.3f}")
        
        # Find optimal k using elbow method
        optimal_k = self._find_elbow_point(results['k_values'], results['inertias'])
        
        # Also consider silhouette score
        best_silhouette_k = results['k_values'][np.argmax(results['silhouette_scores'])]
        
        results['optimal_k_elbow'] = optimal_k
        results['optimal_k_silhouette'] = best_silhouette_k
        results['recommended_k'] = optimal_k  # Can be adjusted based on domain knowledge
        
        return results
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point in the inertia curve."""
        # Calculate distances from each point to the line connecting first and last points
        p1 = np.array([k_values[0], inertias[0]])
        p2 = np.array([k_values[-1], inertias[-1]])
        
        distances = []
        for i in range(len(k_values)):
            p = np.array([k_values[i], inertias[i]])
            distance = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
            distances.append(distance)
        
        # The elbow is the point with maximum distance
        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]
    
    def fit(self, X: np.ndarray, n_clusters: Optional[int] = None) -> 'KMeansClustering':
        """Fit K-means clustering model."""
        if n_clusters:
            self.n_clusters = n_clusters
        elif self.n_clusters is None:
            # Find optimal k if not specified
            results = self.find_optimal_k(X)
            self.n_clusters = results['recommended_k']
            logger.info(f"Using optimal k={self.n_clusters}")
        
        logger.info(f"Fitting K-means with {self.n_clusters} clusters...")
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.labels = self.kmeans.fit_predict(X)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Evaluate clustering
        metrics = self.evaluator.evaluate_clustering(X, self.labels)
        logger.info(f"Clustering metrics: {metrics}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.kmeans is None:
            raise ValueError("Model must be fitted before prediction")
        return self.kmeans.predict(X)
    
    def fit_predict(self, X: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """Fit model and predict cluster labels."""
        self.fit(X, n_clusters)
        return self.labels
    
    def get_cluster_statistics(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """Calculate statistics for each cluster."""
        if self.labels is None:
            raise ValueError("Model must be fitted first")
        
        stats = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.labels == cluster_id
            cluster_data = X[cluster_mask]
            
            stats[f'cluster_{cluster_id}'] = {
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(self.labels) * 100),
                'center': self.cluster_centers[cluster_id].tolist(),
                'feature_means': {},
                'feature_stds': {},
                'top_features': []
            }
            
            # Calculate feature statistics
            for i, feature in enumerate(feature_names):
                mean_val = float(np.mean(cluster_data[:, i]))
                std_val = float(np.std(cluster_data[:, i]))
                stats[f'cluster_{cluster_id}']['feature_means'][feature] = mean_val
                stats[f'cluster_{cluster_id}']['feature_stds'][feature] = std_val
            
            # Find top distinguishing features (highest deviation from global mean)
            global_means = np.mean(X, axis=0)
            cluster_means = np.mean(cluster_data, axis=0)
            deviations = np.abs(cluster_means - global_means)
            top_indices = np.argsort(deviations)[-5:][::-1]
            
            for idx in top_indices:
                stats[f'cluster_{cluster_id}']['top_features'].append({
                    'feature': feature_names[idx],
                    'cluster_mean': float(cluster_means[idx]),
                    'global_mean': float(global_means[idx]),
                    'deviation': float(deviations[idx])
                })
        
        return stats
    
    def get_cluster_profiles(self, profiles_df: pd.DataFrame) -> pd.DataFrame:
        """Get detailed profiles for each cluster."""
        if self.labels is None:
            raise ValueError("Model must be fitted first")
        
        # Add cluster assignments to profiles
        profiles_df['cluster'] = self.labels
        
        # Create cluster summary
        cluster_profiles = []
        
        for cluster_id in range(self.n_clusters):
            cluster_users = profiles_df[profiles_df['cluster'] == cluster_id]
            
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_users),
                'user_ids': cluster_users['user_id'].tolist()
            }
            
            # Add average features
            numeric_cols = cluster_users.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['cluster']:
                    profile[f'avg_{col}'] = float(cluster_users[col].mean())
            
            cluster_profiles.append(profile)
        
        return pd.DataFrame(cluster_profiles)
    
    def plot_elbow_curve(self, results: Dict, save_path: Optional[str] = None):
        """Plot elbow curve for K selection."""
        plt.figure(figsize=(12, 5))
        
        # Elbow curve
        plt.subplot(1, 2, 1)
        plt.plot(results['k_values'], results['inertias'], 'bo-')
        plt.axvline(x=results['optimal_k_elbow'], color='r', linestyle='--', 
                   label=f"Elbow at k={results['optimal_k_elbow']}")
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares')
        plt.title('Elbow Method For Optimal k')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Silhouette scores
        plt.subplot(1, 2, 2)
        plt.plot(results['k_values'], results['silhouette_scores'], 'go-')
        plt.axvline(x=results['optimal_k_silhouette'], color='r', linestyle='--',
                   label=f"Best silhouette at k={results['optimal_k_silhouette']}")
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score vs k')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved elbow curve plot to {save_path}")
        
        plt.close()
    
    def save_model(self, filepath: str):
        """Save the clustering model."""
        import joblib
        model_data = {
            'kmeans': self.kmeans,
            'n_clusters': self.n_clusters,
            'cluster_centers': self.cluster_centers,
            'labels': self.labels
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved clustering model."""
        import joblib
        model_data = joblib.load(filepath)
        self.kmeans = model_data['kmeans']
        self.n_clusters = model_data['n_clusters']
        self.cluster_centers = model_data['cluster_centers']
        self.labels = model_data['labels']
        logger.info(f"Model loaded from {filepath}")