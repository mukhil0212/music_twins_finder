import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import KMeansClustering, HierarchicalClustering
from src.clustering.evaluation_metrics import ClusteringEvaluator
from src.utils.helpers import create_sample_data
from src.feature_engineering import UserProfileBuilder


class TestKMeansClustering:
    @pytest.fixture
    def sample_features(self):
        """Create sample feature data."""
        np.random.seed(42)
        # Create 3 distinct clusters
        cluster1 = np.random.normal(0, 0.5, (20, 5))
        cluster2 = np.random.normal(3, 0.5, (20, 5))
        cluster3 = np.random.normal(-3, 0.5, (20, 5))
        X = np.vstack([cluster1, cluster2, cluster3])
        return X
    
    def test_kmeans_initialization(self):
        """Test KMeans initialization."""
        kmeans = KMeansClustering(n_clusters=3, random_state=42)
        assert kmeans.n_clusters == 3
        assert kmeans.random_state == 42
        assert kmeans.kmeans is None
    
    def test_find_optimal_k(self, sample_features):
        """Test optimal k finding."""
        kmeans = KMeansClustering()
        results = kmeans.find_optimal_k(sample_features, k_range=range(2, 6))
        
        assert 'k_values' in results
        assert 'inertias' in results
        assert 'silhouette_scores' in results
        assert 'optimal_k_elbow' in results
        assert 'optimal_k_silhouette' in results
        assert 'recommended_k' in results
        
        # Should identify 3 clusters
        assert results['optimal_k_silhouette'] == 3
    
    def test_fit_predict(self, sample_features):
        """Test fit and predict."""
        kmeans = KMeansClustering(n_clusters=3)
        labels = kmeans.fit_predict(sample_features)
        
        assert len(labels) == len(sample_features)
        assert len(np.unique(labels)) == 3
        assert kmeans.cluster_centers is not None
        assert kmeans.cluster_centers.shape == (3, 5)
    
    def test_cluster_statistics(self, sample_features):
        """Test cluster statistics calculation."""
        kmeans = KMeansClustering(n_clusters=3)
        kmeans.fit(sample_features)
        
        feature_names = [f'feature_{i}' for i in range(5)]
        stats = kmeans.get_cluster_statistics(sample_features, feature_names)
        
        assert len(stats) == 3
        for cluster_key in stats:
            assert 'size' in stats[cluster_key]
            assert 'percentage' in stats[cluster_key]
            assert 'feature_means' in stats[cluster_key]
            assert 'top_features' in stats[cluster_key]


class TestHierarchicalClustering:
    @pytest.fixture
    def sample_features(self):
        """Create sample feature data."""
        np.random.seed(42)
        # Create 2 distinct clusters
        cluster1 = np.random.normal(0, 0.5, (15, 4))
        cluster2 = np.random.normal(2, 0.5, (15, 4))
        X = np.vstack([cluster1, cluster2])
        return X
    
    def test_hierarchical_initialization(self):
        """Test Hierarchical clustering initialization."""
        hc = HierarchicalClustering(n_clusters=2, linkage_method='ward')
        assert hc.n_clusters == 2
        assert hc.linkage_method == 'ward'
        assert hc.model is None
    
    def test_fit_predict(self, sample_features):
        """Test fit and predict."""
        hc = HierarchicalClustering(n_clusters=2)
        labels = hc.fit_predict(sample_features)
        
        assert len(labels) == len(sample_features)
        assert len(np.unique(labels)) == 2
        assert hc.linkage_matrix is not None
    
    def test_compare_linkage_methods(self, sample_features):
        """Test linkage method comparison."""
        hc = HierarchicalClustering()
        results = hc.compare_linkage_methods(sample_features)
        
        assert 'ward' in results
        assert 'complete' in results
        assert 'average' in results
        
        for method, method_results in results.items():
            assert 'n_clusters' in method_results
            assert 'silhouette_scores' in method_results


class TestClusteringEvaluator:
    @pytest.fixture
    def sample_data_and_labels(self):
        """Create sample data with known clusters."""
        np.random.seed(42)
        # Create 3 well-separated clusters
        cluster1 = np.random.normal(0, 0.3, (20, 3))
        cluster2 = np.random.normal(5, 0.3, (20, 3))
        cluster3 = np.random.normal(-5, 0.3, (20, 3))
        X = np.vstack([cluster1, cluster2, cluster3])
        labels = np.array([0]*20 + [1]*20 + [2]*20)
        return X, labels
    
    def test_evaluate_clustering(self, sample_data_and_labels):
        """Test clustering evaluation metrics."""
        X, labels = sample_data_and_labels
        evaluator = ClusteringEvaluator()
        
        metrics = evaluator.evaluate_clustering(X, labels)
        
        assert 'n_clusters' in metrics
        assert 'silhouette_score' in metrics
        assert 'davies_bouldin_index' in metrics
        assert 'calinski_harabasz_index' in metrics
        assert 'within_cluster_sum_of_squares' in metrics
        
        # Good clustering should have high silhouette score
        assert metrics['silhouette_score'] > 0.7
        # Good clustering should have low Davies-Bouldin index
        assert metrics['davies_bouldin_index'] < 1.0
    
    def test_stability_analysis(self, sample_data_and_labels):
        """Test clustering stability analysis."""
        X, _ = sample_data_and_labels
        evaluator = ClusteringEvaluator()
        
        def clustering_func(data):
            kmeans = KMeansClustering(n_clusters=3)
            return kmeans.fit_predict(data)
        
        stability_results = evaluator.stability_analysis(
            X, clustering_func, n_iterations=5, subsample_ratio=0.8
        )
        
        assert 'iteration_metrics' in stability_results
        assert 'label_consistency_mean' in stability_results
        assert len(stability_results['iteration_metrics']) == 5
        
        # Should have consistent results for well-separated clusters
        assert stability_results['label_consistency_mean'] > 0.8


class TestIntegration:
    def test_full_clustering_pipeline(self):
        """Test complete clustering pipeline with sample data."""
        # Create sample data
        sample_users = create_sample_data(n_users=50, random_state=42)
        
        # Build profiles
        builder = UserProfileBuilder()
        profiles_df, feature_names = builder.build_profiles(sample_users)
        
        # Extract features
        feature_cols = [col for col in profiles_df.columns 
                       if col not in ['user_id', 'display_name', 'cluster_assignment']]
        X = profiles_df[feature_cols].values
        
        # Find optimal k
        kmeans = KMeansClustering()
        optimal_k_results = kmeans.find_optimal_k(X, k_range=range(2, 8))
        
        # Fit with optimal k
        optimal_k = optimal_k_results['recommended_k']
        labels = kmeans.fit_predict(X, n_clusters=optimal_k)
        
        # Evaluate
        evaluator = ClusteringEvaluator()
        metrics = evaluator.evaluate_clustering(X, labels)
        
        # Basic assertions
        assert len(labels) == len(profiles_df)
        assert len(np.unique(labels)) == optimal_k
        assert metrics['silhouette_score'] > 0  # Should have some structure
        
        # Get cluster statistics
        stats = kmeans.get_cluster_statistics(X, feature_cols)
        assert len(stats) == optimal_k
        
        # Check cluster sizes are reasonable
        for cluster_stat in stats.values():
            assert cluster_stat['size'] > 0
            assert cluster_stat['percentage'] > 0


if __name__ == '__main__':
    pytest.main([__file__])