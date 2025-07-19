import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.similarity import SimilarityMatcher
from src.utils.helpers import create_sample_data
from src.feature_engineering import UserProfileBuilder


class TestSimilarityMatcher:
    @pytest.fixture
    def sample_features_and_ids(self):
        """Create sample feature data with user IDs."""
        np.random.seed(42)
        n_users = 20
        n_features = 5
        
        # Create features with some similar users
        features = np.random.rand(n_users, n_features)
        # Make some users very similar
        features[1] = features[0] + np.random.normal(0, 0.01, n_features)
        features[5] = features[4] + np.random.normal(0, 0.01, n_features)
        
        user_ids = [f'user_{i}' for i in range(n_users)]
        cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2] * 2)[:n_users]
        
        return features, user_ids, cluster_labels
    
    def test_similarity_matcher_initialization(self):
        """Test SimilarityMatcher initialization."""
        matcher = SimilarityMatcher(metric='cosine')
        assert matcher.metric == 'cosine'
        assert matcher.user_features is None
        assert matcher.user_ids is None
        
        # Test invalid metric
        with pytest.raises(ValueError):
            SimilarityMatcher(metric='invalid_metric')
    
    def test_fit(self, sample_features_and_ids):
        """Test fitting the similarity matcher."""
        features, user_ids, cluster_labels = sample_features_and_ids
        
        matcher = SimilarityMatcher(metric='cosine')
        matcher.fit(features, user_ids, cluster_labels)
        
        assert matcher.user_features is not None
        assert matcher.user_ids == user_ids
        assert matcher.cluster_labels is not None
        assert matcher.nn_model is not None
    
    def test_find_similar_users(self, sample_features_and_ids):
        """Test finding similar users."""
        features, user_ids, cluster_labels = sample_features_and_ids
        
        matcher = SimilarityMatcher(metric='cosine')
        matcher.fit(features, user_ids, cluster_labels)
        
        # Find similar users for user_0
        similar_users = matcher.find_similar_users('user_0', top_n=5)
        
        assert len(similar_users) == 5
        assert all('user_id' in user for user in similar_users)
        assert all('similarity_score' in user for user in similar_users)
        assert all('rank' in user for user in similar_users)
        
        # user_1 should be most similar to user_0
        assert similar_users[0]['user_id'] == 'user_1'
        assert similar_users[0]['similarity_score'] > 0.9
        
        # Check descending order of similarity
        similarities = [user['similarity_score'] for user in similar_users]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_find_similar_users_same_cluster(self, sample_features_and_ids):
        """Test finding similar users within same cluster."""
        features, user_ids, cluster_labels = sample_features_and_ids
        
        matcher = SimilarityMatcher(metric='cosine')
        matcher.fit(features, user_ids, cluster_labels)
        
        # Find similar users in same cluster
        similar_users = matcher.find_similar_users('user_0', top_n=3, same_cluster_only=True)
        
        # All should be from cluster 0
        user_cluster = cluster_labels[0]
        for user in similar_users:
            user_idx = user_ids.index(user['user_id'])
            assert cluster_labels[user_idx] == user_cluster
    
    def test_find_taste_twins(self, sample_features_and_ids):
        """Test finding taste twins with high similarity."""
        features, user_ids, cluster_labels = sample_features_and_ids
        
        matcher = SimilarityMatcher(metric='cosine')
        matcher.fit(features, user_ids, cluster_labels)
        
        # Find taste twins for user_0
        twins = matcher.find_taste_twins('user_0', similarity_threshold=0.9)
        
        # Should find user_1 as twin
        assert len(twins) >= 1
        assert twins[0]['user_id'] == 'user_1'
        assert all(user['similarity_score'] >= 0.9 for user in twins)
    
    def test_calculate_similarity_matrix(self, sample_features_and_ids):
        """Test similarity matrix calculation."""
        features, user_ids, cluster_labels = sample_features_and_ids
        
        matcher = SimilarityMatcher(metric='cosine')
        matcher.fit(features, user_ids, cluster_labels)
        
        # Calculate for subset
        subset_users = user_ids[:5]
        sim_matrix = matcher.calculate_similarity_matrix(subset_users)
        
        assert sim_matrix.shape == (5, 5)
        assert np.allclose(np.diag(sim_matrix), 1.0)  # Self-similarity should be 1
        assert np.allclose(sim_matrix, sim_matrix.T)  # Should be symmetric
    
    def test_cross_cluster_search(self, sample_features_and_ids):
        """Test cross-cluster similarity search."""
        features, user_ids, cluster_labels = sample_features_and_ids
        
        matcher = SimilarityMatcher(metric='cosine')
        matcher.fit(features, user_ids, cluster_labels)
        
        # Find similar users across clusters
        results = matcher.cross_cluster_search('user_0', top_n_per_cluster=2)
        
        assert len(results) <= len(np.unique(cluster_labels))
        for cluster, users in results.items():
            assert len(users) <= 2
            assert all('similarity_score' in user for user in users)
    
    def test_different_metrics(self, sample_features_and_ids):
        """Test different similarity metrics."""
        features, user_ids, cluster_labels = sample_features_and_ids
        
        for metric in ['cosine', 'euclidean', 'manhattan']:
            matcher = SimilarityMatcher(metric=metric)
            matcher.fit(features, user_ids, cluster_labels)
            
            similar_users = matcher.find_similar_users('user_0', top_n=5)
            assert len(similar_users) == 5
            assert all(0 <= user['similarity_score'] <= 1 for user in similar_users)


class TestSimilarityIntegration:
    def test_full_similarity_pipeline(self):
        """Test complete similarity matching pipeline."""
        # Create sample data
        sample_users = create_sample_data(n_users=30, random_state=42)
        
        # Build profiles
        builder = UserProfileBuilder()
        profiles_df, feature_names = builder.build_profiles(sample_users)
        
        # Extract features
        feature_cols = [col for col in profiles_df.columns 
                       if col not in ['user_id', 'display_name', 'cluster_assignment']]
        X = profiles_df[feature_cols].values
        user_ids = profiles_df['user_id'].tolist()
        
        # Create mock clusters
        cluster_labels = np.array([i % 3 for i in range(len(profiles_df))])
        
        # Create and fit matcher
        matcher = SimilarityMatcher(metric='cosine')
        matcher.fit(X, user_ids, cluster_labels)
        
        # Test various functions
        target_user = user_ids[0]
        
        # Find similar users
        similar_users = matcher.find_similar_users(target_user, top_n=10)
        assert len(similar_users) == 10
        
        # Find with explanations
        similar_with_exp = matcher.find_similar_users_with_explanation(
            target_user, top_n=5, feature_names=feature_cols
        )
        assert len(similar_with_exp) == 5
        assert all('similar_features' in user for user in similar_with_exp)
        assert all('different_features' in user for user in similar_with_exp)
        
        # Get similarity statistics
        stats = matcher.get_similarity_statistics()
        assert 'mean_similarity' in stats
        assert 'std_similarity' in stats
        assert 'similarity_distribution' in stats
        
        # Test diversity score
        diversity = matcher.get_diversity_score(target_user, similar_users[:5])
        assert isinstance(diversity, float)
        assert diversity >= 0


if __name__ == '__main__':
    pytest.main([__file__])