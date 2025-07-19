import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityMatcher:
    def __init__(self, metric: str = 'cosine'):
        self.metric = metric
        self.valid_metrics = ['cosine', 'euclidean', 'manhattan', 'minkowski']
        
        if metric not in self.valid_metrics:
            raise ValueError(f"Metric must be one of {self.valid_metrics}")
        
        self.user_features = None
        self.user_ids = None
        self.cluster_labels = None
        self.nn_model = None
        
    def fit(self, user_features: np.ndarray, user_ids: List[str], 
            cluster_labels: Optional[np.ndarray] = None):
        """Fit the similarity matcher with user features."""
        self.user_features = user_features
        self.user_ids = user_ids
        self.cluster_labels = cluster_labels
        
        # Fit nearest neighbors model for efficient similarity search
        if self.metric == 'cosine':
            # For cosine similarity, we normalize and use euclidean distance
            # This is mathematically equivalent but more efficient
            normalized_features = user_features / np.linalg.norm(user_features, axis=1, keepdims=True)
            self.nn_model = NearestNeighbors(metric='euclidean')
            self.nn_model.fit(normalized_features)
        else:
            self.nn_model = NearestNeighbors(metric=self.metric)
            self.nn_model.fit(user_features)
        
        logger.info(f"Fitted similarity matcher with {len(user_ids)} users")
        
    def find_similar_users(self, user_id: str, top_n: int = 10, 
                          same_cluster_only: bool = False) -> List[Dict]:
        """Find top N similar users for a given user."""
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not found in fitted data")
        
        user_idx = self.user_ids.index(user_id)
        user_features = self.user_features[user_idx].reshape(1, -1)
        
        # Filter by cluster if requested
        if same_cluster_only and self.cluster_labels is not None:
            user_cluster = self.cluster_labels[user_idx]
            cluster_mask = self.cluster_labels == user_cluster
            cluster_indices = np.where(cluster_mask)[0]
            
            # Search within cluster
            cluster_features = self.user_features[cluster_mask]
            if self.metric == 'cosine':
                cluster_features = cluster_features / np.linalg.norm(cluster_features, axis=1, keepdims=True)
                user_features = user_features / np.linalg.norm(user_features)
            
            distances, indices = self.nn_model.kneighbors(user_features, n_neighbors=min(top_n + 1, len(cluster_indices)))
            
            # Map back to original indices
            indices = cluster_indices[indices[0]]
            distances = distances[0]
        else:
            # Search all users
            if self.metric == 'cosine':
                user_features = user_features / np.linalg.norm(user_features)
            
            distances, indices = self.nn_model.kneighbors(user_features, n_neighbors=min(top_n + 1, len(self.user_ids)))
            indices = indices[0]
            distances = distances[0]
        
        # Convert distances to similarities
        if self.metric == 'cosine':
            # For normalized vectors, euclidean distance relates to cosine similarity
            similarities = 1 - (distances ** 2) / 2
        elif self.metric == 'euclidean':
            # Convert distance to similarity (inverse)
            max_distance = np.max(distances[1:]) if len(distances) > 1 else 1
            similarities = 1 - (distances / max_distance)
        else:
            # Generic distance to similarity conversion
            similarities = 1 / (1 + distances)
        
        # Create results (excluding the user themselves)
        similar_users = []
        for idx, (user_idx, similarity) in enumerate(zip(indices, similarities)):
            if self.user_ids[user_idx] != user_id:
                similar_user = {
                    'user_id': self.user_ids[user_idx],
                    'similarity_score': float(similarity),
                    'rank': len(similar_users) + 1
                }
                
                if self.cluster_labels is not None:
                    similar_user['cluster'] = int(self.cluster_labels[user_idx])
                
                similar_users.append(similar_user)
                
                if len(similar_users) >= top_n:
                    break
        
        return similar_users
    
    def find_similar_users_with_explanation(self, user_id: str, top_n: int = 10,
                                          feature_names: List[str] = None) -> List[Dict]:
        """Find similar users with explanations of why they're similar."""
        similar_users = self.find_similar_users(user_id, top_n)
        
        if not feature_names:
            return similar_users
        
        # Get user features
        user_idx = self.user_ids.index(user_id)
        user_features = self.user_features[user_idx]
        
        # Add explanations
        for similar_user in similar_users:
            similar_idx = self.user_ids.index(similar_user['user_id'])
            similar_features = self.user_features[similar_idx]
            
            # Find most similar features
            feature_diffs = np.abs(user_features - similar_features)
            most_similar_indices = np.argsort(feature_diffs)[:5]  # Top 5 most similar features
            
            similar_user['similar_features'] = [
                {
                    'feature': feature_names[idx],
                    'user_value': float(user_features[idx]),
                    'similar_user_value': float(similar_features[idx]),
                    'difference': float(feature_diffs[idx])
                }
                for idx in most_similar_indices
            ]
            
            # Find most different features
            most_different_indices = np.argsort(feature_diffs)[-3:]  # Top 3 most different
            similar_user['different_features'] = [
                {
                    'feature': feature_names[idx],
                    'user_value': float(user_features[idx]),
                    'similar_user_value': float(similar_features[idx]),
                    'difference': float(feature_diffs[idx])
                }
                for idx in most_different_indices
            ]
        
        return similar_users
    
    def calculate_similarity_matrix(self, user_ids: Optional[List[str]] = None) -> np.ndarray:
        """Calculate full similarity matrix between users."""
        if user_ids:
            # Calculate for subset of users
            indices = [self.user_ids.index(uid) for uid in user_ids if uid in self.user_ids]
            features_subset = self.user_features[indices]
        else:
            features_subset = self.user_features
        
        if self.metric == 'cosine':
            similarity_matrix = cosine_similarity(features_subset)
        elif self.metric == 'euclidean':
            distances = euclidean_distances(features_subset)
            # Convert to similarity
            max_distance = np.max(distances)
            similarity_matrix = 1 - (distances / max_distance)
        else:
            # Use sklearn's pairwise distances
            from sklearn.metrics import pairwise_distances
            distances = pairwise_distances(features_subset, metric=self.metric)
            similarity_matrix = 1 / (1 + distances)
        
        return similarity_matrix
    
    def find_taste_twins(self, user_id: str, similarity_threshold: float = 0.8) -> List[Dict]:
        """Find users who are extremely similar (taste twins)."""
        similar_users = self.find_similar_users(user_id, top_n=50)  # Get more candidates
        
        # Filter by similarity threshold
        taste_twins = [
            user for user in similar_users 
            if user['similarity_score'] >= similarity_threshold
        ]
        
        return taste_twins[:10]  # Return top 10 taste twins
    
    def cross_cluster_search(self, user_id: str, top_n_per_cluster: int = 3) -> Dict[int, List[Dict]]:
        """Find similar users from each cluster."""
        if self.cluster_labels is None:
            raise ValueError("Cluster labels not provided")
        
        user_idx = self.user_ids.index(user_id)
        user_cluster = self.cluster_labels[user_idx]
        
        results = {}
        unique_clusters = np.unique(self.cluster_labels)
        
        for cluster in unique_clusters:
            if cluster == user_cluster:
                # Find more similar users in the same cluster
                cluster_results = self.find_similar_users(
                    user_id, top_n=top_n_per_cluster * 2, same_cluster_only=True
                )
            else:
                # Find similar users in other clusters
                cluster_mask = self.cluster_labels == cluster
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) == 0:
                    continue
                
                # Calculate similarities with users in this cluster
                user_features = self.user_features[user_idx].reshape(1, -1)
                cluster_features = self.user_features[cluster_mask]
                
                if self.metric == 'cosine':
                    similarities = cosine_similarity(user_features, cluster_features)[0]
                else:
                    distances = pairwise_distances(
                        user_features, cluster_features, metric=self.metric
                    )[0]
                    similarities = 1 / (1 + distances)
                
                # Get top users from this cluster
                top_indices = np.argsort(similarities)[-top_n_per_cluster:][::-1]
                
                cluster_results = []
                for idx in top_indices:
                    cluster_results.append({
                        'user_id': self.user_ids[cluster_indices[idx]],
                        'similarity_score': float(similarities[idx]),
                        'cluster': int(cluster)
                    })
            
            results[int(cluster)] = cluster_results
        
        return results
    
    def get_diversity_score(self, user_id: str, similar_users: List[Dict]) -> float:
        """Calculate how diverse the similar users are."""
        if not similar_users:
            return 0.0
        
        # Get features of similar users
        similar_indices = [self.user_ids.index(user['user_id']) for user in similar_users]
        similar_features = self.user_features[similar_indices]
        
        # Calculate pairwise distances between similar users
        if len(similar_features) > 1:
            distances = pairwise_distances(similar_features, metric=self.metric)
            # Average distance (excluding diagonal)
            mask = np.ones_like(distances, dtype=bool)
            np.fill_diagonal(mask, False)
            diversity = np.mean(distances[mask])
        else:
            diversity = 0.0
        
        return float(diversity)
    
    def get_similarity_statistics(self) -> Dict:
        """Calculate overall similarity statistics."""
        # Sample similarity calculations to avoid memory issues with large datasets
        n_samples = min(1000, len(self.user_ids))
        sample_indices = np.random.choice(len(self.user_ids), n_samples, replace=False)
        
        sample_features = self.user_features[sample_indices]
        similarity_matrix = self.calculate_similarity_matrix(
            [self.user_ids[i] for i in sample_indices]
        )
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        stats = {
            'mean_similarity': float(np.mean(upper_triangle)),
            'std_similarity': float(np.std(upper_triangle)),
            'min_similarity': float(np.min(upper_triangle)),
            'max_similarity': float(np.max(upper_triangle)),
            'median_similarity': float(np.median(upper_triangle))
        }
        
        # Distribution of similarities
        hist, bin_edges = np.histogram(upper_triangle, bins=20)
        stats['similarity_distribution'] = {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
        
        return stats