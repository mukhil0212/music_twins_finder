import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

from config.spotify_config import SpotifyConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    def __init__(self, scaling_method='standard'):
        self.scaling_method = scaling_method
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None  # No scaling
        self.feature_names = []
        
    def extract_features(self, user_data: Dict) -> np.ndarray:
        """Extract comprehensive audio features from user data."""
        features = {}
        
        # Extract audio feature statistics
        audio_stats = self._extract_audio_statistics(user_data.get('audio_features', []))
        features.update(audio_stats)
        
        # Extract genre features
        genre_features = self._extract_genre_features(user_data.get('genre_distribution', {}))
        features.update(genre_features)
        
        # Extract temporal features
        temporal_features = self._extract_temporal_features(user_data.get('listening_patterns', {}))
        features.update(temporal_features)
        
        # Extract artist diversity metrics
        diversity_features = self._extract_diversity_metrics(user_data)
        features.update(diversity_features)
        
        # Extract track popularity features
        popularity_features = self._extract_popularity_features(user_data)
        features.update(popularity_features)
        
        # Store feature names for later reference
        if not self.feature_names:
            self.feature_names = list(features.keys())
        
        # Convert to numpy array
        feature_vector = np.array([features[name] for name in self.feature_names])
        
        return feature_vector
    
    def _extract_audio_statistics(self, audio_features: List[Dict]) -> Dict:
        """Extract statistical features from audio features."""
        stats = {}
        
        if not audio_features:
            # Return zeros if no audio features available
            for feature in SpotifyConfig.AUDIO_FEATURES:
                stats[f"{feature}_mean"] = 0
                stats[f"{feature}_std"] = 0
                stats[f"{feature}_median"] = 0
                stats[f"{feature}_min"] = 0
                stats[f"{feature}_max"] = 0
            return stats
        
        # Calculate statistics for each audio feature
        for feature in SpotifyConfig.AUDIO_FEATURES:
            values = [f[feature] for f in audio_features if f and feature in f]
            
            if values:
                stats[f"{feature}_mean"] = np.mean(values)
                stats[f"{feature}_std"] = np.std(values)
                stats[f"{feature}_median"] = np.median(values)
                stats[f"{feature}_min"] = np.min(values)
                stats[f"{feature}_max"] = np.max(values)
            else:
                stats[f"{feature}_mean"] = 0
                stats[f"{feature}_std"] = 0
                stats[f"{feature}_median"] = 0
                stats[f"{feature}_min"] = 0
                stats[f"{feature}_max"] = 0
        
        # Add derived features
        stats['energy_valence_ratio'] = stats['energy_mean'] / (stats['valence_mean'] + 0.01)
        stats['acoustic_electronic_ratio'] = stats['acousticness_mean'] / (1 - stats['acousticness_mean'] + 0.01)
        stats['mood_variance'] = np.mean([stats['valence_std'], stats['energy_std']])
        
        return stats
    
    def _extract_genre_features(self, genre_distribution: Dict) -> Dict:
        """Extract features from genre distribution."""
        features = {}
        
        # Top genres (binary encoding for most common genres)
        common_genres = [
            'pop', 'rock', 'hip hop', 'electronic', 'indie', 'classical',
            'jazz', 'metal', 'r&b', 'country', 'latin', 'alternative'
        ]
        
        for genre in common_genres:
            features[f"genre_{genre.replace(' ', '_')}"] = 0
            
        for genre, weight in genre_distribution.items():
            for common_genre in common_genres:
                if common_genre in genre.lower():
                    features[f"genre_{common_genre.replace(' ', '_')}"] += weight
        
        # Genre diversity metrics
        if genre_distribution:
            genre_weights = list(genre_distribution.values())
            features['genre_diversity'] = -sum(p * np.log(p + 1e-10) for p in genre_weights)  # Shannon entropy
            features['genre_count'] = len(genre_distribution)
            features['genre_concentration'] = max(genre_weights) if genre_weights else 0
        else:
            features['genre_diversity'] = 0
            features['genre_count'] = 0
            features['genre_concentration'] = 0
        
        return features
    
    def _extract_temporal_features(self, listening_patterns: Dict) -> Dict:
        """Extract temporal listening pattern features."""
        features = {}
        
        # Hour distribution features
        hour_dist = listening_patterns.get('hour_distribution', [0] * 24)
        if hour_dist and any(hour_dist):
            # Peak listening hours
            peak_hour = np.argmax(hour_dist)
            features['peak_listening_hour'] = peak_hour
            
            # Morning (6-12), Afternoon (12-18), Evening (18-24), Night (0-6)
            features['morning_listening'] = sum(hour_dist[6:12])
            features['afternoon_listening'] = sum(hour_dist[12:18])
            features['evening_listening'] = sum(hour_dist[18:24])
            features['night_listening'] = sum(hour_dist[0:6])
            
            # Listening time concentration
            features['hour_concentration'] = max(hour_dist)
            features['hour_entropy'] = -sum(p * np.log(p + 1e-10) for p in hour_dist if p > 0)
        else:
            features.update({
                'peak_listening_hour': 0,
                'morning_listening': 0,
                'afternoon_listening': 0,
                'evening_listening': 0,
                'night_listening': 0,
                'hour_concentration': 0,
                'hour_entropy': 0
            })
        
        # Day distribution features
        day_dist = listening_patterns.get('day_distribution', [0] * 7)
        if day_dist and any(day_dist):
            features['weekend_ratio'] = (day_dist[5] + day_dist[6]) / sum(day_dist)
            features['weekday_concentration'] = max(day_dist[:5]) if day_dist[:5] else 0
        else:
            features['weekend_ratio'] = 0
            features['weekday_concentration'] = 0
        
        # Listening diversity
        features['listening_diversity'] = listening_patterns.get('listening_diversity', 0)
        
        return features
    
    def _extract_diversity_metrics(self, user_data: Dict) -> Dict:
        """Extract artist and track diversity metrics."""
        features = {}
        
        # Artist diversity across time ranges
        all_artist_ids = set()
        time_range_artists = {}
        
        for time_range in SpotifyConfig.TIME_RANGES:
            artists = user_data.get('top_artists', {}).get(time_range, [])
            artist_ids = {artist['id'] for artist in artists}
            time_range_artists[time_range] = artist_ids
            all_artist_ids.update(artist_ids)
        
        features['total_unique_artists'] = len(all_artist_ids)
        
        # Artist consistency across time ranges
        if len(time_range_artists) >= 2:
            time_ranges = list(time_range_artists.values())
            consistency_scores = []
            for i in range(len(time_ranges)):
                for j in range(i + 1, len(time_ranges)):
                    if len(time_ranges[i]) > 0 and len(time_ranges[j]) > 0:
                        overlap = len(time_ranges[i].intersection(time_ranges[j]))
                        total = len(time_ranges[i].union(time_ranges[j]))
                        consistency_scores.append(overlap / total if total > 0 else 0)
            features['artist_consistency'] = np.mean(consistency_scores) if consistency_scores else 0
        else:
            features['artist_consistency'] = 0
        
        # Track diversity
        all_track_ids = set()
        for time_range in SpotifyConfig.TIME_RANGES:
            tracks = user_data.get('top_tracks', {}).get(time_range, [])
            all_track_ids.update(track['id'] for track in tracks)
        
        features['total_unique_tracks'] = len(all_track_ids)
        
        # Playlist diversity
        playlists = user_data.get('playlists', [])
        features['playlist_count'] = len(playlists)
        features['avg_playlist_size'] = np.mean([p['tracks_total'] for p in playlists]) if playlists else 0
        
        return features
    
    def _extract_popularity_features(self, user_data: Dict) -> Dict:
        """Extract popularity-based features."""
        features = {}
        
        # Track popularity statistics
        all_popularities = []
        for time_range in SpotifyConfig.TIME_RANGES:
            tracks = user_data.get('top_tracks', {}).get(time_range, [])
            all_popularities.extend([track['popularity'] for track in tracks])
        
        if all_popularities:
            features['track_popularity_mean'] = np.mean(all_popularities)
            features['track_popularity_std'] = np.std(all_popularities)
            features['track_popularity_min'] = np.min(all_popularities)
            features['track_popularity_max'] = np.max(all_popularities)
        else:
            features.update({
                'track_popularity_mean': 0,
                'track_popularity_std': 0,
                'track_popularity_min': 0,
                'track_popularity_max': 0
            })
        
        # Artist popularity statistics
        artist_popularities = []
        artist_followers = []
        for time_range in SpotifyConfig.TIME_RANGES:
            artists = user_data.get('top_artists', {}).get(time_range, [])
            artist_popularities.extend([artist['popularity'] for artist in artists])
            artist_followers.extend([artist['followers'] for artist in artists])
        
        if artist_popularities:
            features['artist_popularity_mean'] = np.mean(artist_popularities)
            features['artist_popularity_std'] = np.std(artist_popularities)
        else:
            features['artist_popularity_mean'] = 0
            features['artist_popularity_std'] = 0
        
        if artist_followers:
            features['artist_followers_mean'] = np.mean(artist_followers)
            features['artist_followers_std'] = np.std(artist_followers)
        else:
            features['artist_followers_mean'] = 0
            features['artist_followers_std'] = 0
        
        return features
    
    def fit_transform(self, users_data: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """Extract and scale features for multiple users."""
        logger.info(f"Extracting features for {len(users_data)} users...")
        
        # Extract features for all users
        feature_vectors = []
        user_ids = []
        
        for user_data in users_data:
            try:
                features = self.extract_features(user_data)
                feature_vectors.append(features)
                user_ids.append(user_data['user_id'])
            except Exception as e:
                logger.error(f"Failed to extract features for user {user_data.get('user_id', 'unknown')}: {str(e)}")
                continue
        
        if not feature_vectors:
            raise ValueError("No valid feature vectors extracted")
        
        # Convert to numpy array
        X = np.array(feature_vectors)

        # Scale features if scaler is available
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
            logger.info(f"Extracted and scaled {X_scaled.shape[1]} features for {X_scaled.shape[0]} users")
        else:
            X_scaled = X
            logger.info(f"Extracted {X_scaled.shape[1]} raw features for {X_scaled.shape[0]} users (no scaling)")

        return X_scaled, user_ids
    
    def transform(self, users_data: List[Dict]) -> np.ndarray:
        """Transform new user data using fitted scaler."""
        feature_vectors = []
        
        for user_data in users_data:
            features = self.extract_features(user_data)
            feature_vectors.append(features)
        
        X = np.array(feature_vectors)
        return self.scaler.transform(X)
    
    def get_feature_importance(self, X: np.ndarray) -> pd.DataFrame:
        """Calculate feature importance based on variance."""
        feature_variance = np.var(X, axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'variance': feature_variance,
            'importance': feature_variance / np.sum(feature_variance)
        })
        
        return importance_df.sort_values('importance', ascending=False)