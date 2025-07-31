"""
Music Taste Feature Extraction
Simplified feature extraction focused on music taste compatibility analysis.
Designed to work well with small datasets (2 users) without over-normalization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicTasteExtractor:
    """Simplified feature extractor for music taste compatibility."""
    
    def __init__(self):
        self.audio_features = [
            'danceability', 'energy', 'valence', 'acousticness', 
            'instrumentalness', 'speechiness', 'liveness', 'tempo', 'loudness'
        ]
        
    def extract_user_profile(self, user_data: Dict) -> Dict:
        """Extract music taste profile for a single user."""
        profile = {}
        
        # Core audio features (averages - most important for taste)
        audio_profile = self._extract_audio_profile(user_data.get('audio_features', []))
        profile.update(audio_profile)
        
        # Genre preferences
        genre_profile = self._extract_genre_profile(user_data.get('genre_distribution', {}))
        profile.update(genre_profile)
        
        # Artist diversity
        artist_profile = self._extract_artist_profile(user_data)
        profile.update(artist_profile)
        
        # Track characteristics
        track_profile = self._extract_track_profile(user_data)
        profile.update(track_profile)
        
        return profile
    
    def _extract_audio_profile(self, audio_features: List[Dict]) -> Dict:
        """Extract core audio feature averages."""
        profile = {}
        
        if not audio_features:
            # Return neutral values if no features
            for feature in self.audio_features:
                if feature in ['tempo']:
                    profile[f"{feature}_avg"] = 120.0  # Neutral tempo
                elif feature in ['loudness']:
                    profile[f"{feature}_avg"] = -10.0  # Neutral loudness
                else:
                    profile[f"{feature}_avg"] = 0.5  # Neutral for 0-1 features
            return profile
        
        # Calculate averages for each audio feature
        for feature in self.audio_features:
            values = [track.get(feature, 0) for track in audio_features if track.get(feature) is not None]
            
            if values:
                avg_val = np.mean(values)
                # Keep original scale - don't normalize
                profile[f"{feature}_avg"] = float(avg_val)
            else:
                if feature == 'tempo':
                    profile[f"{feature}_avg"] = 120.0
                elif feature == 'loudness':
                    profile[f"{feature}_avg"] = -10.0
                else:
                    profile[f"{feature}_avg"] = 0.5
        
        # Calculate some derived features
        if 'energy_avg' in profile and 'valence_avg' in profile:
            profile['energy_valence_product'] = profile['energy_avg'] * profile['valence_avg']
            profile['mood_score'] = (profile['energy_avg'] + profile['valence_avg']) / 2
        
        return profile
    
    def _extract_genre_profile(self, genre_distribution: Dict) -> Dict:
        """Extract genre preference features."""
        profile = {}
        
        if not genre_distribution:
            profile['genre_diversity'] = 0.0
            profile['main_genre_dominance'] = 0.0
            return profile
        
        # Genre diversity (entropy-like measure)
        total_tracks = sum(genre_distribution.values())
        if total_tracks > 0:
            proportions = [count / total_tracks for count in genre_distribution.values()]
            profile['genre_diversity'] = -sum(p * np.log2(p + 1e-10) for p in proportions)
            
            # Main genre dominance
            max_genre_count = max(genre_distribution.values())
            profile['main_genre_dominance'] = max_genre_count / total_tracks
        else:
            profile['genre_diversity'] = 0.0
            profile['main_genre_dominance'] = 0.0
        
        # Top genre preferences (binary features for major genres)
        major_genres = ['pop', 'rock', 'hip hop', 'electronic', 'indie', 'r&b', 'country', 'jazz']
        
        for genre in major_genres:
            # Check if genre appears in user's distribution (case-insensitive)
            has_genre = any(genre.lower() in g.lower() for g in genre_distribution.keys())
            profile[f"prefers_{genre.replace(' ', '_')}"] = 1.0 if has_genre else 0.0
        
        return profile
    
    def _extract_artist_profile(self, user_data: Dict) -> Dict:
        """Extract artist-related features."""
        profile = {}
        
        # Collect all artists
        all_artists = set()
        
        # From top tracks
        for time_range in ['short_term', 'medium_term', 'long_term']:
            tracks = user_data.get('top_tracks', {}).get(time_range, [])
            for track in tracks:
                if 'artists' in track:
                    for artist in track['artists']:
                        if isinstance(artist, str):
                            all_artists.add(artist)
                        elif isinstance(artist, dict) and 'name' in artist:
                            all_artists.add(artist['name'])
        
        # From top artists
        for time_range in ['short_term', 'medium_term', 'long_term']:
            artists = user_data.get('top_artists', {}).get(time_range, [])
            for artist in artists:
                if isinstance(artist, str):
                    all_artists.add(artist)
                elif isinstance(artist, dict) and 'name' in artist:
                    all_artists.add(artist['name'])
        
        profile['artist_diversity'] = len(all_artists)
        profile['artist_diversity_log'] = np.log2(len(all_artists) + 1)
        
        return profile
    
    def _extract_track_profile(self, user_data: Dict) -> Dict:
        """Extract track-level characteristics."""
        profile = {}
        
        # Collect track popularity
        popularities = []
        durations = []
        
        for time_range in ['short_term', 'medium_term', 'long_term']:
            tracks = user_data.get('top_tracks', {}).get(time_range, [])
            for track in tracks:
                if 'popularity' in track:
                    popularities.append(track['popularity'])
                if 'duration_ms' in track:
                    durations.append(track['duration_ms'] / 1000)  # Convert to seconds
        
        # Popularity preferences
        if popularities:
            profile['avg_track_popularity'] = np.mean(popularities)
            profile['prefers_mainstream'] = 1.0 if np.mean(popularities) > 60 else 0.0
        else:
            profile['avg_track_popularity'] = 50.0
            profile['prefers_mainstream'] = 0.5
        
        # Duration preferences
        if durations:
            profile['avg_track_duration'] = np.mean(durations)
            profile['prefers_long_tracks'] = 1.0 if np.mean(durations) > 240 else 0.0  # > 4 minutes
        else:
            profile['avg_track_duration'] = 210.0  # 3.5 minutes
            profile['prefers_long_tracks'] = 0.5
        
        return profile
    
    def compare_users(self, user1_data: Dict, user2_data: Dict) -> Dict:
        """Compare two users and return detailed similarity analysis."""
        
        # Extract profiles
        user1_profile = self.extract_user_profile(user1_data)
        user2_profile = self.extract_user_profile(user2_data)
        
        logger.info(f"User 1 profile keys: {list(user1_profile.keys())[:10]}...")
        logger.info(f"User 2 profile keys: {list(user2_profile.keys())[:10]}...")
        
        # Calculate similarities
        similarities = {}
        
        # Audio feature similarities
        audio_similarities = self._compare_audio_features(user1_profile, user2_profile)
        similarities.update(audio_similarities)
        
        # Genre similarities
        genre_similarities = self._compare_genres(user1_data, user2_data, user1_profile, user2_profile)
        similarities.update(genre_similarities)
        
        # Artist similarities
        artist_similarities = self._compare_artists(user1_data, user2_data)
        similarities.update(artist_similarities)
        
        # Overall compatibility score
        compatibility_score = self._calculate_overall_compatibility(similarities)
        
        return {
            'user1_profile': user1_profile,
            'user2_profile': user2_profile,
            'similarities': similarities,
            'compatibility_score': compatibility_score,
            'twin_level': self._determine_twin_level(compatibility_score)
        }
    
    def _compare_audio_features(self, profile1: Dict, profile2: Dict) -> Dict:
        """Compare audio features between two profiles."""
        similarities = {}
        
        # Core audio features with different weights
        feature_weights = {
            'danceability_avg': 0.15,
            'energy_avg': 0.15,
            'valence_avg': 0.15,
            'acousticness_avg': 0.10,
            'instrumentalness_avg': 0.05,
            'speechiness_avg': 0.05,
            'tempo_avg': 0.10,
            'loudness_avg': 0.10,
            'mood_score': 0.15
        }
        
        weighted_similarity_sum = 0.0
        total_weight = 0.0
        
        for feature, weight in feature_weights.items():
            if feature in profile1 and feature in profile2:
                val1 = profile1[feature]
                val2 = profile2[feature]
                
                # Calculate similarity based on feature type
                if feature == 'tempo_avg':
                    # Tempo similarity (normalize difference)
                    max_diff = 80  # Reasonable tempo difference range
                    diff = abs(val1 - val2)
                    similarity = max(0, 1 - (diff / max_diff))
                elif feature == 'loudness_avg':
                    # Loudness similarity 
                    max_diff = 20  # dB range
                    diff = abs(val1 - val2)
                    similarity = max(0, 1 - (diff / max_diff))
                else:
                    # For 0-1 features, use 1 - absolute difference
                    similarity = 1 - abs(val1 - val2)
                
                similarities[f"{feature}_similarity"] = similarity
                weighted_similarity_sum += similarity * weight
                total_weight += weight
        
        # Overall audio similarity
        if total_weight > 0:
            similarities['audio_overall_similarity'] = weighted_similarity_sum / total_weight
        else:
            similarities['audio_overall_similarity'] = 0.5
        
        return similarities
    
    def _compare_genres(self, user1_data: Dict, user2_data: Dict, profile1: Dict, profile2: Dict) -> Dict:
        """Compare genre preferences."""
        similarities = {}
        
        genre1 = user1_data.get('genre_distribution', {})
        genre2 = user2_data.get('genre_distribution', {})
        
        if not genre1 or not genre2:
            similarities['genre_overlap'] = 0.0
            similarities['genre_similarity'] = 0.0
            return similarities
        
        # Calculate genre overlap (Jaccard similarity)
        set1 = set(genre1.keys())
        set2 = set(genre2.keys())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        similarities['genre_overlap'] = intersection / union if union > 0 else 0.0
        
        # Calculate weighted genre similarity
        total_similarity = 0.0
        common_genres = set1.intersection(set2)
        
        if common_genres:
            for genre in common_genres:
                # Normalize by total tracks
                total1 = sum(genre1.values())
                total2 = sum(genre2.values())
                
                prop1 = genre1[genre] / total1 if total1 > 0 else 0
                prop2 = genre2[genre] / total2 if total2 > 0 else 0
                
                # Similarity for this genre (1 - absolute difference in proportions)
                genre_sim = 1 - abs(prop1 - prop2)
                total_similarity += genre_sim
            
            similarities['genre_similarity'] = total_similarity / len(common_genres)
        else:
            similarities['genre_similarity'] = 0.0
        
        return similarities
    
    def _compare_artists(self, user1_data: Dict, user2_data: Dict) -> Dict:
        """Compare artist preferences."""
        similarities = {}
        
        # Collect artists from both users
        artists1 = self._get_user_artists(user1_data)
        artists2 = self._get_user_artists(user2_data)
        
        if not artists1 or not artists2:
            similarities['artist_overlap'] = 0.0
            return similarities
        
        # Calculate artist overlap
        intersection = len(artists1.intersection(artists2))
        union = len(artists1.union(artists2))
        
        similarities['artist_overlap'] = intersection / union if union > 0 else 0.0
        similarities['shared_artists_count'] = intersection
        
        return similarities
    
    def _get_user_artists(self, user_data: Dict) -> set:
        """Get all artists for a user."""
        artists = set()
        
        # From top tracks
        for time_range in ['short_term', 'medium_term', 'long_term']:
            tracks = user_data.get('top_tracks', {}).get(time_range, [])
            for track in tracks:
                if 'artists' in track:
                    for artist in track['artists']:
                        if isinstance(artist, str):
                            artists.add(artist.lower())
                        elif isinstance(artist, dict) and 'name' in artist:
                            artists.add(artist['name'].lower())
        
        # From top artists
        for time_range in ['short_term', 'medium_term', 'long_term']:
            top_artists = user_data.get('top_artists', {}).get(time_range, [])
            for artist in top_artists:
                if isinstance(artist, str):
                    artists.add(artist.lower())
                elif isinstance(artist, dict) and 'name' in artist:
                    artists.add(artist['name'].lower())
        
        return artists
    
    def _calculate_overall_compatibility(self, similarities: Dict) -> float:
        """Calculate overall compatibility score."""
        
        # Weighted components
        weights = {
            'audio_overall_similarity': 0.40,  # Core audio features
            'genre_similarity': 0.25,         # Genre compatibility  
            'genre_overlap': 0.15,            # Shared genres
            'artist_overlap': 0.20             # Shared artists
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for component, weight in weights.items():
            if component in similarities:
                weighted_sum += similarities[component] * weight
                total_weight += weight
        
        # Normalize to 0-1 range
        if total_weight > 0:
            compatibility = weighted_sum / total_weight
        else:
            compatibility = 0.0
        
        return max(0.0, min(1.0, compatibility))
    
    def _determine_twin_level(self, score: float) -> Tuple[str, bool]:
        """Determine twin level based on compatibility score."""
        if score >= 0.85:
            return "Perfect Music Twins", True
        elif score >= 0.75:
            return "Music Twins", True
        elif score >= 0.65:
            return "Very Similar Taste", False
        elif score >= 0.50:
            return "Similar Taste", False
        elif score >= 0.35:
            return "Some Common Ground", False
        else:
            return "Different Musical Tastes", False