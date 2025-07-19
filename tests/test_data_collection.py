import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import SpotifyAuth, SpotifyCollector
from src.utils.helpers import create_sample_data


class TestSpotifyAuth:
    @patch('spotipy.Spotify')
    @patch('spotipy.oauth2.SpotifyOAuth')
    def test_spotify_auth_initialization(self, mock_oauth, mock_spotify):
        """Test SpotifyAuth initialization."""
        auth = SpotifyAuth()
        assert auth.auth_manager is not None
        assert auth._spotify is None
        
    @patch('spotipy.Spotify')
    def test_spotify_client_property(self, mock_spotify):
        """Test Spotify client property."""
        auth = SpotifyAuth()
        client = auth.spotify
        assert client is not None
        assert auth._spotify is not None


class TestSpotifyCollector:
    @patch('src.data_collection.spotify_collector.SpotifyAuth')
    def test_collector_initialization(self, mock_auth):
        """Test SpotifyCollector initialization."""
        collector = SpotifyCollector()
        assert collector.data_dir is not None
        assert os.path.exists(collector.data_dir)
    
    def test_extract_genres(self):
        """Test genre extraction from artists."""
        collector = SpotifyCollector()
        artists = [
            {'genres': ['rock', 'indie rock', 'alternative']},
            {'genres': ['rock', 'classic rock']},
            {'genres': ['indie', 'indie rock']}
        ]
        
        genre_dist = collector.extract_genres(artists)
        
        assert 'rock' in genre_dist
        assert 'indie rock' in genre_dist
        assert sum(genre_dist.values()) == pytest.approx(1.0)
    
    def test_extract_temporal_patterns(self):
        """Test temporal pattern extraction."""
        collector = SpotifyCollector()
        recently_played = [
            {'played_at': '2023-01-01T10:00:00Z', 'track': {'id': 'track1'}},
            {'played_at': '2023-01-01T14:00:00Z', 'track': {'id': 'track2'}},
            {'played_at': '2023-01-01T20:00:00Z', 'track': {'id': 'track3'}},
            {'played_at': '2023-01-02T10:00:00Z', 'track': {'id': 'track1'}}
        ]
        
        patterns = collector.extract_temporal_patterns(recently_played)
        
        assert 'hour_distribution' in patterns
        assert 'day_distribution' in patterns
        assert 'listening_diversity' in patterns
        assert len(patterns['hour_distribution']) == 24
        assert len(patterns['day_distribution']) == 7
        assert sum(patterns['hour_distribution']) == pytest.approx(1.0)
    
    @patch.object(SpotifyCollector, 'spotify')
    def test_collect_user_data_structure(self, mock_spotify):
        """Test collected user data structure."""
        # Mock Spotify API responses
        mock_spotify.current_user.return_value = {
            'id': 'test_user',
            'display_name': 'Test User'
        }
        mock_spotify.current_user_top_tracks.return_value = {
            'items': [{'id': f'track_{i}', 'name': f'Track {i}', 
                      'artists': [{'name': 'Artist'}], 
                      'popularity': 80, 'duration_ms': 200000} 
                     for i in range(5)],
            'next': None
        }
        mock_spotify.current_user_top_artists.return_value = {
            'items': [{'id': f'artist_{i}', 'name': f'Artist {i}',
                      'genres': ['rock'], 'popularity': 70,
                      'followers': {'total': 10000}}
                     for i in range(5)],
            'next': None
        }
        mock_spotify.audio_features.return_value = [
            {'id': f'track_{i}', 'danceability': 0.5, 'energy': 0.7}
            for i in range(5)
        ]
        mock_spotify.current_user_recently_played.return_value = {
            'items': []
        }
        mock_spotify.current_user_playlists.return_value = {
            'items': [],
            'next': None
        }
        
        collector = SpotifyCollector()
        user_data = collector.collect_user_data(save_to_file=False)
        
        # Check structure
        assert 'user_id' in user_data
        assert 'display_name' in user_data
        assert 'top_tracks' in user_data
        assert 'top_artists' in user_data
        assert 'audio_features' in user_data
        assert 'genre_distribution' in user_data
        assert 'listening_patterns' in user_data
        assert 'summary_stats' in user_data


class TestSampleData:
    def test_create_sample_data(self):
        """Test sample data creation."""
        n_users = 10
        sample_data = create_sample_data(n_users=n_users, random_state=42)
        
        assert len(sample_data) == n_users
        
        for user in sample_data:
            assert 'user_id' in user
            assert 'display_name' in user
            assert 'audio_features' in user
            assert 'genre_distribution' in user
            assert 'listening_patterns' in user
            assert 'summary_stats' in user
            
            # Check audio features
            assert len(user['audio_features']) > 0
            
            # Check genre distribution
            assert sum(user['genre_distribution'].values()) == pytest.approx(1.0)
            
            # Check listening patterns
            assert len(user['listening_patterns']['hour_distribution']) == 24
            assert sum(user['listening_patterns']['hour_distribution']) == pytest.approx(1.0)
    
    def test_sample_data_clustering_structure(self):
        """Test that sample data has clustering structure."""
        sample_data = create_sample_data(n_users=100, random_state=42)
        
        # Extract energy values for different user types
        energy_values = {i: [] for i in range(4)}
        
        for i, user in enumerate(sample_data):
            cluster_type = i % 4
            energy_mean = user['summary_stats']['energy_mean']
            energy_values[cluster_type].append(energy_mean)
        
        # Check that different clusters have different characteristics
        cluster_means = [np.mean(values) for values in energy_values.values()]
        assert len(set(cluster_means)) > 1  # Not all means are the same


if __name__ == '__main__':
    pytest.main([__file__])