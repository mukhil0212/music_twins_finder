#!/usr/bin/env python3
"""
Test script to diagnose Spotify API authentication issues
"""
import sys
import os
sys.path.append('.')

from src.data_collection.spotify_collector import SpotifyCollector
from config.spotify_config import SpotifyConfig

def test_spotify_auth():
    """Test Spotify authentication and basic API calls"""
    try:
        print("Initializing Spotify collector...")
        collector = SpotifyCollector()
        
        print("Testing user authentication...")
        user_info = collector.get_user_info()
        print(f"✓ Authenticated as: {user_info['display_name']} ({user_info['id']})")
        
        print("Testing top tracks API...")
        tracks = collector.get_top_tracks(limit=5)
        print(f"✓ Retrieved {len(tracks)} top tracks")
        
        if tracks:
            track_ids = [track['id'] for track in tracks[:3]]
            print(f"Testing audio features API with track IDs: {track_ids}")
            features = collector.get_audio_features(track_ids)
            print(f"✓ Retrieved audio features for {len(features)} tracks")
            
            if features:
                print("Sample audio features:")
                for i, feature in enumerate(features[:1]):
                    if feature:
                        print(f"  Track {i+1}: danceability={feature.get('danceability')}, energy={feature.get('energy')}")
                    else:
                        print(f"  Track {i+1}: No features available")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_spotify_auth()
    sys.exit(0 if success else 1)