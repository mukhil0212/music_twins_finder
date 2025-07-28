#!/usr/bin/env python3
"""
Simple script for users to collect their own Spotify data
Run this script to generate your personal music dataset
"""

import os
import sys
import json
from datetime import datetime

# Add the src directory to the path
sys.path.append('.')

def collect_user_data():
    """Collect data for the currently authenticated Spotify user."""
    try:
        from src.data_collection.spotify_collector import SpotifyCollector
        
        print("🎵 Starting Spotify data collection...")
        print("This will collect your top tracks, artists, and listening patterns.")
        print()
        
        # Initialize collector
        collector = SpotifyCollector()
        
        # Get user info first
        user_info = collector.get_user_info()
        user_id = user_info['id']
        display_name = user_info.get('display_name', user_id)
        
        print(f"✅ Authenticated as: {display_name} ({user_id})")
        print()
        
        # Collect comprehensive data
        print("📊 Collecting your music data...")
        user_data = collector.collect_user_data(user_id)
        
        # Create output directory
        os.makedirs('data/raw', exist_ok=True)
        
        # Save to file
        output_file = f'data/raw/{user_id}_data.json'
        with open(output_file, 'w') as f:
            json.dump(user_data, f, indent=2, default=str)
        
        print(f"✅ Data saved successfully to: {output_file}")
        print()
        print("📈 Data Summary:")
        print(f"   • User: {display_name}")
        print(f"   • Top Tracks: {sum(len(tracks) for tracks in user_data.get('top_tracks', {}).values())}")
        print(f"   • Top Artists: {sum(len(artists) for artists in user_data.get('top_artists', {}).values())}")
        print(f"   • Genres: {len(user_data.get('genre_distribution', {}))}")
        print(f"   • Playlists: {len(user_data.get('playlists', []))}")
        print()
        print("🎯 Your data is ready for music twins analysis!")
        
        return output_file
        
    except ImportError as e:
        print("❌ Error: Missing dependencies")
        print("Please install requirements: pip install -r requirements.txt")
        return None
        
    except Exception as e:
        print(f"❌ Error collecting data: {str(e)}")
        print()
        print("💡 Troubleshooting:")
        print("1. Make sure you have valid Spotify API credentials in .env file")
        print("2. Ensure you've authenticated with Spotify (run the main app first)")
        print("3. Check that your Spotify app has the required permissions")
        return None

def main():
    """Main function to run data collection."""
    print("=" * 60)
    print("🎵 SPOTIFY DATA COLLECTOR")
    print("=" * 60)
    print()
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ Error: .env file not found")
        print("Please create a .env file with your Spotify API credentials")
        print()
        print("Required format:")
        print("SPOTIFY_CLIENT_ID=your_client_id")
        print("SPOTIFY_CLIENT_SECRET=your_client_secret")
        print("SPOTIFY_REDIRECT_URI=http://127.0.0.1:8888/callback")
        return
    
    # Collect data
    output_file = collect_user_data()
    
    if output_file:
        print("=" * 60)
        print("🎉 SUCCESS! Your Spotify data has been collected.")
        print(f"📁 File location: {output_file}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ FAILED! Data collection was unsuccessful.")
        print("Please check the error messages above and try again.")
        print("=" * 60)

if __name__ == "__main__":
    main()
