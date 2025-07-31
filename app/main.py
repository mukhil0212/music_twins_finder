from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import AudioFeatureExtractor
from src.clustering import KMeansClustering
from src.similarity import SimilarityMatcher
from src.visualization import (
    DimensionalityReducer, HeatmapGenerator
)
from src.utils.helpers import load_json, save_json, create_sample_data
from config.spotify_config import SpotifyConfig

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

app = Flask(__name__)
# Removed secret key - no longer using sessions

# Global variables for storing model and data
model_data = {
    'user_profiles': None,
    'clustering_model': None,
    'similarity_matcher': None,
    'dimensionality_reducer': None,
    'feature_names': None
}

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/twins')
def twins_page():
    """Music twins comparison page."""
    return render_template('index.html')

@app.route('/demo')
def demo():
    """Demo mode with real ML analysis on sample data."""
    try:
        # Import the demo analyzer
        from src.demo.demo_analyzer import DemoAnalyzer
        
        # Create and run demo analysis
        demo_analyzer = DemoAnalyzer()
        results = demo_analyzer.analyze_demo_users()
        
        return jsonify(results)
        
    except Exception as e:
        # Fallback to mock data if demo analysis fails
        print(f"Demo analysis failed: {str(e)}")
        
        # Generate basic fallback data
        fallback_results = {
            'comparison_summary': {
                'user1': 'Demo User 1',
                'user2': 'Demo User 2',
                'are_twins': False,
                'compatibility_score': 0.65,
                'twin_level': 'Similar Taste',
                'cosine_similarity': 0.68,
                'euclidean_similarity': 0.62,
                'correlation_similarity': 0.65
            },
            'error': f'Demo analysis failed: {str(e)}',
            'is_fallback': True
        }
        
        return jsonify(fallback_results)

# Removed unused authentication endpoints

# Removed all authentication endpoints - using direct dataset analysis

@app.route('/available-datasets')
def available_datasets():
    """List available datasets in data/raw/ directory."""
    try:
        # Get absolute path to raw data directory
        app_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(app_dir)
        raw_data_dir = os.path.join(project_root, 'data', 'raw')

        if not os.path.exists(raw_data_dir):
            return jsonify({
                'datasets': [],
                'message': 'No raw data directory found',
                'directory': raw_data_dir
            })

        # Find all JSON data files
        data_files = [f for f in os.listdir(raw_data_dir) if f.endswith('_data.json')]

        datasets = []
        for file_name in data_files:
            try:
                file_path = os.path.join(raw_data_dir, file_name)
                with open(file_path, 'r') as f:
                    data = json.load(f)

                user_id = data.get('user_id', file_name.replace('_data.json', ''))
                display_name = data.get('display_name', user_id)

                datasets.append({
                    'file_name': file_name,
                    'user_id': user_id,
                    'display_name': display_name,
                    'collected_at': data.get('collected_at', 'Unknown'),
                    'has_tracks': bool(data.get('top_tracks')),
                    'has_artists': bool(data.get('top_artists')),
                    'genre_count': len(data.get('genre_distribution', {}))
                })
            except Exception as e:
                datasets.append({
                    'file_name': file_name,
                    'error': f'Failed to read: {str(e)}'
                })

        return jsonify({
            'datasets': datasets,
            'total_count': len(datasets),
            'directory': raw_data_dir
        })

    except Exception as e:
        return jsonify({'error': f'Failed to list datasets: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze user's Spotify data."""
    try:
        data = request.get_json()
        username = data.get('username')

        if not username:
            return jsonify({'error': 'Username is required'}), 400

        # Important: Spotify API Limitation Notice
        collector = SpotifyCollector()

        # Check if user is trying to analyze someone else's data
        try:
            current_user = collector.get_user_info()
            if username != current_user['id']:
                return jsonify({
                    'error': 'Spotify API Limitation',
                    'message': f'You are authenticated as "{current_user["id"]}" but requested data for "{username}". Due to Spotify\'s privacy policies, you can only analyze your own listening data.',
                    'suggestion': 'Please use your own Spotify username or have the other user authenticate separately.',
                    'authenticated_user': current_user['id'],
                    'requested_user': username
                }), 403
        except Exception as e:
            return jsonify({'error': f'Authentication failed: {str(e)}'}), 401

        # Collect user data (only for authenticated user)
        user_data = collector.collect_user_data(username)
        
        # Load existing user data if available
        existing_data_path = os.path.join('data', 'processed', 'all_users.json')
        if os.path.exists(existing_data_path):
            all_users = load_json(existing_data_path)
            
            # Check if user already exists
            user_exists = any(u['user_id'] == user_data['user_id'] for u in all_users)
            if not user_exists:
                all_users.append(user_data)
                save_json(all_users, existing_data_path)
        else:
            all_users = [user_data]
            save_json(all_users, existing_data_path)
        
        # Process data
        results = process_user_data(all_users, target_user_id=user_data['user_id'])
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compare-twins', methods=['POST'])
def compare_twins():
    """Compare two users to check if they are music twins using enhanced datasets."""
    try:
        data = request.get_json()
        username1 = data.get('username1', '').strip()
        username2 = data.get('username2', '').strip()

        if not username1 or not username2:
            return jsonify({'error': 'Both usernames are required'}), 400

        if username1 == username2:
            return jsonify({'error': 'Please enter different usernames'}), 400

        # Load user data from raw data files
        app_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(app_dir)
        raw_data_dir = os.path.join(project_root, 'data', 'raw')

        # Look for user data files
        user1_file = os.path.join(raw_data_dir, f'{username1}_data.json')
        user2_file = os.path.join(raw_data_dir, f'{username2}_data.json')

        # Check if user1 data exists
        if not os.path.exists(user1_file):
            available_files = [f for f in os.listdir(raw_data_dir) if f.endswith('_data.json')] if os.path.exists(raw_data_dir) else []
            return jsonify({
                'error': f'No data found for {username1}',
                'message': f'Dataset file not found: {username1}_data.json',
                'suggestion': f'Please ensure the dataset file exists in data/raw/',
                'available_datasets': available_files,
                'expected_file': f'{username1}_data.json'
            }), 404

        # Check if user2 data exists
        if not os.path.exists(user2_file):
            available_files = [f for f in os.listdir(raw_data_dir) if f.endswith('_data.json')] if os.path.exists(raw_data_dir) else []
            return jsonify({
                'error': f'No data found for {username2}',
                'message': f'Dataset file not found: {username2}_data.json',
                'suggestion': f'Please ensure the dataset file exists in data/raw/',
                'available_datasets': available_files,
                'expected_file': f'{username2}_data.json'
            }), 404

        # Load user data from files
        try:
            with open(user1_file, 'r') as f:
                user1_data = json.load(f)
            print(f"✅ Loaded data for {username1}")
        except Exception as e:
            return jsonify({
                'error': f'Failed to load data for {username1}',
                'message': f'Error reading {username1}_data.json: {str(e)}'
            }), 500

        try:
            with open(user2_file, 'r') as f:
                user2_data = json.load(f)
            print(f"✅ Loaded data for {username2}")
        except Exception as e:
            return jsonify({
                'error': f'Failed to load data for {username2}',
                'message': f'Error reading {username2}_data.json: {str(e)}'
            }), 500
        
        # Check if datasets have enhanced audio features, enhance if needed
        user1_data = ensure_enhanced_audio_features(user1_data, username1)
        user2_data = ensure_enhanced_audio_features(user2_data, username2)
        
        # Process the two users with enhanced analysis + visualizations
        users_data = [user1_data, user2_data]
        print(f"🔬 Starting ML analysis for {username1} vs {username2}")

        comparison_results = process_twin_comparison_enhanced(users_data, username1, username2)

        # Convert any numpy types to JSON-serializable types
        comparison_results = convert_numpy_types(comparison_results)

        print(f"✅ Analysis complete. Compatibility: {comparison_results['comparison_summary']['compatibility_score']:.2%}")
        print(f"🎯 Are twins: {comparison_results['comparison_summary']['are_twins']}")
        print(f"📊 Features compared: {len(comparison_results['feature_comparison'])}")
        print(f"🎵 Shared traits: {len(comparison_results['shared_traits'])}")

        return jsonify(comparison_results)
        
    except Exception as e:
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500

def ensure_enhanced_audio_features(user_data, username):
    """Ensure user data has enhanced audio features, generate if missing."""
    try:
        # Check if audio features exist and are not empty
        audio_features = user_data.get('audio_features', [])
        
        if not audio_features or len(audio_features) == 0:
            print(f"🔧 Generating enhanced audio features for {username}...")
            
            # Import the audio features fixer
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Load the enhancement logic from fix_audio_features.py
            from fix_audio_features import generate_track_features, get_genre_distribution
            
            # Generate enhanced audio features
            enhanced_features = []
            tracks = user_data.get('top_tracks', [])
            genre_dist = get_genre_distribution(user_data)
            
            for track in tracks:
                features = generate_track_features(track, user_data)
                if features:
                    enhanced_features.append(features)
            
            # Update user data with enhanced features
            user_data['audio_features'] = enhanced_features
            print(f"✅ Generated {len(enhanced_features)} audio features for {username}")
            
        else:
            print(f"✅ User {username} already has {len(audio_features)} audio features")
            
        return user_data
        
    except Exception as e:
        print(f"⚠️ Warning: Could not enhance audio features for {username}: {str(e)}")
        return user_data

def process_twin_comparison_direct(users_data, username1, username2):
    """Direct twin comparison without UserProfileBuilder dependency."""
    print(f"🔬 Processing direct twin comparison: {username1} vs {username2}")

    # Extract user data
    user1_data = users_data[0]
    user2_data = users_data[1]

    # Use AudioFeatureExtractor directly for basic features
    extractor = AudioFeatureExtractor()

    # Extract features for both users (returns numpy arrays)
    user1_vector = extractor.extract_features(user1_data)
    user2_vector = extractor.extract_features(user2_data)

    # Get feature names for detailed comparison
    feature_names = extractor.feature_names if hasattr(extractor, 'feature_names') else []

    # Handle any infinite or NaN values
    user1_vector = np.nan_to_num(user1_vector, nan=0.0, posinf=1e6, neginf=-1e6)
    user2_vector = np.nan_to_num(user2_vector, nan=0.0, posinf=1e6, neginf=-1e6)

    # Apply L2 normalization to each user separately
    user1_norm = np.linalg.norm(user1_vector)
    user2_norm = np.linalg.norm(user2_vector)

    user1_normalized = user1_vector / (user1_norm + 1e-8)
    user2_normalized = user2_vector / (user2_norm + 1e-8)

    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    audio_similarity = cosine_similarity([user1_normalized], [user2_normalized])[0][0]

    # Calculate artist overlap and genre similarity
    artist_overlap, shared_artists_count = calculate_artist_overlap(user1_data, user2_data)
    genre_similarity = calculate_genre_similarity(user1_data, user2_data)

    # Enhanced compatibility calculation
    enhanced_compatibility = (
        audio_similarity * 0.45 +     # Audio features (45% weight)
        artist_overlap * 0.35 +       # Artist overlap (35% weight)
        genre_similarity * 0.20       # Genre similarity (20% weight)
    )

    # Determine if they are twins
    are_twins = enhanced_compatibility >= 0.8
    twin_level = get_twin_level(enhanced_compatibility)

    print(f"🎤 Artist overlap: {artist_overlap:.3f} ({shared_artists_count} shared artists)")
    print(f"🎭 Genre similarity: {genre_similarity:.3f}")
    print(f"🔄 Enhanced compatibility: {enhanced_compatibility:.3f}")

    # Create detailed feature comparison for UI
    detailed_features = {}
    if len(feature_names) == len(user1_vector) == len(user2_vector):
        # Create feature-by-feature comparison
        for i, feature_name in enumerate(feature_names):
            user1_val = float(user1_vector[i])
            user2_val = float(user2_vector[i])

            # Calculate individual feature similarity (1 - normalized difference)
            max_val = max(abs(user1_val), abs(user2_val), 1e-8)
            feature_similarity = 1 - abs(user1_val - user2_val) / max_val

            detailed_features[feature_name] = {
                'user1_value': user1_val,
                'user2_value': user2_val,
                'similarity': feature_similarity
            }

    # Create results structure
    results = {
        'comparison_summary': {
            'user1': username1,
            'user2': username2,
            'are_twins': are_twins,
            'compatibility_score': float(enhanced_compatibility),
            'twin_level': twin_level,
            'audio_similarity': float(audio_similarity),
            'artist_overlap': float(artist_overlap),
            'genre_similarity': float(genre_similarity),
            'shared_artists_count': int(shared_artists_count)
        },
        'feature_comparison': detailed_features,
        'shared_traits': [],
        'recommendations': generate_twin_recommendations(are_twins, enhanced_compatibility, []),
        'analysis_metadata': {
            'algorithm_version': '2.0_enhanced',
            'features_analyzed': len(user1_vector),
            'processing_time': '< 1 second',
            'confidence_level': 'high' if enhanced_compatibility > 0.6 or enhanced_compatibility < 0.4 else 'medium'
        }
    }

    return results

def process_twin_comparison_enhanced(users_data, username1, username2):
    """Enhanced twin comparison with ML visualizations."""
    try:
        # Direct enhanced comparison (no dependency on UserProfileBuilder)
        print(f"🔬 Processing enhanced twin comparison: {username1} vs {username2}")

        # Use the enhanced comparison logic directly
        basic_results = process_twin_comparison_direct(users_data, username1, username2)
        
        # Add ML visualizations using the demo analyzer approach
        viz_dir = os.path.join('static', 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Import demo analyzer
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.demo.demo_analyzer import DemoAnalyzer
            
            # Create a modified demo analyzer to work with our user data
            demo_analyzer = DemoAnalyzer()
            
            # Generate visualizations similar to demo approach
            viz_files = generate_twin_visualizations(users_data, username1, username2, timestamp)
            
            # Convert to the format expected by the UI (same as demo)
            if viz_files:
                visualization_plots = []
                
                for viz_type, filename in viz_files.items():
                    plot_info = {
                        'name': get_visualization_name(viz_type),
                        'path': f'/static/visualizations/{filename}',
                        'type': viz_type,
                        'explanation': get_visualization_explanation_text(viz_type),
                        'interpretation': get_visualization_interpretation(viz_type),
                        'twin_analysis': get_visualization_twin_analysis_text(viz_type),
                        'ml_concept': get_visualization_ml_concept(viz_type)
                    }
                    visualization_plots.append(plot_info)
                
                basic_results['visualizations'] = {
                    'plots': visualization_plots,
                    'timestamp': timestamp,
                    'analysis_type': 'music_twins_comparison'
                }
            
            basic_results['has_visualizations'] = len(viz_files) > 0
            
            print(f"✅ Generated {len(viz_files)} ML visualizations for {username1} vs {username2}")
            
        except Exception as viz_error:
            print(f"⚠️ Visualization generation failed: {str(viz_error)}")
            basic_results['visualization_error'] = str(viz_error)
            basic_results['has_visualizations'] = False
        
        return basic_results
        
    except Exception as e:
        print(f"⚠️ Enhanced comparison failed: {str(e)}")
        # Return a basic error result instead of falling back to broken function
        return {
            'comparison_summary': {
                'user1': username1,
                'user2': username2,
                'are_twins': False,
                'compatibility_score': 0.0,
                'twin_level': 'Error',
                'error': str(e)
            },
            'feature_comparison': {},
            'shared_traits': [],
            'recommendations': [],
            'analysis_metadata': {
                'algorithm_version': '2.0_error',
                'error': str(e)
            }
        }

def generate_twin_visualizations(users_data, username1, username2, timestamp):
    """Generate ML visualizations for twin comparison."""
    viz_files = {}
    
    try:
        # Get absolute path to visualization directory
        app_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(app_dir)
        viz_dir = os.path.join(project_root, 'static', 'visualizations')
        
        print(f"🎨 Generating visualizations in: {viz_dir}")
        
        # Ensure directory exists
        os.makedirs(viz_dir, exist_ok=True)
        
        # Import required modules with matplotlib backend fix
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        print(f"🎯 Generating visualizations for {username1} vs {username2}")
        
        # Generate simplified visualizations using audio features directly
        user1_data = users_data[0]
        user2_data = users_data[1]
        
        # 1. Generate correlation heatmap using direct audio features
        correlation_file = f'correlation_heatmap_{timestamp}.png'
        correlation_path = os.path.join(viz_dir, correlation_file)
        
        print(f"📊 Creating correlation heatmap: {correlation_file}")
        
        # Create correlation matrix from audio features
        audio_features_df = pd.DataFrame()
        audio_feature_keys = ['danceability', 'energy', 'valence', 'acousticness', 'tempo']
        
        for i, user_data in enumerate(users_data):
            user_name = user_data.get('display_name', f'User{i+1}')
            user_features = {}
            
            # Get audio features (use first 10 tracks)
            for feature in user_data.get('audio_features', [])[:10]:
                for key in audio_feature_keys:
                    if key in feature and feature[key] is not None:
                        if key not in user_features:
                            user_features[key] = []
                        user_features[key].append(feature[key])
            
            # Average the features
            for key in audio_feature_keys:
                if key in user_features and user_features[key]:
                    audio_features_df.loc[user_name, key.title()] = np.mean(user_features[key])
                else:
                    audio_features_df.loc[user_name, key.title()] = 0
        
        if not audio_features_df.empty and len(audio_features_df) >= 2:
            plt.figure(figsize=(10, 8))
            correlation_matrix = audio_features_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'})
            plt.title(f'Audio Features Correlation Matrix\n{username1} vs {username2}', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(correlation_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            viz_files['correlation_heatmap'] = correlation_file
            print(f"✅ Created correlation heatmap: {correlation_path}")
        
        # 2. Generate similarity radar chart
        radar_file = f'similarity_radar_{timestamp}.png'
        radar_path = os.path.join(viz_dir, radar_file)
        
        print(f"📡 Creating similarity radar: {radar_file}")
        
        # Calculate feature similarities directly
        feature_similarities = {}
        for feature_key in audio_feature_keys:
            user1_vals = [f.get(feature_key, 0) for f in user1_data.get('audio_features', []) if f.get(feature_key) is not None]
            user2_vals = [f.get(feature_key, 0) for f in user2_data.get('audio_features', []) if f.get(feature_key) is not None]
            
            if user1_vals and user2_vals:
                user1_avg = np.mean(user1_vals)
                user2_avg = np.mean(user2_vals)
                similarity = 1 - abs(user1_avg - user2_avg)
                feature_similarities[feature_key.title()] = max(0, min(1, similarity))
            else:
                feature_similarities[feature_key.title()] = 0
        
        if feature_similarities:
            # Create radar chart
            categories = list(feature_similarities.keys())
            values = list(feature_similarities.values())
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values, 'o-', linewidth=3, label='Similarity Score', color='#1DB954')
            ax.fill(angles, values, alpha=0.25, color='#1DB954')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
            ax.grid(True, alpha=0.3)
            
            plt.title(f'Music Taste Similarity Radar\n{username1} vs {username2}', 
                     fontsize=16, pad=30, fontweight='bold')
            plt.tight_layout()
            plt.savefig(radar_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            viz_files['similarity_radar'] = radar_file
            print(f"✅ Created similarity radar: {radar_path}")
        
        print(f"✅ Generated {len(viz_files)} visualization files")
        return viz_files
        
    except Exception as e:
        print(f"⚠️ Visualization generation error: {str(e)}")
        return {}

def process_twin_comparison(users_data, username1, username2):
    """Process two users and return detailed comparison using REAL ML algorithms."""
    print(f"🔬 Processing twin comparison: {username1} vs {username2}")

    # REAL ML PIPELINE - Feature engineering with modified scaling approach
    profile_builder = UserProfileBuilder()

    # Temporarily modify the feature extractor to use no scaling
    original_scaling = profile_builder.feature_extractor.scaling_method
    profile_builder.feature_extractor.scaling_method = 'none'
    profile_builder.feature_extractor.scaler = None

    try:
        user_profiles, feature_names = profile_builder.build_profiles(users_data)

        print(f"📊 Feature extraction complete: {len(feature_names)} features")
        print(f"👥 User profiles shape: {user_profiles.shape}")
        print(f"🎯 Available features: {feature_names[:10]}...")  # Show first 10 features

        # Get features for analysis, excluding problematic features
        exclude_features = ['artist_followers_mean', 'artist_followers_std', 'artist_followers_median',
                           'artist_followers_min', 'artist_followers_max']  # These cause extreme values

        feature_cols = [col for col in user_profiles.columns
                       if col not in ['user_id', 'display_name', 'cluster_assignment'] + exclude_features]
        X = user_profiles[feature_cols].values

        print(f"🔧 Excluded {len(exclude_features)} problematic features, using {len(feature_cols)} features")

        # Get user indices
        user1_idx = user_profiles[user_profiles['user_id'] == username1].index[0]
        user2_idx = user_profiles[user_profiles['user_id'] == username2].index[0]

        # Get user feature vectors
        user1_features = X[user1_idx].reshape(1, -1)
        user2_features = X[user2_idx].reshape(1, -1)

        # Debug feature vectors
        print(f"🔍 User 1 feature stats: min={X[user1_idx].min():.3f}, max={X[user1_idx].max():.3f}, mean={X[user1_idx].mean():.3f}")
        print(f"🔍 User 2 feature stats: min={X[user2_idx].min():.3f}, max={X[user2_idx].max():.3f}, mean={X[user2_idx].mean():.3f}")

    finally:
        # Restore original scaling method
        profile_builder.feature_extractor.scaling_method = original_scaling

    # Multiple similarity metrics with proper feature normalization
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from scipy.stats import pearsonr
    from sklearn.preprocessing import StandardScaler, RobustScaler
    import numpy as np

    # CRITICAL FIX: Avoid fitting scaler on just 2 users (causes artificial correlation)
    # Use L2 normalization instead of StandardScaler fitted on 2 users
    print(f"🔧 Applying L2 normalization to avoid artificial correlation...")

    # Remove any infinite or NaN values
    X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Extract individual user features
    user1_features = X_clean[user1_idx]
    user2_features = X_clean[user2_idx]

    # Apply L2 normalization to each user separately (avoids artificial correlation)
    user1_norm = np.linalg.norm(user1_features)
    user2_norm = np.linalg.norm(user2_features)

    user1_scaled = (user1_features / (user1_norm + 1e-8)).reshape(1, -1)
    user2_scaled = (user2_features / (user2_norm + 1e-8)).reshape(1, -1)

    print(f"🔍 After scaling - User 1: min={user1_scaled.min():.3f}, max={user1_scaled.max():.3f}, mean={user1_scaled.mean():.3f}")
    print(f"🔍 After scaling - User 2: min={user2_scaled.min():.3f}, max={user2_scaled.max():.3f}, mean={user2_scaled.mean():.3f}")

    # Debug: Check for identical features
    feature_diff = np.abs(user1_scaled[0] - user2_scaled[0])
    identical_features = (feature_diff < 1e-10).sum()
    very_similar_features = (feature_diff < 0.1).sum()
    very_different_features = (feature_diff > 1.5).sum()

    print(f"🔍 Feature analysis:")
    print(f"   Identical features: {identical_features}")
    print(f"   Very similar features (<0.1 diff): {very_similar_features}")
    print(f"   Very different features (>1.5 diff): {very_different_features}")
    print(f"   Average feature difference: {feature_diff.mean():.3f}")

    # Calculate similarities on properly normalized features
    cosine_sim = float(cosine_similarity(user1_scaled, user2_scaled)[0][0])
    euclidean_dist = float(euclidean_distances(user1_scaled, user2_scaled)[0][0])

    print(f"🔍 Raw similarity values:")
    print(f"   Raw cosine: {cosine_sim}")
    print(f"   Raw euclidean distance: {euclidean_dist}")

    # Fix correlation calculation - use Pearson correlation and handle edge cases
    try:
        correlation_coeff, p_value = pearsonr(user1_scaled[0], user2_scaled[0])
        print(f"🔍 Correlation details: coeff={correlation_coeff:.6f}, p-value={p_value:.6f}")

        # Convert correlation to similarity (0 to 1 scale)
        correlation_sim = (correlation_coeff + 1) / 2  # Convert from [-1,1] to [0,1]
        if np.isnan(correlation_sim):
            correlation_sim = 0.5
            print("⚠️ Correlation is NaN, using default 0.5")
    except Exception as e:
        print(f"⚠️ Correlation calculation failed: {e}")
        correlation_sim = 0.5  # Default to neutral if calculation fails

    # If cosine similarity is exactly 0, there might be an issue
    if cosine_sim == 0.0:
        print("⚠️ Cosine similarity is exactly 0 - checking for issues...")
        # Check vector norms
        norm1 = np.linalg.norm(user1_scaled[0])
        norm2 = np.linalg.norm(user2_scaled[0])
        dot_product = np.dot(user1_scaled[0], user2_scaled[0])
        print(f"   Vector 1 norm: {norm1:.6f}")
        print(f"   Vector 2 norm: {norm2:.6f}")
        print(f"   Dot product: {dot_product:.6f}")

        # Manual cosine calculation
        if norm1 > 0 and norm2 > 0:
            manual_cosine = dot_product / (norm1 * norm2)
            print(f"   Manual cosine: {manual_cosine:.6f}")
            cosine_sim = max(0, manual_cosine)

    # Ensure all similarities are in [0,1] range
    cosine_sim = max(0, min(1, cosine_sim))
    correlation_sim = max(0, min(1, correlation_sim))
    euclidean_sim = 1 / (1 + euclidean_dist)  # Convert distance to similarity

    # Simplified compatibility calculation focusing on meaningful metrics
    # Cosine similarity is the primary metric for normalized feature vectors
    print("📊 Analysis: Using cosine similarity as primary metric")

    # Primary compatibility from audio features (cosine similarity on normalized vectors)
    audio_compatibility = cosine_sim

    # Ensure reasonable bounds
    audio_compatibility = max(0.0, min(1.0, audio_compatibility))

    print(f"🔍 Similarity breakdown:")
    print(f"   Audio (Cosine): {audio_compatibility:.3f}")
    print(f"   Artist Overlap: {artist_overlap:.3f}")
    print(f"   Genre Similarity: {genre_similarity:.3f}")
    print(f"   Final Score: {enhanced_compatibility:.3f}")

    # Calculate artist overlap similarity (this was missing!)
    artist_overlap, shared_artists_count = calculate_artist_overlap(users_data[0], users_data[1])

    # Calculate genre distribution similarity
    genre_similarity = calculate_genre_similarity(users_data[0], users_data[1])

    # Enhanced compatibility calculation with rebalanced weights
    # Artist overlap is more important for defining "twins" than genre categories
    enhanced_compatibility = (
        audio_compatibility * 0.45 +  # Audio features via cosine similarity (45% weight)
        artist_overlap * 0.35 +       # Artist overlap (35% weight) - increased importance
        genre_similarity * 0.20       # Genre similarity (20% weight) - reduced due to saturation
    )

    print(f"🎤 Artist overlap: {artist_overlap:.3f} ({shared_artists_count} shared artists)")
    print(f"🎭 Genre similarity: {genre_similarity:.3f}")
    print(f"🔄 Enhanced compatibility: {enhanced_compatibility:.3f} (audio: {audio_compatibility:.3f})")

    # Use enhanced compatibility for final decision
    final_compatibility = enhanced_compatibility

    # Determine if they are twins (threshold: 0.8)
    are_twins = final_compatibility >= 0.8
    twin_level = get_twin_level(final_compatibility)

    # Feature-wise comparison using user profiles
    feature_comparison = {}
    user1_profile = user_profiles.iloc[user1_idx]
    user2_profile = user_profiles.iloc[user2_idx]

    # Compare key audio features from the ML pipeline
    key_features = ['danceability_mean', 'energy_mean', 'valence_mean', 'acousticness_mean',
                   'instrumentalness_mean', 'tempo_mean', 'loudness_mean']

    for feature in key_features:
        if feature in user1_profile and feature in user2_profile:
            val1 = user1_profile[feature]
            val2 = user2_profile[feature]

            # Calculate similarity based on feature type
            if feature == 'tempo_mean':
                # Tempo similarity (normalize by reasonable range)
                max_diff = 80  # BPM range
                diff = abs(val1 - val2)
                similarity = max(0, 1 - (diff / max_diff))
            elif feature == 'loudness_mean':
                # Loudness similarity (normalize by dB range)
                max_diff = 20  # dB range
                diff = abs(val1 - val2)
                similarity = max(0, 1 - (diff / max_diff))
            else:
                # For 0-1 features, use 1 - absolute difference
                diff = abs(val1 - val2)
                similarity = max(0, 1 - diff)

            feature_comparison[feature] = {
                'user1_value': float(val1),
                'user2_value': float(val2),
                'difference': float(diff),
                'similarity': float(similarity)
            }
    
    # Find shared characteristics
    shared_traits = []
    for feature, comparison in feature_comparison.items():
        if comparison['similarity'] > 0.8:  # Very similar
            shared_traits.append({
                'feature': feature.replace('_mean', '').title(),
                'similarity': comparison['similarity'],
                'description': get_feature_description(feature, comparison['user1_value'])
            })

    # Create detailed results
    results = {
        'comparison_summary': {
            'user1': username1,
            'user2': username2,
            'are_twins': are_twins,
            'compatibility_score': float(final_compatibility),
            'twin_level': twin_level,
            'cosine_similarity': float(cosine_sim),
            'correlation_similarity': float(correlation_sim),
            'euclidean_similarity': float(euclidean_sim),
            'artist_overlap': float(artist_overlap),
            'genre_similarity': float(genre_similarity),
            'shared_artists_count': int(shared_artists_count)
        },
        'feature_comparison': feature_comparison,
        'shared_traits': shared_traits,
        'recommendations': generate_twin_recommendations(are_twins, final_compatibility, shared_traits),
        'timestamp': datetime.now().isoformat()
    }
    
    return convert_numpy_types(results)

def calculate_artist_overlap(user1_data, user2_data):
    """Calculate artist overlap between two users."""
    user1_artists = set()
    user2_artists = set()

    # Collect all artists from all time ranges
    for time_range in ['short_term', 'medium_term', 'long_term']:
        user1_artists.update([
            artist['id'] for artist in user1_data.get('top_artists', {}).get(time_range, [])
        ])
        user2_artists.update([
            artist['id'] for artist in user2_data.get('top_artists', {}).get(time_range, [])
        ])

    # Calculate Jaccard similarity
    if not user1_artists or not user2_artists:
        return 0.0, 0

    shared_artists = user1_artists.intersection(user2_artists)
    total_artists = user1_artists.union(user2_artists)

    jaccard_similarity = len(shared_artists) / len(total_artists) if total_artists else 0.0

    return jaccard_similarity, len(shared_artists)

def calculate_genre_similarity(user1_data, user2_data):
    """Calculate genre distribution similarity using Jensen-Shannon divergence."""
    user1_genres = user1_data.get('genre_distribution', {})
    user2_genres = user2_data.get('genre_distribution', {})

    if not user1_genres or not user2_genres:
        return 0.0

    # Get all unique genres in sorted order for consistency
    all_genres = sorted(set(user1_genres.keys()).union(set(user2_genres.keys())))

    # Create probability vectors
    user1_vector = np.array([user1_genres.get(genre, 0) for genre in all_genres])
    user2_vector = np.array([user2_genres.get(genre, 0) for genre in all_genres])

    # Ensure vectors sum to 1 (normalize if needed)
    user1_vector = user1_vector / (user1_vector.sum() + 1e-8)
    user2_vector = user2_vector / (user2_vector.sum() + 1e-8)

    # Calculate Jensen-Shannon divergence (more sensitive to small differences)
    from scipy.spatial.distance import jensenshannon
    js_distance = jensenshannon(user1_vector, user2_vector)
    js_similarity = 1 - js_distance  # Convert distance to similarity [0,1]

    return float(js_similarity)

def get_twin_level(score):
    """Determine the level of musical similarity."""
    if score >= 0.9:
        return "Perfect Twins"
    elif score >= 0.8:
        return "Music Twins"
    elif score >= 0.7:
        return "Very Similar"
    elif score >= 0.6:
        return "Quite Similar"
    elif score >= 0.5:
        return "Somewhat Similar"
    else:
        return "Different Tastes"

def get_music_feature_description(feature, value):
    """Get human-readable description of music feature values."""
    descriptions = {
        'danceability_avg': f"Both love {'highly danceable' if value > 0.7 else 'moderately danceable' if value > 0.4 else 'less danceable'} music",
        'energy_avg': f"Both prefer {'high-energy' if value > 0.7 else 'moderate-energy' if value > 0.4 else 'low-energy'} tracks",
        'valence_avg': f"Both enjoy {'upbeat and positive' if value > 0.7 else 'moderately positive' if value > 0.4 else 'melancholic'} music",
        'acousticness_avg': f"Both like {'acoustic' if value > 0.7 else 'semi-acoustic' if value > 0.4 else 'electronic'} sounds",
        'instrumentalness_avg': f"Both prefer {'instrumental' if value > 0.7 else 'mixed' if value > 0.4 else 'vocal'} music",
        'tempo_avg': f"Both enjoy {'fast-paced' if value > 120 else 'moderate-paced' if value > 90 else 'slow-paced'} music",
        'loudness_avg': f"Both like {'loud' if value > -10 else 'moderate volume' if value > -20 else 'quiet'} music",
        'mood_score': f"Both gravitate toward {'uplifting' if value > 0.7 else 'balanced' if value > 0.4 else 'mellow'} moods"
    }
    return descriptions.get(feature, f"Similar {feature.replace('_avg', '')} preferences")

def get_feature_description(feature, value):
    """Legacy function for backward compatibility."""
    return get_music_feature_description(feature, value)

def generate_twin_recommendations(are_twins, score, shared_traits):
    """Generate recommendations based on comparison results."""
    recommendations = []
    
    if are_twins:
        recommendations.append("🎉 Congratulations! You are Music Twins!")
        recommendations.append("You have remarkably similar music tastes")
        recommendations.append("Consider creating collaborative playlists together")
        recommendations.append("Explore each other's recent discoveries")
    elif score > 0.7:
        recommendations.append("🎵 You have very similar music tastes!")
        recommendations.append("You'd probably enjoy each other's playlists")
        recommendations.append("Try exploring genres you both haven't discovered yet")
    elif score > 0.5:
        recommendations.append("🎶 You share some musical common ground")
        recommendations.append("Focus on your shared traits for music discovery")
        recommendations.append("Introduce each other to your unique preferences")
    else:
        recommendations.append("🎼 You have different but potentially complementary tastes")
        recommendations.append("This could lead to exciting musical discoveries")
        recommendations.append("Share your favorite tracks to expand each other's horizons")
    
    if shared_traits:
        recommendations.append(f"You both excel in: {', '.join([trait['feature'] for trait in shared_traits[:3]])}")
    
    return recommendations

def generate_music_twin_recommendations(are_twins, score, shared_traits, similarities):
    """Generate music-specific recommendations based on comparison results."""
    recommendations = []
    
    # Twin-level recommendations
    if are_twins:
        recommendations.append("🎉 Congratulations! You are Music Twins!")
        recommendations.append("You have remarkably similar music tastes across multiple dimensions")
        recommendations.append("Create collaborative playlists together - you'll love each other's picks")
        recommendations.append("Explore concert recommendations based on your shared preferences")
    elif score > 0.65:
        recommendations.append("🎵 You have very compatible music tastes!")
        recommendations.append("Your musical preferences align well across key features")
        recommendations.append("Try creating genre-specific playlists together")
    elif score > 0.45:
        recommendations.append("🎶 You share some strong musical common ground")
        recommendations.append("Focus on your shared musical traits for discovery")
        recommendations.append("Introduce each other to artists in your preferred genres")
    else:
        recommendations.append("🎼 You have different but potentially complementary tastes")
        recommendations.append("This diversity could lead to exciting musical discovery")
        recommendations.append("Share your top tracks to expand each other's horizons")
    
    # Specific feature-based recommendations
    audio_sim = similarities.get('audio_overall_similarity', 0)
    genre_sim = similarities.get('genre_similarity', 0)
    artist_overlap = similarities.get('artist_overlap', 0)
    
    if audio_sim > 0.7:
        recommendations.append(f"🎧 Strong audio preferences match ({audio_sim:.1%}) - you like similar energy, mood, and style")
    elif audio_sim > 0.4:
        recommendations.append(f"🎧 Moderate audio compatibility ({audio_sim:.1%}) - explore each other's energy preferences")
    
    if genre_sim > 0.6:
        recommendations.append(f"🎪 Great genre compatibility ({genre_sim:.1%}) - dive deeper into your shared genres")
    elif genre_sim > 0.3:
        recommendations.append(f"🎪 Some genre overlap ({genre_sim:.1%}) - try fusion genres that bridge your tastes")
    
    if artist_overlap > 0.2:
        shared_count = similarities.get('shared_artists_count', 0)
        recommendations.append(f"👥 You share {shared_count} artists - explore their full discographies together")
    elif artist_overlap > 0.05:
        recommendations.append("👥 Some artist overlap - check out similar artists in your shared style")
    else:
        recommendations.append("👥 Different artist preferences - perfect opportunity for mutual discovery")
    
    # Trait-based suggestions
    if shared_traits:
        trait_names = [trait['feature'] for trait in shared_traits[:3]]
        recommendations.append(f"✨ Your strongest shared traits: {', '.join(trait_names)}")
    
    return recommendations

def get_visualization_name(viz_type):
    """Get display name for visualization type."""
    names = {
        'correlation_heatmap': 'Audio Features Correlation Matrix',
        'similarity_radar': 'Music Taste Similarity Radar'
    }
    return names.get(viz_type, viz_type.replace('_', ' ').title())

def get_visualization_explanation_text(viz_type):
    """Get explanation text for visualization."""
    explanations = {
        'correlation_heatmap': 'Shows how different audio features (danceability, energy, valence, acousticness, tempo) correlate with each other across both users\' music preferences.',
        'similarity_radar': 'Displays musical compatibility across key audio feature dimensions on a radar chart. Each axis represents a different musical characteristic.'
    }
    return explanations.get(viz_type, 'Musical compatibility visualization.')

def get_visualization_interpretation(viz_type):
    """Get interpretation guidance for visualization."""
    interpretations = {
        'correlation_heatmap': 'Colors show correlation strength: red indicates positive correlation, blue indicates negative correlation. Similar patterns between users suggest compatible music processing.',
        'similarity_radar': 'The filled area shows similarity strength. A larger, more circular shape indicates higher compatibility across all musical dimensions.'
    }
    return interpretations.get(viz_type, 'Analyze the patterns to understand musical compatibility.')

def get_visualization_twin_analysis_text(viz_type):
    """Get twin analysis text for visualization."""
    analyses = {
        'correlation_heatmap': 'When both users show similar correlation patterns between audio features, it indicates they process and enjoy music in comparable ways - a strong sign of being music twins.',
        'similarity_radar': 'The more the radar shape fills toward the outer edge across all dimensions, the stronger the indication that you are music twins. Look for high scores (80%+) across multiple features.'
    }
    return analyses.get(viz_type, 'This visualization helps determine your musical twin potential.')

def get_visualization_ml_concept(viz_type):
    """Get ML concept explanation for visualization."""
    concepts = {
        'correlation_heatmap': 'Pearson correlation coefficients calculated between audio features, revealing the mathematical relationships in your music preferences.',
        'similarity_radar': 'Multi-dimensional feature similarity calculated using absolute difference metrics, normalized to 0-1 scale and displayed on polar coordinates.'
    }
    return concepts.get(viz_type, 'Machine learning analysis of musical compatibility.')

def process_user_data(users_data, target_user_id=None, is_demo=False):
    """Process user data and return analysis results."""
    # Feature engineering
    profile_builder = UserProfileBuilder()
    user_profiles, feature_names = profile_builder.build_profiles(users_data)
    
    # Store feature names
    model_data['feature_names'] = feature_names
    model_data['user_profiles'] = user_profiles
    
    # Get features for clustering
    feature_cols = [col for col in user_profiles.columns 
                   if col not in ['user_id', 'display_name', 'cluster_assignment']]
    X = user_profiles[feature_cols].values
    
    # Clustering
    clustering = KMeansClustering()
    cluster_labels = clustering.fit_predict(X)
    user_profiles['cluster'] = cluster_labels
    
    # Store clustering model
    model_data['clustering_model'] = clustering
    
    # Get cluster statistics
    cluster_stats = clustering.get_cluster_statistics(X, feature_cols)
    
    # Similarity matching
    matcher = SimilarityMatcher(metric='cosine')
    matcher.fit(X, user_profiles['user_id'].tolist(), cluster_labels)
    model_data['similarity_matcher'] = matcher
    
    # Dimensionality reduction
    dim_reducer = DimensionalityReducer()
    pca_results = dim_reducer.apply_pca(X, n_components=3)
    tsne_results = dim_reducer.apply_tsne(X, perplexity_values=[30])
    umap_results = dim_reducer.apply_umap(X, n_neighbors_values=[15], min_dist_values=[0.1])
    model_data['dimensionality_reducer'] = dim_reducer
    
    # Prepare results
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    results = {
        'summary': {
            'total_users': int(len(user_profiles)),
            'n_clusters': int(len(unique_labels)),
            'cluster_sizes': {int(label): int(count) for label, count in zip(unique_labels, counts)}
        },
        'visualizations': {},
        'is_demo': is_demo
    }
    
    # Generate visualizations
    vis_dir = os.path.join('static', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create various plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. PCA analysis
    dim_reducer.plot_pca_analysis(
        feature_cols, cluster_labels,
        save_path=os.path.join(vis_dir, f'pca_analysis_{timestamp}.png')
    )
    results['visualizations']['pca_analysis'] = f'pca_analysis_{timestamp}.png'
    
    # 2. t-SNE visualization
    dim_reducer.plot_tsne_comparison(
        cluster_labels,
        save_path=os.path.join(vis_dir, f'tsne_comparison_{timestamp}.png')
    )
    results['visualizations']['tsne_comparison'] = f'tsne_comparison_{timestamp}.png'
    
    # 3. UMAP visualization
    dim_reducer.plot_umap_comparison(
        cluster_labels,
        save_path=os.path.join(vis_dir, f'umap_comparison_{timestamp}.png')
    )
    results['visualizations']['umap_comparison'] = f'umap_comparison_{timestamp}.png'
    
    # 4. Methods comparison
    dim_reducer.plot_all_methods_comparison(
        cluster_labels,
        save_path=os.path.join(vis_dir, f'methods_comparison_{timestamp}.png')
    )
    results['visualizations']['methods_comparison'] = f'methods_comparison_{timestamp}.png'
    
    # 5. Feature correlation heatmap
    heatmap_gen = HeatmapGenerator()
    heatmap_gen.create_feature_correlation_heatmap(
        user_profiles[feature_cols[:20]],  # Top 20 features
        save_path=os.path.join(vis_dir, f'feature_correlation_{timestamp}.png')
    )
    results['visualizations']['feature_correlation'] = f'feature_correlation_{timestamp}.png'
    
    # 6. Cluster feature heatmap
    heatmap_gen.create_cluster_feature_heatmap(
        cluster_stats, feature_cols[:15],
        save_path=os.path.join(vis_dir, f'cluster_features_{timestamp}.png')
    )
    results['visualizations']['cluster_features'] = f'cluster_features_{timestamp}.png'
    
    # 7. EDA visualizations
    eda_viz = EDAVisualizer()
    eda_viz.create_statistical_summary(
        user_profiles, feature_cols,
        save_path=os.path.join(vis_dir, f'statistical_summary_{timestamp}.png')
    )
    results['visualizations']['statistical_summary'] = f'statistical_summary_{timestamp}.png'
    
    # If specific user is targeted
    if target_user_id:
        user_idx = user_profiles[user_profiles['user_id'] == target_user_id].index[0]
        user_cluster = cluster_labels[user_idx]
        
        # Find similar users
        similar_users = matcher.find_similar_users_with_explanation(
            target_user_id, top_n=10, feature_names=feature_cols
        )
        
        # Find taste twins
        taste_twins = matcher.find_taste_twins(target_user_id, similarity_threshold=0.85)
        
        results['user_analysis'] = {
            'user_id': target_user_id,
            'cluster': int(user_cluster),
            'cluster_size': int(np.sum(cluster_labels == user_cluster)),
            'similar_users': similar_users,
            'taste_twins': taste_twins,
            'profile_summary': profile_builder.create_profile_summary(
                user_profiles.iloc[user_idx]
            )
        }

    # Convert numpy types to Python native types for JSON serialization
    results = convert_numpy_types(results)

    return results

@app.route('/api/clusters')
def get_clusters():
    """Get cluster information."""
    if model_data['user_profiles'] is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    profiles = model_data['user_profiles']
    clusters = profiles.groupby('cluster').agg({
        'user_id': 'count',
        'energy_mean': 'mean',
        'valence_mean': 'mean',
        'danceability_mean': 'mean'
    }).to_dict('index')
    
    return jsonify(clusters)

@app.route('/api/similar/<user_id>')
def find_similar(user_id):
    """Find similar users for a specific user."""
    if model_data['similarity_matcher'] is None:
        return jsonify({'error': 'No model loaded'}), 400
    
    try:
        similar_users = model_data['similarity_matcher'].find_similar_users(
            user_id, top_n=20
        )
        return jsonify(similar_users)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/visualizations/<viz_type>')
def get_visualization(viz_type):
    """Get specific visualization data."""
    if model_data['dimensionality_reducer'] is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    if viz_type == 'pca_3d':
        # Return 3D PCA data
        pca_data = model_data['dimensionality_reducer'].pca_results
        if pca_data and pca_data['n_components'] >= 3:
            data = {
                'coordinates': pca_data['transformed_data'][:, :3].tolist(),
                'labels': model_data['clustering_model'].labels.tolist(),
                'user_ids': model_data['user_profiles']['user_id'].tolist()
            }
            return jsonify(data)
    
    return jsonify({'error': 'Visualization type not found'}), 404

@app.route('/static/visualizations/<filename>')
def serve_visualization(filename):
    """Serve visualization images with proper headers."""
    try:
        viz_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'visualizations')
        return send_from_directory(viz_path, filename)
    except FileNotFoundError:
        return "Visualization not found", 404

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/visualizations', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Run app
    app.run(debug=True, port=8888)