from flask import Flask, render_template, request, jsonify, session, send_from_directory, redirect
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import secrets

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import SpotifyCollector
from src.feature_engineering import AudioFeatureExtractor, UserProfileBuilder
from src.clustering import KMeansClustering
from src.similarity import SimilarityMatcher
from src.visualization import (
    DimensionalityReducer, HeatmapGenerator, 
    ClusterVisualizer, EDAVisualizer
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
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

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

@app.route('/current-user')
def current_user():
    """Get information about the currently authenticated user."""
    try:
        collector = SpotifyCollector()
        user_info = collector.get_user_info()
        return jsonify({
            'authenticated_user': user_info['id'],
            'display_name': user_info.get('display_name', 'No display name'),
            'followers': user_info.get('followers', {}).get('total', 0),
            'country': user_info.get('country', 'Unknown'),
            'product': user_info.get('product', 'Unknown')
        })
    except Exception as e:
        return jsonify({'error': f'Not authenticated: {str(e)}'}), 401

@app.route('/auth/login')
def login():
    """Initiate Spotify OAuth login."""
    try:
        from src.data_collection.spotify_auth import SpotifyAuth
        auth = SpotifyAuth()

        # Generate state for security
        state = secrets.token_urlsafe(16)
        session['oauth_state'] = state

        # Create authorization URL using the auth_manager
        auth_url = auth.auth_manager.get_authorize_url(state=state)
        return redirect(auth_url)

    except Exception as e:
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/callback')
def callback():
    """Handle Spotify OAuth callback."""
    try:
        from src.data_collection.spotify_auth import SpotifyAuth

        # Verify state parameter
        if request.args.get('state') != session.get('oauth_state'):
            return jsonify({'error': 'Invalid state parameter'}), 400

        # Get authorization code
        code = request.args.get('code')
        if not code:
            return jsonify({'error': 'No authorization code received'}), 400

        # Exchange code for token using auth_manager
        auth = SpotifyAuth()
        token_info = auth.auth_manager.get_access_token(code)

        # Store token in session
        session['token_info'] = token_info

        # Redirect back to main page
        return redirect('/')

    except Exception as e:
        return jsonify({'error': f'Callback failed: {str(e)}'}), 500

@app.route('/auth/logout')
def logout():
    """Logout and clear session."""
    session.clear()
    return redirect('/')

@app.route('/auth/clear')
def clear_auth():
    """Clear stored Spotify authentication to force re-authentication."""
    try:
        from src.data_collection.spotify_auth import SpotifyAuth
        auth = SpotifyAuth()

        if auth.clear_authentication():
            session.clear()
            return jsonify({
                'success': True,
                'message': 'Authentication cleared. Next login will require re-authentication.'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to clear authentication.'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error clearing authentication: {str(e)}'
        }), 500

@app.route('/auth/switch-user')
def switch_user():
    """Clear current authentication and redirect to login for a different user."""
    try:
        from src.data_collection.spotify_auth import SpotifyAuth
        auth = SpotifyAuth()

        # Clear current authentication
        auth.clear_authentication()
        session.clear()

        # Redirect to login
        return redirect('/auth/login')

    except Exception as e:
        return jsonify({'error': f'Failed to switch user: {str(e)}'}), 500

@app.route('/collect-user-data', methods=['POST'])
def collect_user_data():
    """Collect and save data for the currently authenticated user."""
    try:
        collector = SpotifyCollector()

        # Get current user info
        user_info = collector.get_user_info()
        user_id = user_info['id']

        # Collect comprehensive user data
        user_data = collector.collect_user_data(user_id)

        # Save to a multi-user storage file
        storage_file = os.path.join('data', 'multi_user_storage.json')
        os.makedirs(os.path.dirname(storage_file), exist_ok=True)

        # Load existing data
        if os.path.exists(storage_file):
            with open(storage_file, 'r') as f:
                all_users_data = json.load(f)
        else:
            all_users_data = {}

        # Add/update this user's data
        all_users_data[user_id] = {
            'data': user_data,
            'collected_at': datetime.now().isoformat(),
            'display_name': user_info.get('display_name', user_id)
        }

        # Save updated data
        with open(storage_file, 'w') as f:
            json.dump(all_users_data, f, indent=2, default=convert_numpy_types)

        return jsonify({
            'success': True,
            'user_id': user_id,
            'display_name': user_info.get('display_name', user_id),
            'message': f'Data collected successfully for {user_id}',
            'total_users_collected': len(all_users_data)
        })

    except Exception as e:
        return jsonify({'error': f'Data collection failed: {str(e)}'}), 500

@app.route('/list-collected-users')
def list_collected_users():
    """List all users who have had their data collected."""
    try:
        storage_file = os.path.join('data', 'multi_user_storage.json')

        if not os.path.exists(storage_file):
            return jsonify({'users': []})

        with open(storage_file, 'r') as f:
            all_users_data = json.load(f)

        users_list = []
        for user_id, user_info in all_users_data.items():
            users_list.append({
                'user_id': user_id,
                'display_name': user_info.get('display_name', user_id),
                'collected_at': user_info.get('collected_at'),
                'has_data': 'data' in user_info
            })

        return jsonify({'users': users_list})

    except Exception as e:
        return jsonify({'error': f'Failed to list users: {str(e)}'}), 500

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
    """Compare two users to check if they are music twins using stored data."""
    try:
        data = request.get_json()
        username1 = data.get('username1', '').strip()
        username2 = data.get('username2', '').strip()

        if not username1 or not username2:
            return jsonify({'error': 'Both usernames are required'}), 400

        if username1 == username2:
            return jsonify({'error': 'Please enter different usernames'}), 400

        # Load stored user data
        storage_file = os.path.join('data', 'multi_user_storage.json')

        if not os.path.exists(storage_file):
            return jsonify({
                'error': 'No user data found',
                'message': 'No users have collected their data yet. Please collect data for both users first.',
                'suggestion': 'Use the "Collect My Data" button after authenticating as each user.'
            }), 404

        with open(storage_file, 'r') as f:
            all_users_data = json.load(f)

        # Check if both users have data
        if username1 not in all_users_data:
            return jsonify({
                'error': f'No data found for {username1}',
                'message': f'User "{username1}" has not collected their data yet.',
                'suggestion': f'Have "{username1}" authenticate and collect their data first.',
                'available_users': list(all_users_data.keys())
            }), 404

        if username2 not in all_users_data:
            return jsonify({
                'error': f'No data found for {username2}',
                'message': f'User "{username2}" has not collected their data yet.',
                'suggestion': f'Have "{username2}" authenticate and collect their data first.',
                'available_users': list(all_users_data.keys())
            }), 404

        # Get user data
        user1_data = all_users_data[username1]['data']
        user2_data = all_users_data[username2]['data']
        
        # Process the two users
        users_data = [user1_data, user2_data]
        comparison_results = process_twin_comparison(users_data, username1, username2)
        
        return jsonify(comparison_results)
        
    except Exception as e:
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500

def process_twin_comparison(users_data, username1, username2):
    """Process two users and return detailed comparison."""
    # Feature engineering
    profile_builder = UserProfileBuilder()
    user_profiles, feature_names = profile_builder.build_profiles(users_data)
    
    # Get features for analysis
    feature_cols = [col for col in user_profiles.columns 
                   if col not in ['user_id', 'display_name', 'cluster_assignment']]
    X = user_profiles[feature_cols].values
    
    # Get user indices
    user1_idx = user_profiles[user_profiles['user_id'] == username1].index[0]
    user2_idx = user_profiles[user_profiles['user_id'] == username2].index[0]
    
    # Calculate direct similarity
    user1_features = X[user1_idx].reshape(1, -1)
    user2_features = X[user2_idx].reshape(1, -1)
    
    # Multiple similarity metrics
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from scipy.spatial.distance import correlation
    
    cosine_sim = float(cosine_similarity(user1_features, user2_features)[0][0])
    euclidean_dist = float(euclidean_distances(user1_features, user2_features)[0][0])
    correlation_sim = 1 - correlation(user1_features[0], user2_features[0])
    
    # Overall compatibility score (weighted average)
    compatibility_score = (cosine_sim * 0.5 + correlation_sim * 0.3 + (1/(1+euclidean_dist)) * 0.2)
    
    # Determine if they are twins (threshold: 0.8)
    are_twins = compatibility_score >= 0.8
    twin_level = get_twin_level(compatibility_score)
    
    # Feature-wise comparison
    feature_comparison = {}
    user1_profile = user_profiles.iloc[user1_idx]
    user2_profile = user_profiles.iloc[user2_idx]
    
    # Compare key audio features
    key_features = ['danceability_mean', 'energy_mean', 'valence_mean', 'acousticness_mean', 
                   'instrumentalness_mean', 'tempo_mean', 'loudness_mean']
    
    for feature in key_features:
        if feature in user1_profile and feature in user2_profile:
            diff = abs(user1_profile[feature] - user2_profile[feature])
            feature_comparison[feature] = {
                'user1_value': float(user1_profile[feature]),
                'user2_value': float(user2_profile[feature]),
                'difference': float(diff),
                'similarity': float(1 - min(diff, 1))  # Normalize to 0-1
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
            'compatibility_score': float(compatibility_score),
            'twin_level': twin_level,
            'cosine_similarity': float(cosine_sim),
            'correlation_similarity': float(correlation_sim),
            'euclidean_similarity': float(1/(1+euclidean_dist))
        },
        'feature_comparison': feature_comparison,
        'shared_traits': shared_traits,
        'recommendations': generate_twin_recommendations(are_twins, compatibility_score, shared_traits),
        'timestamp': datetime.now().isoformat()
    }
    
    return convert_numpy_types(results)

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

def get_feature_description(feature, value):
    """Get human-readable description of feature values."""
    descriptions = {
        'danceability_mean': f"Both love {'highly danceable' if value > 0.7 else 'moderately danceable' if value > 0.4 else 'less danceable'} music",
        'energy_mean': f"Both prefer {'high-energy' if value > 0.7 else 'moderate-energy' if value > 0.4 else 'low-energy'} tracks",
        'valence_mean': f"Both enjoy {'upbeat and positive' if value > 0.7 else 'moderately positive' if value > 0.4 else 'melancholic'} music",
        'acousticness_mean': f"Both like {'acoustic' if value > 0.7 else 'semi-acoustic' if value > 0.4 else 'electronic'} sounds",
        'instrumentalness_mean': f"Both prefer {'instrumental' if value > 0.7 else 'mixed' if value > 0.4 else 'vocal'} music",
        'tempo_mean': f"Both enjoy {'fast-paced' if value > 120 else 'moderate-paced' if value > 90 else 'slow-paced'} music",
        'loudness_mean': f"Both like {'loud' if value > -10 else 'moderate volume' if value > -20 else 'quiet'} music"
    }
    return descriptions.get(feature, f"Similar {feature} preferences")

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