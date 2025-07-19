import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.patches import Circle
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterVisualizer:
    def __init__(self):
        self.color_palette = sns.color_palette("husl", 10)
        
    def plot_cluster_distribution(self, cluster_labels: np.ndarray,
                                save_path: Optional[str] = None):
        """Plot distribution of users across clusters."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count users per cluster
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        # Bar plot
        ax1.bar(unique_labels, counts, color=self.color_palette[:len(unique_labels)])
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Number of Users')
        ax1.set_title('Users per Cluster')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            ax1.text(label, count + 0.5, str(count), ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(counts, labels=[f'Cluster {i}' for i in unique_labels],
               autopct='%1.1f%%', colors=self.color_palette[:len(unique_labels)])
        ax2.set_title('Cluster Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster distribution plot to {save_path}")
        
        plt.close()
    
    def plot_feature_importance_by_cluster(self, cluster_stats: Dict,
                                         top_n_features: int = 10,
                                         save_path: Optional[str] = None):
        """Plot top distinguishing features for each cluster."""
        n_clusters = len(cluster_stats)
        fig, axes = plt.subplots((n_clusters + 1) // 2, 2, figsize=(16, 4 * ((n_clusters + 1) // 2)))
        
        if n_clusters == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (cluster_name, stats) in enumerate(cluster_stats.items()):
            ax = axes[idx]
            
            # Get top features
            top_features = stats.get('top_features', [])[:top_n_features]
            
            if top_features:
                features = [f['feature'] for f in top_features]
                deviations = [f['deviation'] for f in top_features]
                
                # Create horizontal bar plot
                y_pos = np.arange(len(features))
                ax.barh(y_pos, deviations, color=self.color_palette[idx % len(self.color_palette)])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('Deviation from Global Mean')
                ax.set_title(f'{cluster_name} - Top Distinguishing Features')
                ax.grid(True, alpha=0.3, axis='x')
            else:
                ax.text(0.5, 0.5, 'No feature data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{cluster_name}')
        
        # Remove empty subplots
        for idx in range(n_clusters, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.close()
    
    def create_user_similarity_network(self, similarity_matrix: np.ndarray,
                                     user_ids: List[str],
                                     cluster_labels: np.ndarray,
                                     threshold: float = 0.7,
                                     max_edges: int = 500,
                                     save_path: Optional[str] = None):
        """Create network visualization of user similarities."""
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, user_id in enumerate(user_ids):
            G.add_node(user_id, cluster=int(cluster_labels[i]))
        
        # Add edges (only above threshold and limit total edges)
        edges = []
        for i in range(len(user_ids)):
            for j in range(i + 1, len(user_ids)):
                if similarity_matrix[i, j] > threshold:
                    edges.append((i, j, similarity_matrix[i, j]))
        
        # Sort by similarity and take top edges
        edges.sort(key=lambda x: x[2], reverse=True)
        edges = edges[:max_edges]
        
        for i, j, weight in edges:
            G.add_edge(user_ids[i], user_ids[j], weight=weight)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Draw nodes colored by cluster
        node_colors = [self.color_palette[G.nodes[node]['cluster'] % len(self.color_palette)] 
                      for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                             node_size=300, alpha=0.8)
        
        # Draw edges with varying thickness based on similarity
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, alpha=0.3, 
                             width=[w * 3 for w in weights])
        
        # Draw labels for nodes with many connections
        degree_dict = dict(G.degree())
        high_degree_nodes = {node: node for node, degree in degree_dict.items() 
                           if degree > np.percentile(list(degree_dict.values()), 75)}
        
        nx.draw_networkx_labels(G, pos, high_degree_nodes, font_size=8)
        
        plt.title(f'User Similarity Network (threshold={threshold})', fontsize=16)
        plt.axis('off')
        
        # Add legend for clusters
        unique_clusters = np.unique(cluster_labels)
        legend_elements = [plt.scatter([], [], c=self.color_palette[i % len(self.color_palette)], 
                                     s=100, label=f'Cluster {i}')
                         for i in unique_clusters]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved similarity network plot to {save_path}")
        
        plt.close()
    
    def create_audio_feature_radar_charts(self, cluster_profiles: pd.DataFrame,
                                        audio_features: List[str],
                                        save_path: Optional[str] = None):
        """Create radar charts for audio features by cluster."""
        n_clusters = len(cluster_profiles)
        
        # Create subplots
        cols = min(3, n_clusters)
        rows = (n_clusters + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows),
                               subplot_kw=dict(projection='polar'))
        
        if n_clusters == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Prepare radar chart data
        angles = np.linspace(0, 2 * np.pi, len(audio_features), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for idx, (_, cluster_data) in enumerate(cluster_profiles.iterrows()):
            ax = axes[idx]
            
            # Get feature values
            values = []
            for feature in audio_features:
                col_name = f'avg_{feature}_mean'
                if col_name in cluster_data:
                    values.append(cluster_data[col_name])
                else:
                    values.append(0)
            
            values += values[:1]  # Complete the circle
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, 
                   color=self.color_palette[idx % len(self.color_palette)])
            ax.fill(angles, values, alpha=0.25,
                   color=self.color_palette[idx % len(self.color_palette)])
            
            # Customize
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(audio_features)
            ax.set_ylim(0, 1)
            ax.set_title(f'Cluster {cluster_data["cluster_id"]}', size=14, pad=20)
            ax.grid(True)
        
        # Remove empty subplots
        for idx in range(n_clusters, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Audio Feature Profiles by Cluster', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved radar charts to {save_path}")
        
        plt.close()
    
    def plot_genre_distribution_comparison(self, user_profiles: pd.DataFrame,
                                         genre_columns: List[str],
                                         save_path: Optional[str] = None):
        """Compare genre distributions across clusters."""
        # Group by cluster
        cluster_genre_means = user_profiles.groupby('cluster')[genre_columns].mean()
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cluster_genre_means.T.plot(kind='bar', stacked=True, ax=ax,
                                  colormap='tab20')
        
        ax.set_xlabel('Genre')
        ax.set_ylabel('Average Preference Weight')
        ax.set_title('Genre Distribution by Cluster')
        ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved genre distribution comparison to {save_path}")
        
        plt.close()
    
    def create_interactive_cluster_explorer(self, reduced_features: np.ndarray,
                                          cluster_labels: np.ndarray,
                                          user_profiles: pd.DataFrame,
                                          method_name: str = "PCA",
                                          save_path: Optional[str] = None):
        """Create interactive cluster exploration plot."""
        # Prepare data
        df_plot = pd.DataFrame({
            'x': reduced_features[:, 0],
            'y': reduced_features[:, 1],
            'cluster': cluster_labels.astype(str),
            'user_id': user_profiles['user_id'].values
        })
        
        # Add some profile information for hover
        hover_features = ['energy_mean', 'valence_mean', 'danceability_mean']
        for feature in hover_features:
            if feature in user_profiles.columns:
                df_plot[feature] = user_profiles[feature].values
        
        # Create interactive scatter plot
        fig = px.scatter(df_plot, x='x', y='y', color='cluster',
                        hover_data=['user_id'] + [f for f in hover_features if f in df_plot.columns],
                        title=f'Interactive Cluster Explorer ({method_name})',
                        labels={'x': f'{method_name} 1', 'y': f'{method_name} 2'})
        
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(width=900, height=700)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive cluster explorer to {save_path}")
        
        return fig
    
    def plot_cluster_evolution(self, clustering_results: Dict[int, np.ndarray],
                             save_path: Optional[str] = None):
        """Plot how cluster assignments change with different K values."""
        k_values = sorted(clustering_results.keys())
        n_samples = len(next(iter(clustering_results.values())))
        
        # Create alluvial/flow diagram data
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create flow data
        for i in range(len(k_values) - 1):
            k1, k2 = k_values[i], k_values[i + 1]
            labels1 = clustering_results[k1]
            labels2 = clustering_results[k2]
            
            # Count transitions
            transitions = {}
            for j in range(n_samples):
                key = (labels1[j], labels2[j])
                transitions[key] = transitions.get(key, 0) + 1
            
            # Plot flows
            x1 = i
            x2 = i + 1
            
            for (c1, c2), count in transitions.items():
                y1 = c1 / (k1 - 1) if k1 > 1 else 0.5
                y2 = c2 / (k2 - 1) if k2 > 1 else 0.5
                
                ax.plot([x1, x2], [y1, y2], 
                       alpha=0.3, linewidth=count/n_samples * 20,
                       color=self.color_palette[c1 % len(self.color_palette)])
        
        # Add cluster markers
        for i, k in enumerate(k_values):
            for c in range(k):
                y = c / (k - 1) if k > 1 else 0.5
                ax.scatter(i, y, s=200, c=self.color_palette[c % len(self.color_palette)],
                         edgecolors='black', linewidth=2, zorder=10)
                ax.text(i, y, str(c), ha='center', va='center', fontweight='bold')
        
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([f'K={k}' for k in k_values])
        ax.set_ylabel('Cluster Index (normalized)')
        ax.set_title('Cluster Assignment Evolution', fontsize=16)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster evolution plot to {save_path}")
        
        plt.close()
    
    def create_cluster_summary_dashboard(self, cluster_stats: Dict,
                                       user_profiles: pd.DataFrame,
                                       save_path: Optional[str] = None):
        """Create comprehensive dashboard summarizing all clusters."""
        n_clusters = len(cluster_stats)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Cluster sizes (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        clusters = []
        sizes = []
        for cluster_name, stats in cluster_stats.items():
            clusters.append(cluster_name.replace('cluster_', 'C'))
            sizes.append(stats['size'])
        
        ax1.bar(clusters, sizes, color=self.color_palette[:len(clusters)])
        ax1.set_title('Cluster Sizes', fontsize=14)
        ax1.set_ylabel('Number of Users')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Average energy vs valence (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        energy_vals = []
        valence_vals = []
        for cluster_name, stats in cluster_stats.items():
            if 'feature_means' in stats:
                energy_vals.append(stats['feature_means'].get('energy_mean', 0))
                valence_vals.append(stats['feature_means'].get('valence_mean', 0))
        
        scatter = ax2.scatter(energy_vals, valence_vals, 
                            s=[s*10 for s in sizes],
                            c=range(len(clusters)),
                            cmap='viridis',
                            alpha=0.7)
        
        for i, cluster in enumerate(clusters):
            ax2.annotate(cluster, (energy_vals[i], valence_vals[i]), 
                        ha='center', va='center')
        
        ax2.set_xlabel('Average Energy')
        ax2.set_ylabel('Average Valence')
        ax2.set_title('Cluster Mood Map', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature diversity (bottom left)
        ax3 = fig.add_subplot(gs[1:, :2])
        
        # Calculate feature variance for each cluster
        feature_vars = []
        for i, (cluster_name, stats) in enumerate(cluster_stats.items()):
            if 'feature_stds' in stats:
                vars = list(stats['feature_stds'].values())
                feature_vars.append(np.mean(vars))
            else:
                feature_vars.append(0)
        
        ax3.barh(clusters, feature_vars, color=self.color_palette[:len(clusters)])
        ax3.set_xlabel('Average Feature Standard Deviation')
        ax3.set_title('Feature Diversity by Cluster', fontsize=14)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Cluster characteristics table (bottom right)
        ax4 = fig.add_subplot(gs[1:, 2:])
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        headers = ['Cluster', 'Size', 'Top Feature', 'Diversity']
        
        for i, (cluster_name, stats) in enumerate(cluster_stats.items()):
            row = [
                f"C{i}",
                str(stats['size']),
                stats['top_features'][0]['feature'] if stats.get('top_features') else 'N/A',
                f"{feature_vars[i]:.3f}" if i < len(feature_vars) else 'N/A'
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         colLabels=headers,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(self.color_palette[(i-1) % len(self.color_palette)] + (0.3,))
        
        ax4.set_title('Cluster Summary Statistics', fontsize=14, pad=20)
        
        plt.suptitle('Music Taste Clusters Dashboard', fontsize=18)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster summary dashboard to {save_path}")
        
        plt.close()