import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DimensionalityReducer:
    def __init__(self):
        self.pca = None
        self.tsne = None
        self.umap = None
        self.pca_results = {}
        self.tsne_results = {}
        self.umap_results = {}
        
    def apply_pca(self, X: np.ndarray, n_components: int = 3) -> Dict:
        """Apply PCA and return results with analysis."""
        logger.info(f"Applying PCA with {n_components} components...")
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        self.pca_results = {
            'transformed_data': X_pca,
            'explained_variance': self.pca.explained_variance_,
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(self.pca.explained_variance_ratio_),
            'components': self.pca.components_,
            'n_components': n_components
        }
        
        logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        
        return self.pca_results
    
    def apply_tsne(self, X: np.ndarray, n_components: int = 2, 
                   perplexity_values: List[int] = [5, 30, 50]) -> Dict:
        """Apply t-SNE with multiple perplexity values."""
        logger.info(f"Applying t-SNE with perplexity values: {perplexity_values}")
        
        self.tsne_results = {}
        
        for perplexity in perplexity_values:
            logger.info(f"Running t-SNE with perplexity={perplexity}")
            
            tsne = TSNE(n_components=n_components,
                       perplexity=perplexity,
                       random_state=42,
                       max_iter=1000)
            
            X_tsne = tsne.fit_transform(X)
            
            self.tsne_results[f'perplexity_{perplexity}'] = {
                'transformed_data': X_tsne,
                'kl_divergence': tsne.kl_divergence_,
                'n_iter': tsne.n_iter_,
                'perplexity': perplexity
            }
        
        return self.tsne_results
    
    def apply_umap(self, X: np.ndarray, n_components: int = 2,
                   n_neighbors_values: List[int] = [5, 15, 30],
                   min_dist_values: List[float] = [0.1, 0.25, 0.5]) -> Dict:
        """Apply UMAP with different parameter combinations."""
        logger.info("Applying UMAP with various parameters...")
        
        self.umap_results = {}
        
        for n_neighbors in n_neighbors_values:
            for min_dist in min_dist_values:
                param_key = f'n{n_neighbors}_d{min_dist}'
                logger.info(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}")
                
                reducer = umap.UMAP(n_components=n_components,
                                   n_neighbors=n_neighbors,
                                   min_dist=min_dist,
                                   random_state=42)
                
                X_umap = reducer.fit_transform(X)
                
                self.umap_results[param_key] = {
                    'transformed_data': X_umap,
                    'n_neighbors': n_neighbors,
                    'min_dist': min_dist
                }
        
        return self.umap_results
    
    def plot_pca_analysis(self, feature_names: List[str], 
                         cluster_labels: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None):
        """Create comprehensive PCA visualization."""
        if not self.pca_results:
            raise ValueError("PCA has not been applied yet")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 2D PCA scatter plot
        ax1 = axes[0, 0]
        X_pca = self.pca_results['transformed_data']
        
        if cluster_labels is not None:
            scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.6)
            ax1.legend(*scatter.legend_elements(), title="Clusters")
        else:
            ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
        
        ax1.set_xlabel(f'PC1 ({self.pca_results["explained_variance_ratio"][0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({self.pca_results["explained_variance_ratio"][1]:.2%} variance)')
        ax1.set_title('PCA 2D Projection')
        ax1.grid(True, alpha=0.3)
        
        # 2. Explained variance ratio
        ax2 = axes[0, 1]
        components = range(1, len(self.pca_results['explained_variance_ratio']) + 1)
        ax2.bar(components, self.pca_results['explained_variance_ratio'])
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.set_title('Explained Variance by Component')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative explained variance
        ax3 = axes[1, 0]
        ax3.plot(components, self.pca_results['cumulative_variance_ratio'], 'bo-')
        ax3.axhline(y=0.8, color='r', linestyle='--', label='80% variance')
        ax3.axhline(y=0.9, color='g', linestyle='--', label='90% variance')
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Cumulative Explained Variance')
        ax3.set_title('Cumulative Explained Variance (Scree Plot)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature loadings (biplot)
        ax4 = axes[1, 1]
        loadings = self.pca_results['components'][:2].T
        
        # Plot data points
        if cluster_labels is not None:
            scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.3)
        else:
            ax4.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3)
        
        # Plot feature vectors
        if len(feature_names) <= 20:  # Only show if reasonable number of features
            scale = np.max(np.abs(X_pca[:, :2])) / np.max(np.abs(loadings))
            for i, (feature, loading) in enumerate(zip(feature_names, loadings)):
                ax4.arrow(0, 0, loading[0]*scale*0.8, loading[1]*scale*0.8,
                         head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
                ax4.text(loading[0]*scale*0.85, loading[1]*scale*0.85, feature,
                        fontsize=8, ha='center', va='center')
        
        ax4.set_xlabel(f'PC1 ({self.pca_results["explained_variance_ratio"][0]:.2%} variance)')
        ax4.set_ylabel(f'PC2 ({self.pca_results["explained_variance_ratio"][1]:.2%} variance)')
        ax4.set_title('PCA Biplot with Feature Loadings')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PCA analysis plot to {save_path}")
        
        plt.close()
    
    def plot_tsne_comparison(self, cluster_labels: Optional[np.ndarray] = None,
                           user_ids: Optional[List[str]] = None,
                           save_path: Optional[str] = None):
        """Plot t-SNE results with different perplexity values."""
        if not self.tsne_results:
            raise ValueError("t-SNE has not been applied yet")
        
        n_plots = len(self.tsne_results)
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        
        if n_plots == 1:
            axes = [axes]
        
        for idx, (key, result) in enumerate(self.tsne_results.items()):
            ax = axes[idx]
            X_tsne = result['transformed_data']
            
            if cluster_labels is not None:
                scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                                   c=cluster_labels, cmap='viridis', alpha=0.6)
                ax.legend(*scatter.legend_elements(), title="Clusters", loc='best')
            else:
                ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
            
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title(f't-SNE (perplexity={result["perplexity"]})\nKL divergence: {result["kl_divergence"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved t-SNE comparison plot to {save_path}")
        
        plt.close()
    
    def plot_umap_comparison(self, cluster_labels: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None):
        """Plot UMAP results with different parameters."""
        if not self.umap_results:
            raise ValueError("UMAP has not been applied yet")
        
        # Group by n_neighbors
        n_neighbors_values = sorted(set(r['n_neighbors'] for r in self.umap_results.values()))
        min_dist_values = sorted(set(r['min_dist'] for r in self.umap_results.values()))
        
        fig, axes = plt.subplots(len(n_neighbors_values), len(min_dist_values),
                                figsize=(5*len(min_dist_values), 5*len(n_neighbors_values)))

        # Handle single subplot case
        if len(n_neighbors_values) == 1 and len(min_dist_values) == 1:
            axes = np.array([[axes]])
        elif len(n_neighbors_values) == 1:
            axes = axes.reshape(1, -1)
        elif len(min_dist_values) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, n_neighbors in enumerate(n_neighbors_values):
            for j, min_dist in enumerate(min_dist_values):
                ax = axes[i, j]
                key = f'n{n_neighbors}_d{min_dist}'
                
                if key in self.umap_results:
                    X_umap = self.umap_results[key]['transformed_data']
                    
                    if cluster_labels is not None:
                        scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1],
                                           c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
                        if i == 0 and j == len(min_dist_values) - 1:
                            ax.legend(*scatter.legend_elements(), title="Clusters", 
                                    bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        ax.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6, s=30)
                    
                    ax.set_title(f'n_neighbors={n_neighbors}, min_dist={min_dist}')
                    ax.grid(True, alpha=0.3)
                    
                    if i == len(n_neighbors_values) - 1:
                        ax.set_xlabel('UMAP 1')
                    if j == 0:
                        ax.set_ylabel('UMAP 2')
        
        plt.suptitle('UMAP Parameter Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved UMAP comparison plot to {save_path}")
        
        plt.close()
    
    def create_interactive_3d_plot(self, method: str = 'pca',
                                 cluster_labels: Optional[np.ndarray] = None,
                                 user_ids: Optional[List[str]] = None,
                                 save_path: Optional[str] = None):
        """Create interactive 3D visualization using Plotly."""
        if method == 'pca' and self.pca_results and self.pca_results['n_components'] >= 3:
            X_reduced = self.pca_results['transformed_data'][:, :3]
            title = '3D PCA Visualization'
            axis_labels = [f'PC{i+1}' for i in range(3)]
        else:
            raise ValueError(f"3D data not available for method: {method}")
        
        # Prepare data
        df_plot = pd.DataFrame(X_reduced, columns=axis_labels)
        
        if cluster_labels is not None:
            df_plot['Cluster'] = cluster_labels.astype(str)
            color_col = 'Cluster'
        else:
            color_col = None
        
        if user_ids:
            df_plot['User'] = user_ids
            hover_data = ['User']
        else:
            hover_data = None
        
        # Create 3D scatter plot
        fig = px.scatter_3d(df_plot, 
                          x=axis_labels[0], 
                          y=axis_labels[1], 
                          z=axis_labels[2],
                          color=color_col,
                          hover_data=hover_data,
                          title=title,
                          opacity=0.7)
        
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(
            scene=dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2]
            ),
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive 3D plot to {save_path}")
        
        return fig
    
    def plot_all_methods_comparison(self, cluster_labels: Optional[np.ndarray] = None,
                                  save_path: Optional[str] = None):
        """Create side-by-side comparison of PCA, t-SNE, and UMAP."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # PCA
        if self.pca_results:
            X_pca = self.pca_results['transformed_data'][:, :2]
            ax = axes[0]
            if cluster_labels is not None:
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                                   c=cluster_labels, cmap='viridis', alpha=0.6)
            else:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_title('PCA')
            ax.grid(True, alpha=0.3)
        
        # t-SNE (best perplexity)
        if self.tsne_results:
            # Use perplexity 30 as default
            key = 'perplexity_30' if 'perplexity_30' in self.tsne_results else list(self.tsne_results.keys())[0]
            X_tsne = self.tsne_results[key]['transformed_data']
            ax = axes[1]
            if cluster_labels is not None:
                scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                                   c=cluster_labels, cmap='viridis', alpha=0.6)
            else:
                ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(f't-SNE (perplexity={self.tsne_results[key]["perplexity"]})')
            ax.grid(True, alpha=0.3)
        
        # UMAP (default parameters)
        if self.umap_results:
            # Use n15_d0.1 as default
            key = 'n15_d0.1' if 'n15_d0.1' in self.umap_results else list(self.umap_results.keys())[0]
            X_umap = self.umap_results[key]['transformed_data']
            ax = axes[2]
            if cluster_labels is not None:
                scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1],
                                   c=cluster_labels, cmap='viridis', alpha=0.6)
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Cluster')
            else:
                ax.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6)
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(f'UMAP (n_neighbors={self.umap_results[key]["n_neighbors"]}, '
                        f'min_dist={self.umap_results[key]["min_dist"]})')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Dimensionality Reduction Methods Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved methods comparison plot to {save_path}")
        
        plt.close()
    
    def create_animated_transitions(self, methods: List[str] = ['pca', 'tsne', 'umap'],
                                  cluster_labels: Optional[np.ndarray] = None,
                                  save_path: Optional[str] = None):
        """Create animated transitions between different dimensionality reductions."""
        import plotly.graph_objects as go
        
        frames = []
        
        # Collect data for each method
        data_dict = {}
        if 'pca' in methods and self.pca_results:
            data_dict['PCA'] = self.pca_results['transformed_data'][:, :2]
        if 'tsne' in methods and self.tsne_results:
            key = 'perplexity_30' if 'perplexity_30' in self.tsne_results else list(self.tsne_results.keys())[0]
            data_dict['t-SNE'] = self.tsne_results[key]['transformed_data']
        if 'umap' in methods and self.umap_results:
            key = 'n15_d0.1' if 'n15_d0.1' in self.umap_results else list(self.umap_results.keys())[0]
            data_dict['UMAP'] = self.umap_results[key]['transformed_data']
        
        if not data_dict:
            logger.warning("No dimensionality reduction results available for animation")
            return None
        
        # Create frames
        for method_name, data in data_dict.items():
            frame_data = []
            
            if cluster_labels is not None:
                for cluster in np.unique(cluster_labels):
                    mask = cluster_labels == cluster
                    frame_data.append(
                        go.Scatter(
                            x=data[mask, 0],
                            y=data[mask, 1],
                            mode='markers',
                            name=f'Cluster {cluster}',
                            marker=dict(size=8)
                        )
                    )
            else:
                frame_data.append(
                    go.Scatter(
                        x=data[:, 0],
                        y=data[:, 1],
                        mode='markers',
                        marker=dict(size=8)
                    )
                )
            
            frames.append(go.Frame(data=frame_data, name=method_name))
        
        # Create figure with first frame
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        
        # Add slider
        fig.update_layout(
            title="Dimensionality Reduction Methods Animation",
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play',
                     'method': 'animate',
                     'args': [None, {'frame': {'duration': 1000, 'redraw': True},
                                   'fromcurrent': True,
                                   'transition': {'duration': 500}}]},
                    {'label': 'Pause',
                     'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0, 'redraw': True},
                                     'mode': 'immediate',
                                     'transition': {'duration': 0}}]}
                ]
            }],
            sliders=[{
                'active': 0,
                'steps': [
                    {'args': [[f.name], {'frame': {'duration': 0, 'redraw': True},
                                       'mode': 'immediate',
                                       'transition': {'duration': 0}}],
                     'label': f.name,
                     'method': 'animate'}
                    for f in frames
                ]
            }]
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved animated transitions to {save_path}")
        
        return fig