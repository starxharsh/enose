import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans

# IEEE paper quality formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.grid': True,
    'grid.color': 'lightgray',
    'grid.linestyle': '--',
    'figure.autolayout': True,
    'savefig.dpi': 300,
    'figure.figsize': (7.16, 4.5)
})

def _save_fig(fig, save_dir, filename):
    """Helper to save figure in both PNG and PDF formats."""
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, f"{filename}.pdf"), bbox_inches='tight')

def plot_sensor_responses(data_dict, save_dir):
    """
    Plot raw sensor amplitude vs time for each gas combination.
    data_dict: dict of gas_combination -> pd.DataFrame (with columns for 6 sensors)
    """
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0), sharex=True, sharey=True)
    axes = axes.flatten()
    palette = sns.color_palette("tab10", 6)
    
    for i, (gas_comb, df) in enumerate(data_dict.items()):
        if i >= 8:
            break
        ax = axes[i]
        for j, col in enumerate(df.columns[:6]):
            ax.plot(df.index, df[col], label=col, color=palette[j], linewidth=0.8)
        ax.set_title(gas_comb)
        if i >= 4:
            ax.set_xlabel("Time/Samples")
        if i % 4 == 0:
            ax.set_ylabel("Amplitude")
            
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.1))
    _save_fig(fig, save_dir, "sensor_responses")
    plt.close(fig)

def plot_pca_2d(X, y_labels, save_dir):
    """PCA 2D scatter plot with explained variance ratio."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_labels, palette="tab10", ax=ax, s=15, alpha=0.8)
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_title("PCA 2D Projection")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    _save_fig(fig, save_dir, "pca_2d")
    plt.close(fig)

def plot_pca_3d(X, y_labels, save_dir):
    """PCA 3D scatter plot."""
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    var_ratio = pca.explained_variance_ratio_
    
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(y_labels)
    palette = sns.color_palette("tab10", len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = (y_labels == label)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], label=label, color=palette[i], s=10, alpha=0.8)
        
    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)")
    ax.set_title("PCA 3D Projection")
    ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
    _save_fig(fig, save_dir, "pca_3d")
    plt.close(fig)

def plot_tsne(X, y_labels, save_dir, perplexities=[5, 30, 50]):
    """t-SNE visualization with multiple perplexity values."""
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    
    for ax, perp in zip(axes, perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_tsne = tsne.fit_transform(X)
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_labels, palette="tab10", ax=ax, s=10, alpha=0.8, legend=False)
        ax.set_title(f"Perplexity: {perp}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        
    # Add legend to the last axis
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(np.unique(y_labels)), bbox_to_anchor=(0.5, 1.15))
    _save_fig(fig, save_dir, "tsne_perplexities")
    plt.close(fig)

def plot_lda(X, y_labels, save_dir):
    """LDA projection scatter plot."""
    lda = LinearDiscriminantAnalysis(n_components=2)
    try:
        X_lda = lda.fit_transform(X, y_labels)
        var_ratio = lda.explained_variance_ratio_
        
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=y_labels, palette="tab10", ax=ax, s=15, alpha=0.8)
        ax.set_xlabel(f"LD1 ({var_ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"LD2 ({var_ratio[1]*100:.1f}%)")
        ax.set_title("LDA 2D Projection")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        _save_fig(fig, save_dir, "lda_2d")
        plt.close(fig)
    except Exception as e:
        print(f"LDA failed (needs at least 3 classes for 2 components): {e}")

def plot_kmeans_clustering(X, y_labels, save_dir, k_values=[4, 8]):
    """K-Means clustering results vs true labels side by side."""
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_labels, palette="tab10", ax=axes[0], s=10, alpha=0.7)
        axes[0].set_title("True Labels")
        axes[0].legend(bbox_to_anchor=(0, -0.2), loc='upper left', ncol=2)
        
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="Set2", ax=axes[1], s=10, alpha=0.7)
        axes[1].set_title(f"K-Means (k={k})")
        axes[1].legend(bbox_to_anchor=(0, -0.2), loc='upper left', ncol=4)
        
        _save_fig(fig, save_dir, f"kmeans_k{k}")
        plt.close(fig)

def plot_correlation_heatmap(X, sensor_names, save_dir):
    """Pearson correlation heatmap between sensors."""
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    corr = np.corrcoef(X.T)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=sensor_names, yticklabels=sensor_names, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Sensor Correlation Heatmap")
    _save_fig(fig, save_dir, "correlation_heatmap")
    plt.close(fig)

def plot_class_distribution(y_labels, class_names, save_dir):
    """Bar plot showing sample count per gas combination."""
    unique, counts = np.unique(y_labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    sns.barplot(x=unique, y=counts, palette="tab10", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Gas Combination")
    ax.set_ylabel("Number of Samples")
    plt.xticks(rotation=45, ha='right')
    _save_fig(fig, save_dir, "class_distribution")
    plt.close(fig)

def plot_sensor_drift(data_by_date, save_dir):
    """Compare sensor responses across the 4 experiment dates to visualize drift."""
    # data_by_date: dict of date_str -> pd.DataFrame (with columns for 6 sensors)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.0), sharex=True)
    axes = axes.flatten()
    dates = list(data_by_date.keys())
    palette = sns.color_palette("tab10", len(dates))
    
    first_df = list(data_by_date.values())[0]
    sensor_cols = first_df.columns[:6]
    
    for i, col in enumerate(sensor_cols):
        ax = axes[i]
        for j, date in enumerate(dates):
            df = data_by_date[date]
            if col in df.columns:
                sns.kdeplot(df[col], ax=ax, label=date, color=palette[j], fill=True, alpha=0.3)
        ax.set_title(f"Sensor: {col}")
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(dates), bbox_to_anchor=(0.5, 1.05))
    _save_fig(fig, save_dir, "sensor_drift")
    plt.close(fig)

def plot_feature_distributions(X, feature_names, y_labels, save_dir):
    """Box plots or violin plots of top features across classes."""
    n_features = min(6, len(feature_names)) # Plot up to 6 features to avoid crowding
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.0))
    axes = axes.flatten()
    
    df = pd.DataFrame(X[:, :n_features], columns=feature_names[:n_features])
    df['Label'] = y_labels
    
    for i, col in enumerate(feature_names[:n_features]):
        ax = axes[i]
        sns.boxplot(x='Label', y=col, data=df, ax=ax, palette="tab10", showfliers=False)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    _save_fig(fig, save_dir, "feature_distributions")
    plt.close(fig)

def run_full_exploration(X, y, feature_names, sensor_names, save_dir, data_dict=None, data_by_date=None):
    """Run all analyses and save all plots."""
    print(f"Running full exploration, saving to {save_dir}...")
    
    # 1. PCA
    plot_pca_2d(X, y, save_dir)
    plot_pca_3d(X, y, save_dir)
    
    # 2. t-SNE
    plot_tsne(X, y, save_dir)
    
    # 3. LDA
    plot_lda(X, y, save_dir)
    
    # 4. K-Means
    plot_kmeans_clustering(X, y, save_dir)
    
    # 5. Correlation
    plot_correlation_heatmap(X[:, :len(sensor_names)], sensor_names, save_dir)
    
    # 6. Class Distribution
    plot_class_distribution(y, np.unique(y), save_dir)
    
    # 7. Feature Distributions
    plot_feature_distributions(X, feature_names, y, save_dir)
    
    # Optional plots depending on provided data
    if data_dict is not None:
        plot_sensor_responses(data_dict, save_dir)
        
    if data_by_date is not None:
        plot_sensor_drift(data_by_date, save_dir)
        
    print("Full exploration complete.")
