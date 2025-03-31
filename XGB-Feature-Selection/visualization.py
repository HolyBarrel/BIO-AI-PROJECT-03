from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
import numpy as np

paths = ["output/breast_cancer_wisconsin_original","output/titanic", "output/wine_quality_combined" ]

breast_cancer_df = pd.read_csv(paths[0])
titanic_df = pd.read_csv(paths[1])
wine_quality_df = pd.read_csv(paths[2])

def surrogate_loss_landscape(df, title):
    """
    Visualizes the loss landscape using a surrogate model (Gaussian Process Regression)
    in a 2D space obtained via PCA.

    Parameters:
    - df: DataFrame containing the feature columns and a 'Loss' column.
    - title: String used for titling and saving the plot.
    """
    # Reduce to 2 dimensions using PCA (drop the 'Loss' column)
    features = df.drop(columns=['Loss'])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]

    X = X_pca  # shape (n_samples, 2)
    y = df['Loss'].values

    # Define a kernel using Matern instead of RBF for more flexibility.
    # Adjust length_scale_bounds to allow smaller scales if needed.
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=1.5)
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X, y)

    # Create a grid over the PCA space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict loss values (and standard deviation if desired)
    y_pred, sigma = gp.predict(grid, return_std=True)
    y_pred = y_pred.reshape(xx.shape)

    # Plot the surrogate loss landscape as a contour plot
    plt.figure(figsize=(12, 8))
    contour = plt.contourf(xx, yy, y_pred, cmap='viridis', alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
    plt.colorbar(contour, label='Predicted Loss')
    plt.title("Surrogate Loss Landscape via Gaussian Process Regression (PCA space)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    os.makedirs("visualizations/surrogate", exist_ok=True)
    plt.savefig(f"visualizations/surrogate/{title}_surrogate_loss_landscape.png")
    plt.close()


def is_pareto_efficient(costs):
    """
    Find the Pareto-efficient points.

    Parameters:
    - costs: ndarray of shape (n_points, n_costs)

    Returns:
    - A boolean array indicating whether each point is Pareto efficient.
    """
    is_efficient = [True] * len(costs)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[i] = not any(
                (all(c2 <= c) and any(c2 < c)) for j, c2 in enumerate(costs) if j != i and is_efficient[j]
            )
    return is_efficient

def plot_feature_count_loss(df, title):
    """
    Plots number of selected features vs loss and highlights Pareto front.
    Additionally, plots a bar chart showing the number of elements with exactly one
    used feature and the total number of elements, with counts annotated above each bar.

    Parameters:
    - df: DataFrame with only binary feature selection columns + 'Loss' column.
    - title: String for saving and titling the plots.
    """
    if 'Loss' not in df.columns:
        raise ValueError("DataFrame must contain a 'Loss' column.")

    # Create a working copy and compute the number of selected features per row.
    df = df.copy()
    df['selected_features'] = df.drop(columns=['Loss']).sum(axis=1)

    # Compute Pareto front (assuming is_pareto_efficient is defined elsewhere)
    costs = df[['selected_features', 'Loss']].values
    pareto_mask = is_pareto_efficient(costs)
    df['pareto'] = pareto_mask

    # Plot the scatter plot: Number of Selected Features vs Loss
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='selected_features', y='Loss', hue='pareto', palette={True: 'red', False: 'blue'})
    plt.title('Number of Selected Features vs Loss')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Loss')

    # Save the scatter plot
    os.makedirs("visualizations/scatter", exist_ok=True)
    plt.savefig(f"visualizations/scatter/{title}_scatter_plot.png")
    plt.close()

    # Calculate the counts
    count_single_feature = (df['selected_features'] == 1).sum()
    total_elements = len(df)

    # Plot the bar chart with the two counts
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Only one used feature', 'Total elements'], [count_single_feature, total_elements], color=['green', 'gray'])
    plt.ylabel('Count')
    plt.title('Count of Elements: Only one used feature vs Total')

    # Annotate the bars with their count values
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height}', 
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    # Save the bar chart
    os.makedirs("visualizations/bar", exist_ok=True)
    plt.savefig(f"visualizations/bar/{title}_bar_plot.png")
    plt.close()



def plot_feature_count_loss_combined(df, title):
    """
    Plots number of selected features vs Loss.
    Instead of just marking Pareto front points, the dots are colored based on a composite
    metric computed from the normalized Loss and the normalized number of selected features.
    
    Parameters:
    - df: DataFrame with binary feature selection columns and a 'Loss' column.
    - title: String used for titling and saving the plot.
    """
    if 'Loss' not in df.columns:
        raise ValueError("DataFrame must contain a 'Loss' column.")

    df = df.copy()
    # Calculate the number of selected features (assumes all non-'Loss' columns are binary features)
    df['selected_features'] = df.drop(columns=['Loss']).sum(axis=1)

    # Normalize Loss and selected_features to [0, 1]
    df['loss_norm'] = (df['Loss'] - df['Loss'].min()) / (df['Loss'].max() - df['Loss'].min())
    df['features_norm'] = (df['selected_features'] - df['selected_features'].min()) / (
        df['selected_features'].max() - df['selected_features'].min())

    # Create a composite metric. Here, both metrics are weighted equally.
    df['composite'] = 0.5 * df['loss_norm'] + 0.5 * df['features_norm']

    plt.figure(figsize=(12, 8))
    sc = plt.scatter(
        df['selected_features'], df['Loss'],
        c=df['composite'], cmap='viridis', alpha=0.7
    )
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Loss')
    plt.title('Number of Selected Features vs Loss\nColored by Composite (Loss & Feature Count)')
    cbar = plt.colorbar(sc)
    cbar.set_label('Composite Metric (0=low, 1=high)')

    os.makedirs("visualizations/scatter/combined", exist_ok=True)
    plt.savefig(f"visualizations/scatter/combined/{title}_scatter_plot.png")
    plt.close()
   


def violin_plot(df, title):
    df = df.copy()
    df['selected_features'] = df.drop(columns=['Loss']).sum(axis=1)
    plt.figure(figsize=(12,8))
    sns.boxplot(x='selected_features', y='Loss', data=df)
    plt.title('Distribution of Loss by Number of Selected Features')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Loss')
    plt.savefig(f"visualizations/violin/{title}_violin_plot.png")

def matrix_plot(df, title):
    sns.pairplot(df, vars=df.columns[:-1])
    plt.savefig(f"visualizations/matrix/{title}_matrix_plot.png")

def pca_2d_plot(df, title):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[df.columns[:-1]])
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]

    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="Loss",
        palette=sns.color_palette("hsv", as_cmap=True),
        data=df,
        legend=False,  # disable the discrete legend
        alpha=0.3
    )
    # Create a ScalarMappable for the colorbar with same colormap and normalization
    sm = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(vmin=df["Loss"].min(), vmax=df["Loss"].max()))
    sm.set_array([])  # Required for the colorbar to work
    fig = plt.gcf()
    fig.colorbar(sm, ax=ax, label="Loss")
    
    plt.title("PCA Plot of Feature Combinations")
    os.makedirs("visualizations/pca2d", exist_ok=True)
    plt.savefig(f"visualizations/pca2d/{title}_pca_plot.png")
    plt.close()

def tsne_2d_plot(df, title):
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(df[df.columns[:-1]])
    df['tsne-one'] = tsne_result[:, 0]
    df['tsne-two'] = tsne_result[:, 1]

    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="Loss",
        palette=sns.color_palette("hsv", as_cmap=True),
        data=df,
        legend=False,
        alpha=0.3
    )
    sm = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(vmin=df["Loss"].min(), vmax=df["Loss"].max()))
    sm.set_array([])
    fig = plt.gcf()
    fig.colorbar(sm, ax=ax, label="Loss")
    
    plt.title("t-SNE Plot of Feature Combinations")
    os.makedirs("visualizations/tsne2d", exist_ok=True)
    plt.savefig(f"visualizations/tsne2d/{title}_tsne_plot.png")
    plt.close()

def pca_3d_plot(df, title):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[df.columns[:-1]])
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df['pca-one'], df['pca-two'], df['pca-three'], c=df['Loss'], cmap='hsv', alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Loss")
    
    ax.set_xlabel('PCA One')
    ax.set_ylabel('PCA Two')
    ax.set_zlabel('PCA Three')
    plt.title("PCA Plot of Feature Combinations")
    os.makedirs("visualizations/pca3d", exist_ok=True)
    plt.savefig(f"visualizations/pca3d/{title}_pca_plot.png")
    plt.close()

def tsne_3d_plot(df, title):
    tsne = TSNE(n_components=3)
    tsne_result = tsne.fit_transform(df[df.columns[:-1]])
    df['tsne-one'] = tsne_result[:, 0]
    df['tsne-two'] = tsne_result[:, 1]
    df['tsne-three'] = tsne_result[:, 2]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df['tsne-one'], df['tsne-two'], df['tsne-three'], c=df['Loss'], cmap='hsv', alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Loss")
    
    ax.set_xlabel('t-SNE One')
    ax.set_ylabel('t-SNE Two')
    ax.set_zlabel('t-SNE Three')
    plt.title("t-SNE Plot of Feature Combinations")
    os.makedirs("visualizations/tsne3d", exist_ok=True)
    plt.savefig(f"visualizations/tsne3d/{title}_tsne_plot.png")
    plt.close()



def main():
    
    plot_feature_count_loss(breast_cancer_df, "Breast Cancer Wisconsin Original")
    plot_feature_count_loss(titanic_df, "Titanic")
    plot_feature_count_loss(wine_quality_df, "Wine Quality Combined")

    plot_feature_count_loss_combined(breast_cancer_df, "Breast Cancer Wisconsin Original")
    plot_feature_count_loss_combined(titanic_df, "Titanic")
    plot_feature_count_loss_combined(wine_quality_df, "Wine Quality Combined")

"""
    violin_plot(breast_cancer_df, "Breast Cancer Wisconsin Original")
    violin_plot(titanic_df, "Titanic")
    violin_plot(wine_quality_df, "Wine Quality Combined")

    matrix_plot(breast_cancer_df, "Breast Cancer Wisconsin Original")
    matrix_plot(titanic_df, "Titanic")
    matrix_plot(wine_quality_df, "Wine Quality Combined")
    

    pca_2d_plot(breast_cancer_df, "Breast Cancer Wisconsin Original")
    pca_2d_plot(titanic_df, "Titanic")
    pca_2d_plot(wine_quality_df, "Wine Quality Combined")

    tsne_2d_plot(breast_cancer_df, "Breast Cancer Wisconsin Original")
    tsne_2d_plot(titanic_df, "Titanic")
    tsne_2d_plot(wine_quality_df, "Wine Quality Combined")

    pca_3d_plot(breast_cancer_df, "Breast Cancer Wisconsin Original")
    pca_3d_plot(titanic_df, "Titanic")
    pca_3d_plot(wine_quality_df, "Wine Quality Combined")

    tsne_3d_plot(breast_cancer_df, "Breast Cancer Wisconsin Original")
    tsne_3d_plot(titanic_df, "Titanic")
    tsne_3d_plot(wine_quality_df, "Wine Quality Combined")

    surrogate_loss_landscape(breast_cancer_df, "Breast Cancer Wisconsin Original")
    surrogate_loss_landscape(titanic_df, "Titanic")
    surrogate_loss_landscape(wine_quality_df, "Wine Quality Combined")
"""

if __name__ == "__main__":
    main()