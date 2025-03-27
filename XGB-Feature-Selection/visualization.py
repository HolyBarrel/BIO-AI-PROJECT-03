from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

paths = ["output/breast_cancer_wisconsin_original","output/titanic", "output/wine_quality_combined" ]

breast_cancer_df = pd.read_csv(paths[0])
titanic_df = pd.read_csv(paths[1])
wine_quality_df = pd.read_csv(paths[2])

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

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

    Parameters:
    - df: DataFrame with only binary feature selection columns + 'Loss' column.
    - title: String for saving and titling the plot.
    """
    if 'Loss' not in df.columns:
        raise ValueError("DataFrame must contain a 'Loss' column.")

    df = df.copy()
    df['selected_features'] = df.drop(columns=['Loss']).sum(axis=1)

    costs = df[['selected_features', 'Loss']].values
    pareto_mask = is_pareto_efficient(costs)
    df['pareto'] = pareto_mask

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='selected_features', y='Loss', hue='pareto', palette={True: 'red', False: 'blue'})
    plt.title('Number of Selected Features vs Loss')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Loss')

    os.makedirs("visualizations/scatter", exist_ok=True)
    plt.savefig(f"visualizations/scatter/{title}_scatter_plot.png")
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




if __name__ == "__main__":
    main()