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

def plot_feature_count_loss(df, title):
    df['selected_features'] = df[df.columns[:-1]].sum(axis=1)


    # Create a scatter plot: x-axis is the number of selected features, y-axis is the Loss value.
    plt.figure(figsize=(12, 8))
    plt.scatter(df['selected_features'], df['Loss'], s=100, color='tab:blue', edgecolor='k')
    plt.title("Loss vs. Number of Selected Features")
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"visualizations/fcl/{title}_loss_vs_feature_count.png")

def violin_plot(df, title):
    plt.figure(figsize=(12,8))
    sns.boxplot(x='selected_features', y='Loss', data=df)
    plt.title('Distribution of Loss by Number of Selected Features')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Loss')
    plt.savefig(f"visualizations/violin/{title}_violin_plot.png")

def parallel_coordinates_plot(df, title):
    df['loss_bin'] = pd.qcut(df['Loss'], q=4, labels=False)
    plt.figure(figsize=(12,8))
    parallel_coordinates(df.drop('Loss', axis=1), 'loss_bin', colormap=plt.get_cmap("Set2"))
    plt.title('Parallel Coordinates Plot of Feature Combinations')
    plt.xlabel('Features')
    plt.ylabel('Feature Value')
    plt.savefig(f"visualizations/parallel/{title}_parallel_coordinates.png")

def heatmap_plot(df, title):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[df.columns[:-2]], cmap="YlGnBu", cbar=False, annot=True, fmt=".0f")
    plt.title("Heatmap of Feature Combinations")
    plt.xlabel("Features")
    plt.ylabel("Solution Index")
    plt.savefig(f"visualizations/heat_map/{title}_heatmap.png")

def matrix_plot(df, title):
    sns.pairplot(df, vars=df.columns[:-1])
    plt.savefig(f"visualizations/matrix/{title}_matrix_plot.png")

def pca_2d_plot(df, title):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[df.columns[:-1]])
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="Loss",
        palette=sns.color_palette("hsv", as_cmap=True),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.title("PCA Plot of Feature Combinations")
    plt.savefig(f"visualizations/pca2d/{title}_pca_plot.png")

def tsne_2d_plot(df, title):
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(df[df.columns[:-1]])
    df['tsne-one'] = tsne_result[:, 0]
    df['tsne-two'] = tsne_result[:, 1]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="Loss",
        palette=sns.color_palette("hsv", as_cmap=True),
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.title("t-SNE Plot of Feature Combinations")
    plt.savefig(f"visualizations/tsne2d/{title}_tsne_plot.png")


def pca_3d_plot(df, title):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[df.columns[:-1]])
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['pca-one'], df['pca-two'], df['pca-three'], c=df['Loss'], cmap='hsv', alpha=0.3)
    ax.set_xlabel('PCA One')
    ax.set_ylabel('PCA Two')
    ax.set_zlabel('PCA Three')
    plt.title("PCA Plot of Feature Combinations")
    plt.savefig(f"visualizations/pca3d/{title}_pca_plot.png")

def tsne_3d_plot(df, title):
    tsne = TSNE(n_components=3)
    tsne_result = tsne.fit_transform(df[df.columns[:-1]])
    df['tsne-one'] = tsne_result[:, 0]
    df['tsne-two'] = tsne_result[:, 1]
    df['tsne-three'] = tsne_result[:, 2]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['tsne-one'], df['tsne-two'], df['tsne-three'], c=df['Loss'], cmap='hsv', alpha=0.3)
    ax.set_xlabel('t-SNE One')
    ax.set_ylabel('t-SNE Two')
    ax.set_zlabel('t-SNE Three')
    plt.title("t-SNE Plot of Feature Combinations")
    plt.savefig(f"visualizations/tsne3d/{title}_tsne_plot.png")


def main():
    
    plot_feature_count_loss(breast_cancer_df, "Breast Cancer Wisconsin Original")
    plot_feature_count_loss(titanic_df, "Titanic")
    plot_feature_count_loss(wine_quality_df, "Wine Quality Combined")

    violin_plot(breast_cancer_df, "Breast Cancer Wisconsin Original")
    violin_plot(titanic_df, "Titanic")
    violin_plot(wine_quality_df, "Wine Quality Combined")

    parallel_coordinates_plot(breast_cancer_df, "Breast Cancer Wisconsin Original")
    parallel_coordinates_plot(titanic_df, "Titanic")
    parallel_coordinates_plot(wine_quality_df, "Wine Quality Combined")


    heatmap_plot(breast_cancer_df, "Breast Cancer Wisconsin Original")
    heatmap_plot(titanic_df, "Titanic")
    heatmap_plot(wine_quality_df, "Wine Quality Combined")

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