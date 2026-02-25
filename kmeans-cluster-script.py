#!/usr/bin/env python3
"""
Generate the last two charts from kmeans-clustering-demo.ipynb
and save them as a single PNG.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend for saving files

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


def main(out_path: str = "last_two_charts.png") -> None:
    # --- Load data (matches the notebook) ---
    iris_dataset = load_iris()
    X_raw = iris_dataset.data
    y_raw = iris_dataset.target

    iris = pd.DataFrame(X_raw, columns=iris_dataset.feature_names)
    iris["target"] = y_raw
    iris["species"] = iris["target"].map(
        {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    )

    # Feature matrix
    x = iris.iloc[:, 0:4]

    # --- Fit KMeans (matches the notebook) ---
    kmeans = KMeans(
        n_clusters=3,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=0,
    )
    y_kmeans = kmeans.fit_predict(x)

    X = x.to_numpy()

    # --- Create one figure containing both charts ---
    fig = plt.figure(figsize=(16, 7))

    # 2D cluster plot (features 0 & 1)
    ax1 = fig.add_subplot(1, 2, 1)

    species_names_2d = {
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica",
    }

    for i in range(3):
        ax1.scatter(
            X[y_kmeans == i, 0],
            X[y_kmeans == i, 1],
            s=100,
            label=species_names_2d[i],
        )

    ax1.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200,
        label="Centroids",
    )

    ax1.set_xlabel(iris_dataset.feature_names[0])
    ax1.set_ylabel(iris_dataset.feature_names[1])
    ax1.set_title("K-Means Clusters (2D)")
    ax1.legend()

    # 3D scatter plot (features 2, 3, 0)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    species_names_3d = {
        0: "iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica",
    }

    colors = {
        0: "purple",
        1: "orange",
        2: "green",
    }

    for i in range(3):
        ax2.scatter(
            X[y_kmeans == i, 2],  # Petal Length
            X[y_kmeans == i, 3],  # Petal Width
            X[y_kmeans == i, 0],  # Sepal Length
            s=100,
            c=colors[i],
            label=species_names_3d[i],
        )

    ax2.scatter(
        kmeans.cluster_centers_[:, 2],
        kmeans.cluster_centers_[:, 3],
        kmeans.cluster_centers_[:, 0],
        s=300,
        c="red",
        label="Centroids",
    )

    ax2.set_xlabel("Petal Length (cm)", labelpad=10)
    ax2.set_ylabel("Petal Width (cm)", labelpad=10)
    ax2.set_zlabel("Sepal Length (cm)", labelpad=0)
    ax2.set_title("K-Means Clusters (3D)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()