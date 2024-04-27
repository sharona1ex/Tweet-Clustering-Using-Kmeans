from src.model.kmeans import KMeans
import pickle
import os
from src.helper.config import MODEL
import matplotlib.pyplot as plt


def plot_elbow_curve(errors, k_values):
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, errors, marker='o', linestyle='-')
    plt.title("Elbow Curve")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Error")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    models = dict()
    for model in MODEL:
        if os.path.isfile('models/{}.pkl'.format(model)):
            with open('models/{}.pkl'.format(model), "rb") as file:
                models[model] = pickle.load(file)

    k = [models[model].n_clusters for model in models]
    errors = [models[model].inertia_ for model in models]

    # Plot the elbow curve
    plot_elbow_curve(errors, k)