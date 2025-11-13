import pickle

import matplotlib.pyplot as plt
import numpy as np

from classwise_tda import poset_landscapes
from classwise_tda import visualization as vis

DATA_DIR = "/nfs/home/lverho/classwise_tda/data/mnist/"
PLOT_DIR = "/nfs/home/lverho/classwise_tda/figures/"
with open(DATA_DIR + "pairwise_comparison.pkl", "rb") as file:
    results_dict = pickle.load(file)

plot_matrix = np.full((10, 10), 0.0)
for row in range(10):
    for col in range(row + 1, 10):
        print(f"Running for {row} vs {col}")
        plot_matrix[row, col] = results_dict["pairwise_mse"][
            (str(row), str(col))
        ].item()
        plot_matrix[col, row] = results_dict["pairwise_mse"][
            (str(row), str(col))
        ].item()

        print(f"MSE: {results_dict['pairwise_mse'][(str(row), str(col))].item()}")

        F, ax = plt.subplots(1, 2, figsize=(12, 7))
        gen_lscape_array = poset_landscapes.create_poset_landscape_array(
            results_dict["generalized_landscapes"][str(row), str(col)]
        )
        vis.plot_landscape(
            gen_lscape_array["filt_vals"].values,
            gen_lscape_array.sel({"union": f"{row} U {col}"}).values,
            ax=ax[0],
            legend=True,
        )
        ax[0].set_title("Class-aware landscape")
        ax[0].set_xlabel("Filtration value")
        vis.plot_landscape(
            gen_lscape_array["filt_vals"].values,
            results_dict["unclassified_landscapes"][(str(row), str(col))],
            ax=ax[1],
            legend=True,
        )
        ax[1].set_title("Class-naive landscape")
        ax[1].set_xlabel("Filtration value")
        F.savefig(
            PLOT_DIR + f"mnist_pairwise_landscapes/{row}_vs_{col}_landscapes.png",
            bbox_inches="tight",
        )
        plt.close()

F, ax = plt.subplots(1, 1)
ax.matshow(plot_matrix)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
F.savefig(PLOT_DIR + "mnist_pairwise_mse.png")
