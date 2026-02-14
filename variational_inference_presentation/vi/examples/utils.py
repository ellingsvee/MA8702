import os
import matplotlib.pyplot as plt


def plot_data(x, y, beta: float | None = None, save_path: os.PathLike | None = None):
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")

    if beta is not None:
        plt.plot(x, beta * x, color="red")

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
