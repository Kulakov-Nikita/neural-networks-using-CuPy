import matplotlib.pyplot as plt
import cupy as cp


def plot_samples(images, title: str):
    imgs = cp.asnumpy(images.squeeze(1))
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    fig.suptitle(title)
    for idx, ax in enumerate(axes.flatten()):
        if idx >= len(imgs):
            ax.axis("off")
            continue
        ax.imshow(imgs[idx], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
