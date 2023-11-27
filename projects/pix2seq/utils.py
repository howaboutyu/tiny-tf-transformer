import matplotlib.pyplot as plt
import numpy as np


def visualize_detections(
    image,
    boxes,
    classes,
    scores,
    figsize=(7, 7),
    linewidth=1,
    color=[0, 0, 1],
    save_path=None,
    show=False,
):
    """
    Visualize Detection function taken from: https://keras.io/examples/vision/retinanet/
    """
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        box = np.squeeze(box)
        _cls = np.squeeze(_cls)
        score = np.squeeze(score)

        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    return ax
