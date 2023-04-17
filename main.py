from argparse import ArgumentParser

import nrrdu
import numpy as np
import skimage
import skimage.io
import torch
import torchvision.transforms
from PySide6 import QtWidgets

from interactive import InteractiveWindow
from model import BinRegressor
from row_viewer import MainWindow


def load_image(path: str):
    image = skimage.io.imread(path, as_gray=True)
    image = skimage.img_as_ubyte(image)

    image = torchvision.transforms.ToTensor()(image)
    image = torchvision.transforms.Resize((512, 512))(image)
    image = image.expand(3, *image.shape[1:])

    return image.unsqueeze(0)


def load_model(path: str):
    model = BinRegressor.load_from_checkpoint(path, range=1320, bins=64, bin_overlap=0.7)
    return model


def predict(model: BinRegressor, image, k: int, size: int):
    def bound(prediction: float) -> int:
        rounded = int(round(prediction))
        if rounded < 0:
            return 0
        if rounded >= size:
            return size - 1
        return rounded

    with torch.no_grad():
        prediction = model(image)

        confidences, regression_prediction = prediction

        _, top_indices = torch.topk(confidences[0], k)

    return [bound(regression_prediction[0, index].item()) for index in top_indices]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--atlas', type=str)
    parser.add_argument('--intensity_factor', type=int, default=1)
    args = parser.parse_args()

    model = load_model(args.model)
    atlas = nrrdu.read(args.atlas).astype(np.uint16)
    atlas *= args.intensity_factor

    def predictor(image_path, k):
        image = load_image(image_path)
        predictions = predict(model, image, k, atlas.shape[0])

        return predictions

    def open_interactive_viewer(keeper, reference_image, index: int):
        keeper.interactive_window = InteractiveWindow(atlas, reference_image, slide=index)
        keeper.interactive_window.resize(600, 800)
        keeper.interactive_window.show()

    app = QtWidgets.QApplication()

    window = MainWindow(predictor=predictor, atlas=atlas, on_click=open_interactive_viewer)
    window.resize(1500, 600)
    window.show()

    app.exec()
