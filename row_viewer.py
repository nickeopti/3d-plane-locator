import functools
from typing import Callable

import numpy as np
import skimage
import skimage.io
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, predictor: Callable, atlas: np.ndarray, on_click: Callable):
        super().__init__()
        self.predictor = predictor
        self.atlas = atlas

        self.setWindowTitle('CCFv3 locator')

        load_file_button = QtWidgets.QPushButton('Select slice…')
        load_file_button.clicked.connect(self.load_file)
        load_file_button.setMaximumSize(load_file_button.sizeHint())
        self.setMenuWidget(load_file_button)

        self.res_view = ImageRow(on_click=functools.partial(on_click, keeper=self))
        self.setCentralWidget(self.res_view)

    def load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select slice')

        if not path:
            return

        predictions = self.predictor(path, k=5)

        image = skimage.io.imread(path, as_gray=True)
        image = skimage.img_as_ubyte(image)

        self.res_view.reference = QtGui.QImage(
            np.ascontiguousarray(image),
            image.shape[1], image.shape[0],
            QtGui.QImage.Format.Format_Grayscale8
        )
        self.res_view.set_row(
            [self.read_slice(index) for index in predictions],
            predictions,
            [str(round(prediction / 10)) for prediction in predictions]
        )

    def read_slice(self, index: int):
        slide = self.atlas[index]

        image = QtGui.QImage(
            np.ascontiguousarray(slide),
            slide.shape[1], slide.shape[0],
            QtGui.QImage.Format.Format_Grayscale16
        )

        return image


class ImageRow(QWidget):
    def __init__(
        self,
        reference: QtGui.QImage = None,
        images: list[QtGui.QImage] = None,
        labels: list[str] = None,
        margin: int = 10,
        on_click: Callable = None,
    ) -> None:
        super().__init__()
        self.reference = reference
        self.margin = margin
        self.images = None
        self.on_click = on_click

    def set_row(self, images, indexes, labels):
        if self.images is not None:
            for image in self.images:
                image.deleteLater()

        self.images = [
            Image(image, label, self.margin, functools.partial(self.on_click, reference_image=self.reference, index=index))
            for image, index, label in zip(images, indexes, labels)
        ]
        for image in self.images:
            image.setParent(self)

        self.repaint()

    @property
    def image_height(self) -> float:
        aspect = sum(i.image.width() / i.image.height() for i in self.images)
        height = (self.width() - self.margin * (len(self.images) + 1)) / aspect
        return height

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        if self.images is None:
            return

        qp = QPainter(self)

        height = self.image_height

        m = len(self.images) // 2
        x_offset = sum(img.image.width() / img.image.height() * height for img in self.images[:m]) + self.margin * (m + 1)
        width = self.images[m].image.width() / self.images[m].image.height() * height
        
        scaled_reference = self.reference.scaledToWidth(
            width, mode=Qt.TransformationMode.SmoothTransformation
        )

        qp.drawImage(x_offset, 0, scaled_reference)
        y_offset = scaled_reference.height() + self.margin

        for i, image in enumerate(self.images):
            x_offset = sum(img.image.width() / img.image.height() * height for img in self.images[:i]) + self.margin * (i + 1)
            
            image.image_height = height
            image.show()
            image.move(x_offset, y_offset)


class Image(QWidget):
    def __init__(self, image: QImage, label: str, margin: float = 10, on_click: Callable = None) -> None:
        super().__init__()

        self.image = image
        self.label = label

        self.image_height = None
        self.margin = margin

        self.on_click = on_click

        self.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.on_click is not None:
                self.on_click()

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        if self.image_height is None:
            return

        qp = QPainter(self)

        height = self.image_height
        scaled_image = self.image.scaledToHeight(
            height, mode=Qt.TransformationMode.SmoothTransformation
        )
        qp.drawImage(0, 0, scaled_image)

        fm = QtGui.QFontMetrics(qp.font())
        bb = fm.boundingRect(self.label)
        qp.drawText(
            QtCore.QRect(
                0,
                height + self.margin,
                scaled_image.width(),
                bb.height(),
            ),
            Qt.AlignmentFlag.AlignHCenter,
            self.label,
        )

        self.resize(QtCore.QSize(scaled_image.width(), height + self.margin + bb.height()))
