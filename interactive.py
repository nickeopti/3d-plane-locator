from typing import Any, Callable, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget
from scipy.spatial import transform


class InteractiveWindow(QtWidgets.QMainWindow):
    def __init__(self, data: np.ndarray, reference_image: QImage, slide: Optional[int] = None):
        super().__init__()

        self.setWindowTitle('CCFv3 location finetuner')

        layout = QtWidgets.QGridLayout()
        self.grid = layout
        row = -1

        layout.addWidget(Image(reference_image), (row := row + 1), 0, 1, 3)

        # Slide view
        volume_view = Volume(data)
        layout.addWidget(volume_view, (row := row + 1), 0, 1, 3)

        # Slide selection slider
        self._add_selector(
            title='Slide:',
            action=lambda v: volume_view.set_slide(v - 1),
            row=(row := row + 1),
            integer=True,
            range=(1, data.shape[0]),
            value=data.shape[0] // 2 if slide is None else slide
        )

        # Top-bottom angle selection slider
        self._add_selector(
            title='Dorsoventral angle:',
            action=volume_view.set_vertical_angle,
            row=(row := row + 1),
            integer=False,
            range=(-25, 25),
            value=0,
            scale=10,
            decimals=1,
            suffix='°'
        )

        # Left-right angle selection slider
        self._add_selector(
            title='Mediolateral angle:',
            action=volume_view.set_horisontal_angle,
            row=(row := row + 1),
            integer=False,
            range=(-8, 8),
            value=0,
            scale=10,
            decimals=1,
            suffix='°'
        )

        # Boiler plate
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
    
    def _add_selector(
        self,
        title: str,
        action: Callable[[Any], Any],
        row: int,
        integer: bool,
        range: tuple[int | float, int | float],
        value: int | float,
        scale: float = 1,
        decimals: int = 1,
        suffix: str = ''
    ):
        self.grid.addWidget(QtWidgets.QLabel(title), row, 0, Qt.AlignmentFlag.AlignRight)

        slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        if integer:
            spinner = QtWidgets.QSpinBox()

            slider.valueChanged.connect(spinner.setValue)
            spinner.valueChanged.connect(slider.setValue)
            spinner.valueChanged.connect(action)
        else:
            spinner = QtWidgets.QDoubleSpinBox()

            slider.valueChanged.connect(lambda v: spinner.setValue(v / 10))
            spinner.valueChanged.connect(lambda v: slider.setValue(v * 10))
            spinner.valueChanged.connect(action)

        slider.setRange(*(range if integer else [v * scale for v in range]))
        spinner.setRange(*range)
        spinner.setValue(value)
        if not integer:
            spinner.setDecimals(decimals)
        spinner.setSuffix(suffix)

        self.grid.addWidget(slider, row, 1)
        self.grid.addWidget(spinner, row, 2)


class Image(QWidget):
    def __init__(self, image: np.ndarray | QImage = None) -> None:
        super().__init__()
        
        self.image = image

    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, image):
        if isinstance(image, np.ndarray):
            self._image = QtGui.QImage(
                np.ascontiguousarray(image),
                image.shape[1], image.shape[0],
                QtGui.QImage.Format.Format_Grayscale16
            )
        elif isinstance(image, QImage):
            self._image = image
        elif image is None:
            self._image = None
        else:
            raise ValueError

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        if self.image is None:
            return

        scaled = self.image.scaled(
            self.width(), self.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        qp = QPainter(self)
        qp.drawImage((self.width() - scaled.width()) / 2, 0, scaled)
        self.resize(QtCore.QSize(self.width(), scaled.height()))


class Volume(Image):
    def __init__(self, data: np.ndarray, slide: int = None, vertical_angle: float = 0, horisontal_angle: float = 0) -> None:
        super().__init__()
        self.data = data
        self._slide = slide if slide is not None else data.shape[0] // 2
        self._vertical_angle = vertical_angle
        self._horisontal_angle = horisontal_angle

        dim = self.data.shape
        xx, yy = map(np.arange, dim[1:])
        self.g = np.stack(
            np.meshgrid(xx - dim[1] // 2, yy - dim[2] // 2),
            -1
        ).reshape(-1, 2)

    @property
    def slide(self):
        return self._slide
    
    @slide.setter
    def slide(self, slide):
        self._slide = slide
        self.repaint()

    def set_slide(self, slide):
        self.slide = slide
    
    @property
    def vertical_angle(self):
        return self._vertical_angle
    
    @vertical_angle.setter
    def vertical_angle(self, angle):
        self._vertical_angle = angle
        self.repaint()
    
    def set_vertical_angle(self, angle):
        self.vertical_angle = angle
    
    @property
    def horisontal_angle(self):
        return self._horisontal_angle
    
    @horisontal_angle.setter
    def horisontal_angle(self, angle):
        self._horisontal_angle = angle
        self.repaint()
    
    def set_horisontal_angle(self, angle):
        self.horisontal_angle = angle

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        r = transform.Rotation.from_euler('zy', [self.vertical_angle, self.horisontal_angle], degrees=True)
        R = r.as_matrix()

        dim = self.data.shape

        z = np.zeros((dim[1] * dim[2], 1))
        g = np.hstack([z, self.g])

        p = (R @ g.T).T.round().astype(int)

        slide = self.data[(
            np.clip(p[:, 0] + self.slide, 0, dim[0]-1),
            np.clip(p[:, 1] + dim[1] // 2, 0, dim[1]-1),
            np.clip(p[:, 2] + dim[2] // 2, 0, dim[2]-1)
        )].reshape(dim[1], dim[2], order='F')

        self.image = slide

        super().paintEvent(event)
