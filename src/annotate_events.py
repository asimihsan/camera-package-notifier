#!/usr/bin/env python

from dataclasses import dataclass
from typing import Optional, List
import pathlib
import PIL.Image
import PySide2.QtCore
import PySide2.QtGui
import PySide2.QtWidgets
import sys
import random


@dataclass
class CameraEvent:
    root_path: pathlib.Path
    images_subpath: pathlib.Path
    images_annotated_flag_file: pathlib.Path

    def get_image_paths(self) -> List[pathlib.Path]:
        return sorted(filename for filename in self.images_subpath.iterdir())


class CameraEventManager:
    root_path: pathlib.Path
    images_subpath: str

    def __init__(
        self, root_path: pathlib.Path, images_subpath: str = "images",
    ) -> None:
        self.root_path = root_path
        self.images_subpath = images_subpath

    def get_unannotated_event(self) -> Optional[CameraEvent]:
        paths: pathlib.Path = [path for path in self.root_path.iterdir()]
        random.shuffle(paths)

        child: pathlib.Path
        for child in paths:
            if not child.is_dir():
                continue
            images_path: pathlib.Path = child / self.images_subpath
            if not images_path.is_dir():
                continue
            flag_filename: str = "%s_annotated" % (self.images_subpath,)
            images_annotated_flag_file: pathlib.Path = child / flag_filename
            if images_annotated_flag_file.is_file():
                continue
            return CameraEvent(
                root_path=child,
                images_subpath=images_path,
                images_annotated_flag_file=images_annotated_flag_file,
            )
        return None


class AnnotationApp(PySide2.QtWidgets.QDialog):
    camera_event_manager: CameraEventManager
    layout: Optional[PySide2.QtWidgets.QGridLayout]
    button: PySide2.QtWidgets.QPushButton

    def __init__(
        self,
        camera_event_manager: CameraEventManager,
        parent: Optional[PySide2.QtWidgets.QWidget] = None,
    ) -> None:
        super(AnnotationApp, self).__init__(parent)
        self.camera_event_manager = camera_event_manager
        self.setWindowTitle("Annotation App")
        self.layout = None
        self.button = None

        self.next_event()

    def next_event(self) -> None:
        if self.layout is not None:
            while self.layout.count():
                child = self.layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            self.layout = PySide2.QtWidgets.QGridLayout()
            self.setLayout(self.layout)
        if self.button is not None:
            self.button.deleteLater()
        self.button = PySide2.QtWidgets.QPushButton("I am a button")
        self.button.clicked.connect(self.next_event)
        self.layout.addWidget(self.button, 0, 0)

        event = self.camera_event_manager.get_unannotated_event()
        if event is None:
            print("no more events to annotate")
            sys.exit()

        image_paths = event.get_image_paths()
        images: List[PySide2.QtWidgets.QLabel] = []
        image_path: pathlib.Path
        for image_path in image_paths:
            label = PySide2.QtWidgets.QLabel()
            pixmap = PySide2.QtGui.QPixmap(str(image_path.absolute()))
            pixmap = pixmap.scaled(
                200,
                200,
                PySide2.QtCore.Qt.IgnoreAspectRatio,
                PySide2.QtCore.Qt.SmoothTransformation,
            )
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            images.append(label)

        for i, image in enumerate(images):
            self.layout.addWidget(image, 1, i)


def main(root_path: pathlib.Path) -> None:
    camera_event_manager: CameraEventManager = CameraEventManager(
        root_path, images_subpath="images"
    )
    app = PySide2.QtWidgets.QApplication()
    annotation_app = AnnotationApp(camera_event_manager)
    annotation_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
