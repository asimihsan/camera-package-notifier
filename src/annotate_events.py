#!/usr/bin/env python

from dataclasses import dataclass
from types import prepare_class
from typing import Optional, List, Dict, Any
import json
import pathlib
import PySide2.QtCore
import PySide2.QtGui
import PySide2.QtWidgets
import sys


@dataclass
class CameraEvent:
    root_path: pathlib.Path
    images_subpath: pathlib.Path
    images_annotated_flag_file: pathlib.Path

    def get_image_paths(self) -> List[pathlib.Path]:
        return sorted(filename for filename in self.images_subpath.iterdir())

    def annotate_package_present(self) -> None:
        print("annotation %s as package present" % (self.root_path,))
        data: Dict[str, Any] = {
            "images_subpath": self.images_subpath.stem,
            "package_present": True,
        }
        with self.images_annotated_flag_file.open("w") as f_out:
            json.dump(data, f_out)

    def annotate_package_not_present(self) -> None:
        print("annotation %s as package not present" % (self.root_path,))
        data: Dict[str, Any] = {
            "images_subpath": self.images_subpath.stem,
            "package_present": False,
        }
        with self.images_annotated_flag_file.open("w") as f_out:
            json.dump(data, f_out)


class CameraEventManager:
    root_path: pathlib.Path
    images_subpath: str

    def __init__(
        self, root_path: pathlib.Path, images_subpath: str = "images",
    ) -> None:
        self.root_path = root_path
        self.images_subpath = images_subpath

    def get_unannotated_event(self) -> Optional[CameraEvent]:
        paths: List[pathlib.Path] = [path for path in self.root_path.iterdir()]
        paths.sort()

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
                print("%s is already annotated, skipping..." % (child,))
                continue
            return CameraEvent(
                root_path=child,
                images_subpath=images_path,
                images_annotated_flag_file=images_annotated_flag_file,
            )
        return None


class AnnotationApp(PySide2.QtWidgets.QDialog):
    camera_event_manager: CameraEventManager
    current_event: Optional[CameraEvent]
    app: PySide2.QtWidgets.QApplication
    layout: Optional[PySide2.QtWidgets.QGridLayout]
    next_event_button: PySide2.QtWidgets.QPushButton
    annotate_package_present_button: PySide2.QtWidgets.QPushButton
    annotate_package_not_present_button: PySide2.QtWidgets.QPushButton
    quit_button: PySide2.QtWidgets.QPushButton
    label: PySide2.QtWidgets.QLabel

    def __init__(
        self,
        camera_event_manager: CameraEventManager,
        app: PySide2.QtWidgets.QApplication,
        parent: Optional[PySide2.QtWidgets.QWidget] = None,
    ) -> None:
        super(AnnotationApp, self).__init__(parent)
        self.camera_event_manager = camera_event_manager
        self.current_event = None
        self.app = app
        self.setWindowTitle("Annotation App")
        self.layout = None
        self.next_event_button = None
        self.annotate_package_present_button = None
        self.annotate_package_not_present_button = None
        self.quit_button = None
        self.label = None

        self.create_layout()
        self.create_buttons()
        self.next_event()
        self.create_event_label()

    def create_layout(self) -> None:
        if self.layout is not None:
            while self.layout.count():
                child = self.layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            self.layout = PySide2.QtWidgets.QGridLayout()
            self.setLayout(self.layout)

    def create_buttons(self) -> None:
        if self.layout is None:
            print("ERROR cannot add buttons, layout is not there")
            return

        if self.next_event_button is not None:
            self.next_event_button.deleteLater()
        self.next_event_button = PySide2.QtWidgets.QPushButton("Next event (skip)")
        self.next_event_button.clicked.connect(self.next_event)
        self.layout.addWidget(self.next_event_button, 0, 0)

        if self.annotate_package_present_button is not None:
            self.annotate_package_present_button.deleteLater()
        self.annotate_package_present_button = PySide2.QtWidgets.QPushButton(
            "Annotate - package present"
        )
        self.annotate_package_present_button.clicked.connect(
            self.annotate_package_present
        )
        self.layout.addWidget(self.annotate_package_present_button, 0, 1)

        if self.annotate_package_not_present_button is not None:
            self.annotate_package_not_present_button.deleteLater()
        self.annotate_package_not_present_button = PySide2.QtWidgets.QPushButton(
            "Annotate - package NOT present"
        )
        self.annotate_package_not_present_button.clicked.connect(
            self.annotate_package_not_present
        )
        self.layout.addWidget(self.annotate_package_not_present_button, 0, 2)

        if self.quit_button is not None:
            self.quit_button.deleteLater()
        self.quit_button = PySide2.QtWidgets.QPushButton("Quit")
        self.quit_button.clicked.connect(self.quit_app)
        self.layout.addWidget(self.quit_button, 0, 3)

    def create_event_label(self) -> None:
        if self.label is not None:
            self.label.deleteLater()
        self.label = PySide2.QtWidgets.QLabel(
            "Current event: %s" % (self.current_event.root_path,)
        )
        self.layout.addWidget(self.label, 0, 5)

    def quit_app(self) -> None:
        print("Exiting...")
        self.app.quit()

    def annotate_package_present(self) -> None:
        print("AnnotationApp annotate_package_present entry")
        if self.current_event is None:
            print("ERROR camera event is None")
            return
        self.current_event.annotate_package_present()
        self.next_event()

    def annotate_package_not_present(self) -> None:
        print("AnnotationApp annotate_package_not_present entry")
        if self.current_event is None:
            print("ERROR camera event is None")
            return
        self.current_event.annotate_package_not_present()
        self.next_event()

    def next_event(self) -> None:
        self.current_event = self.camera_event_manager.get_unannotated_event()
        if self.current_event is None:
            print("no more events to annotate")
            sys.exit()

        if self.label is not None:
            self.label.setText("Current event: %s" % (self.current_event.root_path,))

        image_paths = self.current_event.get_image_paths()
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
    annotation_app = AnnotationApp(camera_event_manager, app)
    annotation_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
