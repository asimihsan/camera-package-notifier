#!/usr/bin/env python

from dataclasses import dataclass
import pathlib
import sys
import tkinter as tk
from typing import Optional


@dataclass
class CameraEvent:
    root_path: pathlib.Path
    images_subpath: pathlib.Path
    images_annotated_flag_file: pathlib.Path


class CameraEventManager:
    root_path: pathlib.Path
    images_subpath: str

    def __init__(
        self, root_path: pathlib.Path, images_subpath: str = "images",
    ) -> None:
        self.root_path = root_path
        self.images_subpath = images_subpath

    def get_unannotated_event(self) -> Optional[CameraEvent]:
        child: pathlib.Path
        for child in self.root_path.iterdir():
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


class AnnotationApp(tk.Frame):
    master: tk.Misc
    camera_event_manager: CameraEventManager
    current_camera_event: Optional[CameraEvent]

    def __init__(
        self, master: tk.Misc, camera_event_manager: CameraEventManager
    ) -> None:
        super().__init__(master)
        self.master = master
        self.camera_event_manager = camera_event_manager
        self.pack()
        self.create_widgets()

    def create_widgets(self) -> None:
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit: tk.Button = tk.Button(
            self, text="QUIT", fg="red", command=self.master.destroy
        )
        self.quit.pack(side="bottom")

    def say_hi(self) -> None:
        print("hi there, everyone!")
        self.current_camera_event = self.camera_event_manager.get_unannotated_event()
        print(self.current_camera_event)


def main(root_path: pathlib.Path) -> None:
    camera_event_manager: CameraEventManager = CameraEventManager(
        root_path, images_subpath="images"
    )
    root: tk.Tk = tk.Tk()
    annotation_app: AnnotationApp = AnnotationApp(
        master=root, camera_event_manager=camera_event_manager
    )
    annotation_app.mainloop()


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
