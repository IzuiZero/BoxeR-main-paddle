import os
import paddle
from PIL import Image


class ImageReader:
    def __init__(self, base_path, reader_type):
        self.base_path = base_path
        self.reader_type = reader_type
        self.image_reader = None

    def _init_reader(self):
        if self.reader_type == "pil":
            self.image_reader = PILReader()
        else:
            raise TypeError("unknown image reader type")

    def read(self, image_path):
        image_path = os.path.join(self.base_path, image_path)

        if self.image_reader is None:
            self._init_reader()

        return self.image_reader.read(image_path)


class PILReader:
    def read(self, image_path):
        image_data = Image.open(image_path).convert("RGB")  # Image object

        return image_data