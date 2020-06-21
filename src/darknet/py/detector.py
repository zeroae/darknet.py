from . import Network
from .util import image_to_3darray


class ImageDetector(object):
    network: Network = None
    labels = None

    _last_image_size = None

    def __init__(self, labels, config_file, weights_file, **kwargs):
        self.network = Network(config_file, weights_file, **kwargs)
        self.labels = labels

    def detect(self, image, **kwargs):
        self.load_image(image)
        return self.get_detections(**kwargs)

    def load_image(self, image):
        image, self._last_image_size = image_to_3darray(image, self.network.shape)
        self.network.predict_image(image)

    def get_detections(self, **kwargs):
        if "frame_size" not in kwargs:
            kwargs["frame_size"] = self._last_image_size
        detections = self.network.detect(**kwargs)
        return detections if self.labels is None else [
            (self.labels[label_idx], prob, bbox)
            for label_idx, prob, bbox in detections
        ]


