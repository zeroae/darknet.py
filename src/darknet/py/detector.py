from itertools import chain

import numpy as np
from darknet.py.network import Network
from more_itertools import grouper, flatten

from .network import Network
from .util import image_to_3darray


class DetectorBase(object):
    network: Network = None
    labels = None

    def __init__(self, labels, config_url, weights_url, **kwargs):
        self.network = Network.open(config_url, weights_url, **kwargs)
        self.labels = labels

    def _id_to_label(self, detections):
        return [(self.labels[label_idx], prob, bbox) for label_idx, prob, bbox in detections]


class ImageDetector(DetectorBase):
    _last_image_size = None

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
        return detections if self.labels is None else self._id_to_label(detections)


class StreamDetector(DetectorBase):
    """ StreamDetector run a Yolo based detector on a stream of PyAV Video Frames.

    On a Titan RTX we get about 39.89fps with a batch size of 10
    3.76 s ± 50.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
    def detect(self, frames,  **kwargs):
        if "frame_size" not in kwargs:
            frame = next(frames)
            kwargs["frame_size"] = (frame.width, frame.height)
            frames = chain([frame], frames)
        if "letterbox" not in kwargs:
            kwargs["letterbox"] = 0

        # Reformat the frames to be in RGB format and in the Darknet byteorder
        r_frames = (frame.reformat(width=self.network.width, height=self.network.height, format="rgb24")
                    for frame in frames)
        r_frames = (frame.to_ndarray().transpose((2, 0, 1)) for frame in r_frames)

        # Group the frames based on the network batch size, filter out "None"s
        g_frames = grouper(self.network.batch_size, r_frames, None)
        g_frames = (tuple(filter(lambda x: type(None) != type(x), batch)) for batch in g_frames)

        # Concatenate the frames and create a contiguous float32 array
        g_frames = (np.concatenate(batch, axis=0) for batch in g_frames)
        g_frames = (np.ascontiguousarray(batch.flat, dtype=np.float32) / 255.0 for batch in g_frames)

        # Run the network detection in batch
        g_detections = (
            self.network.detect_batch(batch, **kwargs)
            for batch in g_frames
        )

        # Flatten the batched detections
        detections = flatten(g_detections)
        detections = detections if self.labels is None else (self._id_to_label(dets) for dets in detections)

        return detections
