from abc import ABC, abstractmethod

import numpy as np

from . import Network
from .util import image_to_3darray


class ClassifierBase(ABC):
    network: Network
    labels: list

    def __init__(self, labels, config_file, weights_file, **kwargs):
        self.network = Network(config_file, weights_file, **kwargs)

        self.labels = range(self.network.output_size()) if labels is None else labels
        if len(self.labels) != self.network.output_size():
            raise TypeError("Number of labels does not match size of network output. "
                            f"{len(self.labels)} != {self.network.output_size()}")

    @abstractmethod
    def classify(self, input_obj, top: int = -1):
        pass

    def top_k(self, probabilities, top):
        rv = zip(self.labels, probabilities)
        if top > 0:
            rv = sorted(rv, key=lambda x: x[1], reverse=True)[0:top]
        else:
            rv = list(rv)
        return rv


class Classifier(ClassifierBase):
    def classify(self, input_ndarr: np.ndarray, top: int = -1):
        probabilities = self.network.predict(input)
        return self.top_k(probabilities, top)


class ImageClassifier(ClassifierBase):
    def classify(self, image, top: int = -1):
        image, _ = image_to_3darray(image, self.network.shape)
        probabilities = self.network.predict_image(image)
        return self.top_k(probabilities, top)
