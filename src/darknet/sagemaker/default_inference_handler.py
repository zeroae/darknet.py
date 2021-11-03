import io

import numpy as np
import PIL.Image as Image

from abc import ABC
from glob import glob
from typing import Tuple, List, Union

from sagemaker_inference.encoder import encode
from sagemaker_inference.decoder import decode
from sagemaker_inference.default_inference_handler import DefaultInferenceHandler

from darknet.py.network import Network


class DefaultDarknetInferenceHandler(DefaultInferenceHandler, ABC):
    def default_model_fn(self, model_dir) -> Tuple[Network, List[str]]:
        """
        Loads a model.
        For PyTorch, a default function to load a model cannot be provided.
        Returns: A DarkNet Detector.
        """
        labels_file = glob(f"{model_dir}/*.labels")[0]
        with open(labels_file) as f:
            labels = [line.rstrip() for line in f.readlines()]

        cfg_file = glob(f"{model_dir}/*.cfg")[0]
        weights_file = glob(f"{model_dir}/*.weights")[0]

        return Network(cfg_file, weights_file, batch_size=1), labels

    def default_input_fn(self, input_data, content_type) -> Union[Image.Image, np.array]:
        """A default input_fn that can handle PIL Image Types

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type

        Returns: a PIL Image ready for ImageClassifier
        """
        if content_type.startswith("image/"):
            image = Image.open(io.BytesIO(input_data))
            return {"Image": image}
        else:
            return {"NDArray": decode(input_data, content_type)}

    def default_output_fn(self, prediction, accept):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized (Return Sagemaker format?)
        """
        return encode(prediction, accept)
