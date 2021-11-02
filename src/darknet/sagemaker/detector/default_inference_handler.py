import PIL.Image as Image

from typing import Tuple, List
from sagemaker_inference import encoder, errors

from .. import DefaultDarknetInferenceHandler, Network, image_to_3darray


class DefaultDarknetDetectorInferenceHandler(DefaultDarknetInferenceHandler):
    def default_predict_fn(self, data, model: Tuple[Network, List[str]]):
        """A default predict_fn for DarkNet. Calls a model on data deserialized in input_fn.
        Args:
            data: input data (PIL.Image) for prediction deserialized by input_fn
            model: Darknet model loaded in memory by model_fn

        Returns: detected labels
        """
        network, labels = model
        if not isinstance(data, Image.Image):
            raise TypeError("Input data is not an Image.")

        image, frame_size = image_to_3darray(data, network.shape)
        _ = network.predict_image(image)
        detections = network.detect(frame_size=frame_size)

        return (
            detections
            if labels is None
            else [(labels[label_idx], prob, bbox) for label_idx, prob, bbox in detections]
        )

    def default_output_fn(self, prediction, accept):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized (Return Sagemaker format?)
        """
        prediction = sorted(prediction, key=lambda x: x[1], reverse=True)[0:5]
        return encoder.encode(list(prediction), accept)
