import PIL.Image as Image

from typing import Tuple, List
from sagemaker_inference import content_types, decoder, encoder, errors

from .. import DefaultDarknetInferenceHandler, image_to_3darray


class DefaultDarknetClassifierInferenceHandler(DefaultDarknetInferenceHandler):

    def default_predict_fn(self, data, model: Tuple[Network, List[str]]):
        """A default predict_fn for DarkNet. Calls a model on data deserialized in input_fn.
        Args:
            data: input data (PIL.Image) for prediction deserialized by input_fn
            model: Darknet model loaded in memory by model_fn

        Returns: a prediction
        """
        network, labels = model
        if isinstance(data, Image.Image):
            image, _ = image_to_3darray(data, network.shape)
            probabilities = network.predict_image(image)
        else:
            probabilities = network.predict(data)

        return zip(labels, probabilities)

    def default_output_fn(self, prediction, accept):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized
        """
        prediction = sorted(prediction, key=lambda x: x[1], reverse=True)[0:5]
        return encoder.encode(list(prediction), accept)
