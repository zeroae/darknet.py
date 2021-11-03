from typing import Tuple, List

from sagemaker_inference.errors import UnsupportedFormatError

from .. import DefaultDarknetInferenceHandler, image_to_3darray, Network


class DefaultDarknetClassifierInferenceHandler(DefaultDarknetInferenceHandler):
    def default_predict_fn(self, data, model: Tuple[Network, List[str]]):
        """A default predict_fn for DarkNet. Calls a model on data deserialized in input_fn.
        Args:
            data: input data (PIL.Image) for prediction deserialized by input_fn
            model: Darknet model loaded in memory by model_fn

        Returns: a prediction
        """
        network, labels = model
        max_labels = data.get("MaxLabels", 5)
        # TODO: min_confidence = data.get("MinConfidence", 55)

        if "NDArray" in data:
            probabilities = network.predict(data["NDArray"])
        elif "Image" in data:
            image, _ = image_to_3darray(data["Image"], network.shape)
            probabilities = network.predict_image(image)
        else:
            raise UnsupportedFormatError("Expected an NDArray or an Image")

        rv = [
            {
                "Name": label,
                "Confidence": prob * 100,
            }
            for label, prob in sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)
        ]
        return {"Labels": rv[0:max_labels] if max_labels else rv}
