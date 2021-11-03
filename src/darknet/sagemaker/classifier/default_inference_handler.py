import PIL.Image as Image

from typing import Tuple, List

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
        max_labels = 5
        min_confidence = 55

        if isinstance(data, Image.Image):
            image, _ = image_to_3darray(data, network.shape)
            probabilities = network.predict_image(image)
        else:
            probabilities = network.predict(data)

        rv = [{
            "Name": label,
            "Confidence": prob*100,
        } for label, prob in sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)]
        return {"Labels": rv[0:max_labels] if max_labels else rv}
