from itertools import groupby

from typing import Tuple, List
from sagemaker_inference.errors import UnsupportedFormatError

from .. import DefaultDarknetInferenceHandler, Network, image_to_3darray


class DefaultDarknetDetectorInferenceHandler(DefaultDarknetInferenceHandler):
    def default_predict_fn(self, data, model: Tuple[Network, List[str]]):
        """A default predict_fn for DarkNet. Calls a model on data deserialized in input_fn.
        Args:
            data: input data (PIL.Image) for prediction deserialized by input_fn
            model: Darknet model loaded in memory by model_fn

        Returns: detected labels
        """
        if "Image" not in data:
            raise UnsupportedFormatError("Detector model expects an Image.")

        network, labels = model

        max_labels = data.get("MaxLabels", None)
        min_confidence = data.get("MinConfidence", 55)

        image = data["Image"]
        image, frame_size = image_to_3darray(image, network.shape)
        _ = network.predict_image(image)

        detections = network.detect(
            frame_size=frame_size,
            threshold=min_confidence / 100.0,
            hierarchical_threshold=min_confidence / 100,
        )
        detections = sorted(detections, key=lambda x: x[0])

        def bbox_to_sm_map(x_0, y_0, w, h):
            return {"Width": w, "Height": h, "Left": x_0 - w / 2, "Top": y_0 + h / 2}

        rv = []
        for label_idx, instances in groupby(detections, key=lambda x: x[0]):
            instances = [
                {"Confidence": prob * 100, "BoundingBox": bbox_to_sm_map(*bbox)}
                for _, prob, bbox in sorted(instances, key=lambda x: x[1], reverse=True)
            ]
            detection = {
                "Name": labels[label_idx],
                "Confidence": instances[0]["Confidence"],
                "Instances": instances,
                "Parents": [],
            }
            detection["Confidence"] = detection["Instances"][0]["Confidence"]
            rv.append(detection)
        return {"Labels": rv[0:max_labels] if max_labels else rv}
