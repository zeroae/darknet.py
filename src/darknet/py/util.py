import numpy as np
from PIL import Image


def image_to_3darray(image, target_shape):
    # We assume the original size matches the target_shape (height, width)
    orig_size = target_shape
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        orig_size = image.size
        image = image_scale_and_pad(image, target_shape)
        image = np.asarray(image)
    if isinstance(image, np.ndarray):
        if image.shape[0:2] != target_shape:
            image = Image.fromarray(image)
            orig_size = image.size
            image = image_scale_and_pad(image, target_shape)
            image = np.asarray(image)
        image = image.transpose((2, 0, 1)).astype(dtype=np.float32, order="C")/255
    return image, orig_size


def image_scale_and_pad(image: Image.Image, target_shape) -> Image.Image:
    image = image.convert("RGB")
    if (image.width, image.height) != target_shape:
        from PIL import ImageOps
        image = image.copy()
        image.thumbnail(target_shape)
        image = ImageOps.pad(image, target_shape)
    return image
