import numpy as np
from PIL import Image
from PIL import ImageDraw


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


def image_draw_detections(img: Image.Image, detections) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    colors = ["purple", "blue", "green", "pink", "brown"]

    def xywh_to_bounds(x, y, w, h):
        return x-w/2, y-h/2, x+w/2, y+h/2

    for i, (cat, prob, xywh) in enumerate(detections):
        text = f"{cat}@{prob:.2%}"
        bounds = xywh_to_bounds(*xywh)
        t_w, t_h = draw.textsize(text)
        draw.rectangle(xywh_to_bounds(*xywh), outline=colors[i%5], width=4)
        draw.rectangle((bounds[0], bounds[1]-t_h, bounds[0]+t_w+4, bounds[1]), fill=colors[i%5])
        draw.text((bounds[0]+2,bounds[1]-t_h), text, fill="white")
    return img


