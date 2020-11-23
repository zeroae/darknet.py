from itertools import tee
import av

import fsspec
import pafy

from darknet.py.detector import StreamDetector
from darknet.py.util import image_draw_detections as draw_detections

darknet_gh_url = "github://AlexeyAB:darknet@master"
# Load the Coco labels/metadata
with fsspec.open(f"{darknet_gh_url}/data/coco.names", mode="rt") as f:
    labels = [line.rstrip() for line in f.readlines()]

d = StreamDetector(labels=labels,
                   config_url=f"{darknet_gh_url}/cfg/yolov4.cfg",
                   weights_url="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
                   batch_size=10)

url = pafy.new('https://www.youtube.com/watch?v=1EiC9bvVGnk').getbest().url
icntnr = av.open(url)
icntnr.streams.video[0].thread_type = "AUTO"

frames, detections = tee(icntnr.decode(video=0))
detections = d.detect(detections)

#%%
for frame, dets in zip(frames, detections):
    dets = [(l, c) for l, c, _ in dets]
    print(f"time={frame.time}, dts={frame.dts}, pts={frame.pts}: {[dets]}")

