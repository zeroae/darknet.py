from copy import deepcopy
from itertools import tee
import av
import click
import fsspec
import json
import m3u8
import pafy
from tqdm import tqdm
import webvtt

from darknet.py.detector import StreamDetector


def dets_to_json(dets):
    return json.dumps(
        {
            "detections": [
                {"label": det[0], "confidence": int(det[1] * 100), "bbox": [int(c) for c in det[2]]}
                for det in dets
            ]
        }
    )


@click.command()
@click.argument("playlist-uri", default="https://www.youtube.com/watch?v=iLQNbwMWbGM")
@click.option("--batch-size", "-b", default=4, help="inference batch size (use power of 2)")
@click.option("--labels-uri", "-l", default="github://AlexeyAB:darknet@master/data/coco.names")
@click.option("--config-uri", "-c", default="github://AlexeyAB:darknet@master/cfg/yolov4.cfg")
@click.option("--weights-uri", "-w", default="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
def main(batch_size, playlist_uri, labels_uri, config_uri, weights_uri):
    """
    Creates WebVTT metadata for objects in frames.

    """
    darknet_gh_url = "github://AlexeyAB:darknet@master"
    # Load the Coco labels/metadata
    with fsspec.open(labels_uri, mode="rt") as f:
        labels = [line.rstrip() for line in f.readlines()]

    model = ".".join(weights_uri.split("/")[-1].split(".")[:-1])
    d = StreamDetector(
        labels=labels,
        config_url=config_uri,
        weights_url=weights_uri,
        batch_size=batch_size,
    )

    if playlist_uri.startswith("https://www.youtube.com/watch?v="):
        playlist_uri = pafy.new(playlist_uri).getbest().url

    playlist = m3u8.load(playlist_uri)

    seg_playlist = deepcopy(playlist)
    seg_playlist.segments.clear()
    seg_playlist.is_endlist = False

    vtt_playlist = deepcopy(playlist)
    vtt_playlist.segments.clear()
    vtt_playlist.is_endlist = False

    for idx, segment in enumerate(tqdm(playlist.segments, leave=True, desc=playlist_uri)):
        seg_playlist.add_segment(segment)
        vtt_segment = deepcopy(segment)
        vtt_segment.uri = f"{model}-{idx}.vtt"

        with av.open(segment.absolute_uri) as icntnr:
            istrm = icntnr.streams.video[0]
            istrm.thread_type = "AUTO"

            frames, detections = tee(icntnr.decode(video=0))
            detections = d.detect(detections)

            vtt = webvtt.WebVTT()
            for frame, dets in tqdm(zip(frames, detections), desc=vtt_segment.uri, leave=True):
                if len(dets) == 0:
                    continue
                caption = webvtt.Caption()
                caption.text = dets_to_json(dets)
                caption._start = frame.time
                caption._end = caption._start + 1 / istrm.framerate
                vtt.captions.append(caption)

            vtt.save(vtt_segment.uri)
            vtt_playlist.add_segment(vtt_segment)

            vtt_playlist.dump(f"{model}.m3u8")
            seg_playlist.dump(f"{model}-sync.m3u8")

    vtt_playlist.is_endlist = True
    vtt_playlist.dump(f"{model}.m3u8")

    seg_playlist.is_endlist = True
    seg_playlist.dump(f"{model}-sync.m3u8")


if __name__ == "__main__":
    main()
