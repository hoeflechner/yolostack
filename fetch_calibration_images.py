#!/usr/bin/env python3
"""Fill a calibration directory with full-frame frigate snapshots for TensorRT INT8 export.

TensorRT's INT8 quantization needs ~300 images to estimate activation ranges; ultralytics
falls back to coco8 (4 images) when the export gets no data= argument. Run this on the frigate
host and mount the result as CALIB_DIR (default /workspace/calib) in the yolostack container.

    python3 fetch_calibration_images.py --api http://localhost:8085 --limit 300

Images are never written into the repo: they stay on the frigate host (default ~/frigate-calib)
and are mounted read-only as CALIB_DIR. Delete the directory once the engine is built.
"""
import argparse
import json
import os
import urllib.request

MIN_BYTES = 5_000  # frigate snapshots are ~30 KB at detect resolution; smaller means a broken frame


def fetch(url, binary=False):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read() if binary else json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8085", help="frigate base url")
    parser.add_argument("--out", default=os.path.expanduser("~/frigate-calib/images"),
                        help="outside any git repo; the container mounts its parent as CALIB_DIR")
    parser.add_argument("--limit", type=int, default=300, help="how many images to collect")
    parser.add_argument("--camera", default=None, help="restrict to one camera name")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    query = f"/api/events?limit={args.limit * 2}&include_thumbnails=0"
    if args.camera:
        query += f"&cameras={args.camera}"
    events = fetch(args.api + query)

    got = 0
    for event in events:
        path = os.path.join(args.out, f"{event['camera']}-{event['id']}.jpg")
        if os.path.exists(path):
            got += 1
            continue
        try:
            data = fetch(f"{args.api}/api/events/{event['id']}/snapshot.jpg", binary=True)
        except Exception as error:  # a missing snapshot must not stop the run
            print(f"skip {event['id']}: {error}")
            continue
        if len(data) < MIN_BYTES:
            print(f"skip {event['id']}: only {len(data)} bytes")
            continue
        with open(path, "wb") as image:
            image.write(data)
        got += 1
        if got >= args.limit:
            break

    print(f"{got} images in {args.out} (from {len(events)} events; TensorRT recommends >=300)")


if __name__ == "__main__":
    main()
