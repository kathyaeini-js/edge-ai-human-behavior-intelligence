import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/input.mp4")
parser.add_argument("--output", default="data/outputs/annotated.mp4")
args = parser.parse_args()

input_path = Path(args.input)
output_path = Path(args.output)

if not input_path.exists():
    raise FileNotFoundError(f"Video not found: {input_path}")

output_path.parent.mkdir(parents=True, exist_ok=True)

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(str(input_path))

if not cap.isOpened():
    raise RuntimeError("Could not open the input video.")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create the output video.")

frame_count = 0

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False
        )[0]

        annotated = result.plot()
        writer.write(annotated)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

finally:
    cap.release()
    writer.release()

print(f"Finished! Processed {frame_count} frames.")
print(f"Output video: {output_path.resolve()}")
