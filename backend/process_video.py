
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from backend.behavior import classify_posture


SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def extract_pose(pose_model, crop, device):
    """Estimate COCO-17 body keypoints from a person crop."""

    results = pose_model(
        crop,
        device=device,
        verbose=False
    )

    if not results or results[0].keypoints is None:
        return None, None

    keypoints_obj = results[0].keypoints

    if keypoints_obj.xy is None or len(keypoints_obj.xy) == 0:
        return None, None

    keypoints = keypoints_obj.xy[0].cpu().numpy().tolist()

    if keypoints_obj.conf is not None:
        scores = keypoints_obj.conf[0].cpu().numpy().tolist()
    else:
        scores = [1.0] * len(keypoints)

    return keypoints, scores


def draw_pose(frame, keypoints, scores, x1, y1, threshold=0.25):

    points = []

    for point, score in zip(keypoints, scores):

        if score < threshold:
            points.append(None)
            continue

        x = int(point[0]) + x1
        y = int(point[1]) + y1

        points.append((x, y))

    for a, b in SKELETON:

        if (
            a < len(points)
            and b < len(points)
            and points[a] is not None
            and points[b] is not None
        ):
            cv2.line(
                frame,
                points[a],
                points[b],
                (40, 220, 80),
                2
            )

    for point in points:

        if point is not None:
            cv2.circle(
                frame,
                point,
                3,
                (0, 180, 255),
                -1
            )


def main():

    parser = argparse.ArgumentParser(
        description="Edge AI Human Behavior Intelligence"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="data/outputs/annotated.mp4"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-people", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events_path = output_path.with_name("events.jsonl")

    print("Loading YOLO11 person detector...")

    detector = YOLO("yolo11n.pt")

    print("Loading YOLO11 pose estimation model...")

    pose_model = YOLO("yolo11n-pose.pt")

    print("AI models loaded successfully.")

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video input: {args.input}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps / max(1, args.stride)),
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video.")

    frame_index = 0
    processed = 0
    started = time.perf_counter()

    try:

        with events_path.open("w", encoding="utf-8") as log:

            while True:

                ok, frame = cap.read()

                if not ok:
                    break

                frame_index += 1

                if (
                    args.stride > 1
                    and (frame_index - 1) % args.stride != 0
                ):
                    continue

                results = detector.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=0.30,
                    verbose=False,
                    device=args.device
                )

                result = results[0]
                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:

                    xyxy = boxes.xyxy.cpu().numpy().astype(int)

                    ids = boxes.id

                    track_ids = (
                        ids.int().cpu().tolist()
                        if ids is not None
                        else list(range(len(xyxy)))
                    )

                    confs = (
                        boxes.conf.cpu().tolist()
                        if boxes.conf is not None
                        else [0.0] * len(xyxy)
                    )

                    order = sorted(
                        range(len(xyxy)),
                        key=lambda i: confs[i],
                        reverse=True
                    )[:args.max_people]

                    for i in order:

                        x1, y1, x2, y2 = xyxy[i]

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        crop = frame[y1:y2, x1:x2]

                        if crop.size == 0:
                            continue

                        try:

                            keypoints, scores = extract_pose(
                                pose_model,
                                crop,
                                args.device
                            )

                        except Exception as exc:

                            print(
                                f"Pose warning for track "
                                f"{track_ids[i]}: {exc}"
                            )

                            keypoints, scores = None, None

                        track_id = int(track_ids[i])

                        label = "unknown"
                        extra = {}
                        mean_score = 0.0

                        if keypoints is not None and scores is not None:

                            draw_pose(
                                frame,
                                keypoints,
                                scores,
                                x1,
                                y1
                            )

                            summary = classify_posture(
                                keypoints,
                                scores
                            )

                            label = summary["activity"]
                            extra = summary

                            mean_score = round(
                                float(np.mean(scores)),
                                4
                            )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 160, 20),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Track {track_id}: {label}",
                            (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 160, 20),
                            2
                        )

                        event = {
                            "timestamp": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "frame": frame_index,
                            "track_id": track_id,
                            "activity": label,
                            "detector_confidence": round(
                                float(confs[i]), 4
                            ),
                            "mean_keypoint_confidence": mean_score,
                            "pose_model": "yolo11n-pose",
                            **extra
                        }

                        log.write(json.dumps(event) + "\n")

                cv2.putText(
                    frame,
                    f"Frame {frame_index}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                writer.write(frame)
                processed += 1

                if processed % 25 == 0:

                    elapsed = max(
                        time.perf_counter() - started,
                        1e-6
                    )

                    print(
                        f"Processed {processed} frames | "
                        f"{processed / elapsed:.2f} FPS"
                    )

    finally:

        cap.release()
        writer.release()

    print("\nProcessing completed!")
    print(f"Annotated video: {output_path}")
    print(f"Activity events: {events_path}")


if __name__ == "__main__":
    main()
