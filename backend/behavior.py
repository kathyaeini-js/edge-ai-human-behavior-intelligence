"""Small explainable geometric baseline; not a validated behavior classifier."""
from __future__ import annotations
import math
from typing import Any

# COCO-17 landmark indices used by common RTMPose human aliases:
# 5 left shoulder, 6 right shoulder, 11 left hip, 12 right hip.
def _point(keypoints: list, scores: list, idx: int, threshold: float = 0.25):
    if idx >= len(keypoints) or idx >= len(scores) or float(scores[idx]) < threshold:
        return None
    return float(keypoints[idx][0]), float(keypoints[idx][1])

def classify_posture(keypoints: list, scores: list) -> dict[str, Any]:
    left_shoulder = _point(keypoints, scores, 5)
    right_shoulder = _point(keypoints, scores, 6)
    left_hip = _point(keypoints, scores, 11)
    right_hip = _point(keypoints, scores, 12)

    if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
        return {"activity": "unknown", "reason": "insufficient_landmark_confidence"}

    shoulder = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    )
    hip = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2,
    )

    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    # In image coordinates y increases down. Angle is relative to vertical.
    angle = abs(math.degrees(math.atan2(dx, -dy)))

    if angle > 35:
        activity = "possible_bending_or_leaning"
    else:
        activity = "upright_posture"

    return {
        "activity": activity,
        "reason": "torso_angle_heuristic",
        "torso_angle_degrees": round(angle, 2),
    }
