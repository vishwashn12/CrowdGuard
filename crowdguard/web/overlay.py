"""Frame annotation for the operator video panes.

Boxes, motion trails and the crowd-flow arrow are drawn onto display-scaled
frames by the playback renderer, once per frame at the footage's own frame rate.
Positions come from :mod:`crowdguard.display.timeline`, interpolated between
analysis keyframes, and are given in original-frame pixels; ``scale`` maps them
onto the smaller display image.

The CSRNet density map is deliberately not rendered. The heat map told an
operator little that the crowd count and the tracked boxes do not. The count
itself still drives congestion scoring.

Colour carries meaning consistently across the video and the dashboard: green
is normal, amber is a person on the ground, red is counter-flow.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from crowdguard.display.timeline import DisplayPerson
from crowdguard.models.contracts import AnomalyLabel, AnomalyResult

# BGR, matching the dashboard palette.
COLOR_NORMAL = (96, 200, 96)
COLOR_SITTING = (0, 165, 255)
COLOR_OPPOSITE = (60, 60, 240)
COLOR_FLOW = (220, 120, 255)
COLOR_HUD_BG = (28, 28, 32)
COLOR_VETOED = (150, 150, 150)
COLOR_READY = (210, 140, 60)

_LABEL_COLORS = {
    AnomalyLabel.NORMAL: COLOR_NORMAL,
    AnomalyLabel.SITTING: COLOR_SITTING,
    AnomalyLabel.OPPOSITE: COLOR_OPPOSITE,
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_badge(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.45,
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw a filled label badge with its bottom-left near ``(x, y)``.

    Clamped to the frame so a box at the edge still shows its label.
    """
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, _FONT, scale, thickness)

    pad = 4
    x = max(0, min(x, frame.shape[1] - text_w - 2 * pad))
    y = max(text_h + 2 * pad, min(y, frame.shape[0]))

    cv2.rectangle(frame, (x, y - text_h - 2 * pad), (x + text_w + 2 * pad, y), color, -1)
    cv2.putText(frame, text, (x + pad, y - pad), _FONT, scale, text_color, thickness, cv2.LINE_AA)


def render_display_frame(
    image: np.ndarray,
    scale: float,
    people: Iterable[DisplayPerson],
    keyframe: AnomalyResult | None,
    gate_id: str | None = None,
    crowd_count: float | None = None,
    congestion_level: str | None = None,
) -> np.ndarray:
    """Annotate one display frame.

    Args:
        image: Display-scaled BGR frame. Not modified; a copy is drawn on.
        scale: Display width divided by original width.
        people: People positioned for this exact frame.
        keyframe: The analysis result this frame's labels come from, for the
            crowd-flow arrow and the footer counts.
        gate_id: Shown in the header.
        crowd_count: Latest CSRNet count, shown in the header.
        congestion_level: Latest congestion level, shown in the header.
    """
    canvas = image.copy()
    height, width = canvas.shape[:2]

    # Line weights are set for the size the pane is actually viewed at, not the
    # encoded size. A 960 px frame is shown in a dashboard card roughly a third
    # as wide, so a 1 px box all but vanishes there. These stay legible at card
    # size without cluttering the frame when it is enlarged.
    normal_line = max(2, width // 480)
    alarm_line = max(3, width // 240)
    text_scale = max(0.5, width / 1600)

    # Boxes are drawn in one call per colour group rather than one call per
    # person. Every OpenCV call releases and re-acquires Python's GIL, and with
    # forty-odd people that is forty-odd chances to queue behind the model
    # threads. Batching cut render time under load from ~15 ms to a few ms.
    groups: dict[tuple[tuple[int, int, int], int], list[np.ndarray]] = {}
    alarms: list[tuple[DisplayPerson, tuple[int, int, int], int, int]] = []

    for person in people:
        color = COLOR_VETOED if person.was_vetoed else _LABEL_COLORS.get(person.label, COLOR_NORMAL)
        is_anomalous = person.label.is_anomalous and person.confirmed
        x1, y1, x2, y2 = (int(v * scale) for v in person.bbox)
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        groups.setdefault((color, alarm_line if is_anomalous else normal_line), []).append(corners)
        if is_anomalous:
            alarms.append((person, color, x1, y1))

    for (color, thickness), polygons in groups.items():
        cv2.polylines(canvas, polygons, True, color, thickness)

    for person, color, x1, y1 in alarms:
        draw_badge(canvas, f"#{person.track_id} {person.label.display}", x1, y1, color, scale=text_scale)

        # Motion trail: makes a counter-flow call legible at a glance.
        trail = person.trail[-20:]
        for index in range(1, len(trail)):
            fade = index / max(1, len(trail))
            faded = tuple(int(channel * fade) for channel in color)
            cv2.line(
                canvas,
                (int(trail[index - 1][0] * scale), int(trail[index - 1][1] * scale)),
                (int(trail[index][0] * scale), int(trail[index][1] * scale)),
                faded, alarm_line, cv2.LINE_AA,
            )

    if keyframe is not None and keyframe.crowd_direction_valid:
        # Dominant crowd direction, which every counter-flow call is measured against.
        radius = 32
        origin_x, origin_y = width - radius - 14, radius + 14
        end_x = int(origin_x + (radius - 4) * np.cos(keyframe.crowd_direction_rad))
        end_y = int(origin_y + (radius - 4) * np.sin(keyframe.crowd_direction_rad))
        cv2.circle(canvas, (origin_x, origin_y), radius, (40, 40, 48), -1)
        cv2.arrowedLine(canvas, (origin_x, origin_y), (end_x, end_y), COLOR_FLOW, 2, cv2.LINE_AA, tipLength=0.35)

    header = [part for part in (
        gate_id,
        f"count {crowd_count:.0f}" if crowd_count is not None else None,
        congestion_level,
    ) if part]
    if header:
        draw_badge(canvas, "  |  ".join(header), 8, 30, COLOR_HUD_BG, scale=text_scale + 0.1)

    if keyframe is not None:
        footer = (
            f"tracked {keyframe.tracked_count}   "
            f"sitting {keyframe.sitting_count}   "
            f"counter-flow {keyframe.opposite_count}"
        )
        if not keyframe.classifier_active:
            footer += "   [rule-based]"
        if not keyframe.continuity_ok:
            footer += f"   [gap {keyframe.frames_skipped}f]"
        draw_badge(canvas, footer, 8, height - 8, COLOR_HUD_BG, scale=text_scale)

    return canvas


def render_preview(image: np.ndarray, gate_id: str, message: str = "READY  -  press Start") -> np.ndarray:
    """A still of a staged feed's first frame, shown before the session starts.

    Dimmed so it cannot be mistaken for live video.
    """
    canvas = (image.astype(np.float32) * 0.55).astype(np.uint8)
    draw_badge(canvas, gate_id, 8, 26, COLOR_HUD_BG, scale=0.5)

    (text_w, text_h), _ = cv2.getTextSize(message, _FONT, 0.7, 2)
    height, width = canvas.shape[:2]
    draw_badge(canvas, message, (width - text_w) // 2 - 6, height // 2 + text_h // 2, COLOR_READY, scale=0.7)
    return canvas


def scale_for_display(frame: np.ndarray, target_width: int) -> tuple[np.ndarray, float]:
    """Scale a frame to the display width, returning the image and the factor."""
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame, 1.0
    factor = target_width / width
    resized = cv2.resize(frame, (target_width, int(round(height * factor))), interpolation=cv2.INTER_AREA)
    return resized, factor


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR frame as JPEG bytes, or return empty on failure."""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes() if ok else b""


def placeholder_jpeg(width: int = 640, height: int = 360, message: str = "Awaiting feed") -> bytes:
    """A dark placeholder tile for a pane with no frames yet."""
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    (text_w, _), _ = cv2.getTextSize(message, _FONT, 0.7, 1)
    cv2.putText(canvas, message, ((width - text_w) // 2, height // 2), _FONT, 0.7, (110, 110, 120), 1, cv2.LINE_AA)
    return encode_jpeg(canvas, quality=70)
