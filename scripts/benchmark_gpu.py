"""Measure model throughput on this machine and recommend an anomaly stride.

Why this exists: the anomaly classifier's features are per-frame quantities, so
the pose branch should ideally run at the source frame rate. Whether that is
affordable depends entirely on the hardware, the pose model, ``imgsz``, and how
many cameras share the GPU with CSRNet. Guessing produces either wasted
headroom or silently degraded detection, so measure instead.

    python scripts/benchmark_gpu.py --video "path/to/clip.mp4"
    python scripts/benchmark_gpu.py --video clip.mp4 --cameras 3 --imgsz 640
    python scripts/benchmark_gpu.py --video clip.mp4 --pose-model assets/yolov8s-pose.pt

The report prints measured per-frame latency for each branch, the resulting
sustainable frame rate for N cameras, and the smallest ``anomaly_stride`` that
keeps up. Put that value in your app config.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from crowdguard.config.schemas import AnomalyConfig, PoseTrackModelConfig
from crowdguard.ingestion.frame_packet import FramePacket
from crowdguard.utils.logging_setup import configure_logging

WARMUP_FRAMES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CrowdGuard model throughput")
    parser.add_argument("--video", required=True, help="Sample video to benchmark against")
    parser.add_argument("--frames", type=int, default=30, help="Frames to time (default: 30)")
    parser.add_argument(
        "--cameras", type=int, default=3,
        help="Cameras that will share this GPU (default: 3)",
    )
    parser.add_argument("--pose-model", default="assets/yolov8m-pose.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--csrnet-checkpoint", default="checkpoints/PartAmodel_best.pth")
    parser.add_argument("--density-fps", type=float, default=5.0)
    parser.add_argument(
        "--csrnet-scales", default="0.7,1.0,1.3",
        help="CSRNet multi-scale factors. The default is the calibrated paper "
             "pipeline; '1.0' is ~3x cheaper but under-counts (default: 0.7,1.0,1.3)",
    )
    parser.add_argument(
        "--csrnet-resolution", default="720x540",
        help="CSRNet inference resolution as WxH (default: 720x540)",
    )
    parser.add_argument(
        "--contention-factor", type=float, default=1.15,
        help="Derate applied to single-stream timings when more than one camera "
             "shares the device, covering GPU contention, JPEG encoding and GIL "
             "pressure. 1.15 was measured on an RTX 3050; verify via /api/stats "
             "and raise it if the pose branch still refuses frames (default: 1.6)",
    )
    parser.add_argument(
        "--gpu-budget", type=float, default=0.8,
        help="Fraction of GPU time the models may use, leaving headroom for JPEG "
             "encoding, the decision loop and the web server (default: 0.8)",
    )
    parser.add_argument("--skip-csrnet", action="store_true", help="Benchmark the pose branch only")
    parser.add_argument("--skip-pose", action="store_true", help="Benchmark CSRNet only")
    parser.add_argument(
        "--skip-classifier", action="store_true",
        help="Measure pose detection without the anomaly classifier. Under-measures "
             "the real cost; use only to isolate detection latency.",
    )
    return parser.parse_args()


def load_frames(path: str, count: int) -> tuple[list[np.ndarray], float]:
    """Read frames and the source frame rate from a video."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (1.0 <= fps <= 240.0):
        fps = 25.0

    frames: list[np.ndarray] = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        raise SystemExit(f"No frames could be read from {path}")
    return frames, fps


def make_packet(frame: np.ndarray, index: int, fps: float) -> FramePacket:
    return FramePacket(
        camera_id="bench",
        frame_id=index,
        source_frame_index=index,
        video_timestamp=index / fps,
        wall_clock_timestamp=time.time(),
        frame=frame,
        source_fps=fps,
        anomaly_stride=1,
        density_stride=1,
    )


def time_branch(label: str, run, packets: list[FramePacket]) -> list[float]:
    """Time a callable over packets, discarding warm-up frames."""
    print(f"  {label}: warming up...", end="", flush=True)
    for packet in packets[:WARMUP_FRAMES]:
        run(packet)

    print(f"\r  {label}: timing {len(packets) - WARMUP_FRAMES} frames...", end="", flush=True)
    timings: list[float] = []
    for packet in packets[WARMUP_FRAMES:]:
        started = time.perf_counter()
        run(packet)
        timings.append((time.perf_counter() - started) * 1000.0)

    print(f"\r  {label}: {statistics.median(timings):6.1f} ms median   ")
    return timings


def summarise(timings: list[float]) -> dict:
    ordered = sorted(timings)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max_ms": ordered[-1],
        "fps": 1000.0 / statistics.median(ordered) if ordered else 0.0,
    }


def main() -> int:
    args = parse_args()
    configure_logging("WARNING")

    print(f"\nReading {args.frames} frames from {args.video}")
    frames, source_fps = load_frames(args.video, args.frames + WARMUP_FRAMES)
    packets = [make_packet(f, i, source_fps) for i, f in enumerate(frames)]
    height, width = frames[0].shape[:2]

    if len(packets) <= WARMUP_FRAMES:
        raise SystemExit(
            f"Need more than {WARMUP_FRAMES} frames to benchmark; got {len(packets)}"
        )

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    vram = (
        torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if torch.cuda.is_available() else 0.0
    )

    print(f"Source       : {width}x{height} @ {source_fps:.1f} fps")
    print(f"Device       : {device} ({gpu}{f', {vram:.1f} GB' if vram else ''})")
    print(f"Sharing with : {args.cameras} camera(s)\n")

    pose_stats = None
    csrnet_stats = None

    # ---- pose branch ------------------------------------------------
    #
    # Measured with the classifier loaded and anomaly labelling enabled, which
    # is what actually runs. Detection alone is substantially cheaper than the
    # real path: tracking, building the 10-D feature vector per person and
    # running CatBoost over every mature track all add up, and that cost scales
    # with crowd size. Benchmarking detection alone under-measures badly and
    # recommends a stride the pipeline cannot sustain.
    if not args.skip_pose:
        from crowdguard.models.posetrack.classifier import AnomalyClassifier
        from crowdguard.models.posetrack.infer import PoseTrackInference

        anomaly_config = AnomalyConfig()
        classifier = None
        if not args.skip_classifier:
            try:
                classifier = AnomalyClassifier.load(anomaly_config.classifier_path)
            except Exception as exc:
                print(f"  (classifier unavailable, measuring detection only: {exc})")

        pose_config = PoseTrackModelConfig(model_path=args.pose_model, imgsz=args.imgsz)
        engine = PoseTrackInference(
            pose_config=pose_config,
            anomaly_config=anomaly_config,
            classifier=classifier,
            device=device,
        )
        label = "pose+clf" if classifier else "pose    "
        pose_stats = summarise(time_branch(label, lambda p: engine.infer(p), packets))
        tracks_seen = engine.tracker_for("bench").tracks
        print(f"           tracking {len(tracks_seen)} people, classifier "
              f"{'on' if classifier else 'off'}")

    # ---- density branch ---------------------------------------------
    csrnet_count = None
    if not args.skip_csrnet:
        from crowdguard.models.csrnet.infer import CSRNetInference

        scales = tuple(float(s) for s in args.csrnet_scales.split(",") if s.strip())
        res_w, _, res_h = args.csrnet_resolution.lower().partition("x")
        resolution = (int(res_w), int(res_h))

        engine = CSRNetInference(
            checkpoint_path=args.csrnet_checkpoint,
            device=device,
            inference_resolution=resolution,
            strict_checkpoint=False,
            scales=scales,
        )
        csrnet_stats = summarise(time_branch("csrnet", lambda p: engine.estimate(p), packets))
        # Report the count too: it is what changes when scales or resolution are
        # traded for speed, and it drives every congestion threshold.
        csrnet_count = engine.estimate(packets[WARMUP_FRAMES]).crowd_count

    # ---- report ------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    if pose_stats:
        print(
            f"  Pose   : {pose_stats['median_ms']:6.1f} ms median  "
            f"({pose_stats['p95_ms']:6.1f} ms p95)  ->  {pose_stats['fps']:5.1f} fps single-stream"
        )
    if csrnet_stats:
        count_note = f"   count={csrnet_count:.0f}" if csrnet_count is not None else ""
        print(
            f"  CSRNet : {csrnet_stats['median_ms']:6.1f} ms median  "
            f"({csrnet_stats['p95_ms']:6.1f} ms p95)  ->  {csrnet_stats['fps']:5.1f} fps single-stream"
            f"{count_note}"
        )

    if not pose_stats:
        print("\n  Pose branch not benchmarked, so no stride can be recommended.")
        print("=" * 70 + "\n")
        return 0

    # Both branches contend for one GPU, and work on a single device serialises,
    # so summing the per-frame costs is the right first-order model. Headroom is
    # reserved because JPEG encoding, the decision sweep and the web server all
    # need CPU and some GPU-adjacent time too.
    # Single-stream timings slightly under-measure a multi-camera pipeline,
    # because several YOLO instances and CSRNet share one device and the worker
    # threads contend for the GIL while encoding JPEGs. Measured on an RTX 3050
    # with three cameras: pose cost 27.7 ms alone and 30.8 ms in the running
    # pipeline, about 1.11x, so the 1.15x default is a small safety margin.
    #
    # Treat a much larger apparent factor as a bug rather than a hardware limit.
    # A 1.6x gap here turned out to be the producer pacing only emitted frames,
    # which let clips play back stride-times too fast and flooded the models.
    cameras = max(1, args.cameras)
    contention = 1.0 if cameras == 1 else args.contention_factor

    pose_ms = pose_stats["median_ms"] * contention
    csrnet_ms = (csrnet_stats["median_ms"] if csrnet_stats else 0.0) * contention

    budget_ms = 1000.0 * args.gpu_budget

    if contention > 1.0:
        print(
            f"\n  Applying a {contention:.2f}x contention derate for {cameras} cameras: "
            f"pose {pose_stats['median_ms']:.1f} -> {pose_ms:.1f} ms"
        )

    def pose_cost(stride: int) -> float:
        return cameras * (source_fps / stride) * pose_ms

    def density_cost(fps: float) -> float:
        return cameras * fps * csrnet_ms

    print(
        f"\n  Budget: {budget_ms:.0f} ms of every 1000 ms "
        f"({args.gpu_budget:.0%} of the device), {cameras} camera(s), "
        f"{source_fps:.0f} fps source"
    )
    print(f"    density at {args.density_fps:.1f} fps/cam : {density_cost(args.density_fps):6.0f} ms")
    print(f"    pose at stride 1           : {pose_cost(1):6.0f} ms")

    # Search stride-first, then the richest density rate that still fits.
    #
    # Stride is the scarcer resource and the one that degrades silently: the
    # anomaly classifier's features are only exactly right at stride 1, whereas
    # crowd density genuinely changes over tens of seconds, so 1 fps and 5 fps
    # are near-equivalent for congestion management. Spending the budget on
    # stride first is therefore the better trade.
    density_candidates = sorted(
        {args.density_fps, 2.0, 1.0, 0.5}, reverse=True
    )
    recommendation: tuple[float, int] | None = None
    for stride in range(1, 33):
        pose_ms_total = pose_cost(stride)
        if pose_ms_total > budget_ms:
            continue
        for fps_candidate in density_candidates:
            if fps_candidate > args.density_fps:
                continue
            if pose_ms_total + density_cost(fps_candidate) <= budget_ms:
                recommendation = (fps_candidate, stride)
                break
        if recommendation:
            break

    if recommendation is None:
        print("\n  Nothing fits: the density branch alone exceeds the budget.")
        print(f"  CSRNet costs {csrnet_ms:.0f} ms/frame in this configuration.")
        print("\n  Cheapest effective lever, measured: use a single scale.")
        print("    --csrnet-scales 1.0        roughly 3x faster, under-counts ~10%")
        print("  Lowering --csrnet-resolution is worse value: it costs 2-4x more")
        print("  count accuracy for similar savings, and under-counting makes the")
        print("  system under-react. Otherwise run fewer cameras per GPU.")
        print("=" * 70 + "\n")
        return 0

    rec_fps, rec_stride = recommendation
    total = density_cost(rec_fps) + pose_cost(rec_stride)

    print("\n  RECOMMENDATION")
    print(f"    ingestion.density_fps    : {rec_fps:g}")
    print(f"    ingestion.anomaly_stride : {rec_stride}")
    print(
        f"    projected load           : {total:.0f} ms / 1000 ms "
        f"({total / 10:.0f}% of device)"
    )

    if rec_fps < args.density_fps:
        print(
            f"\n  Note: density was reduced from {args.density_fps:g} to {rec_fps:g} fps/cam to make "
            f"room.\n  Crowd density changes over tens of seconds, so this costs little."
        )

    if rec_stride == 1:
        print("\n  Stride 1: the pose branch runs at the full source rate, which is what")
        print("  the anomaly classifier was trained on. No accuracy compromise.")
    else:
        effective = source_fps / rec_stride
        print(
            f"\n  Stride {rec_stride}: the classifier sees 1 frame in {rec_stride} "
            f"({effective:.1f} fps)."
        )
        print("  Its trajectory features are then computed over longer intervals than it")
        print("  was trained on. scaled_for_stride compensates for the linear part;")
        print("  straightness and cos_sim do not rescale exactly, so expect sitting and")
        print("  counter-flow detection to be less reliable. Congestion is unaffected.")
        print("\n  To reduce the stride, in measured order of value:")
        print("    - yolov8s-pose.pt at imgsz 960   ~2x faster, near-identical recall")
        print("    - --csrnet-scales 1.0            ~3x cheaper density, ~10% under-count")
        print(f"    - lower imgsz                    (currently {args.imgsz}; 640 loses recall)")
        print("    - fewer cameras per GPU")

    # This model is an estimate, and contention is hardware-specific. The
    # pipeline measures the thing that actually matters at runtime, so point
    # the operator at it rather than asking them to trust the arithmetic.
    print("\n  VERIFY once the server is up - this projection is an estimate:")
    print("    curl -s localhost:8000/api/stats")
    print("  buffers.anomaly.dropped should stay near zero and")
    print("  posetrack.continuity_break_ratio should stay low. If frames are")
    print("  being refused, raise anomaly_stride by 1 and re-check.")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
