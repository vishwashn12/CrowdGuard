"""Entry point: start the pipeline and serve the operator dashboard.

    python scripts/run_server.py
    python scripts/run_server.py --config config/app/demo_samples.yaml
    python scripts/run_server.py --port 8080 --log-level DEBUG

The pipeline runs on its own threads inside this process, so the dashboard and
the models share one address space and one GPU context.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from crowdguard.config.loader import load_app_config, load_dotenv
from crowdguard.decision.config import DecisionConfig
from crowdguard.pipeline.orchestrator import Orchestrator
from crowdguard.utils.logging_setup import configure_logging
from crowdguard.venue.loader import load_venue_topology
from crowdguard.web.server import create_app

logger = logging.getLogger("crowdguard.run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrowdGuard server")
    parser.add_argument(
        "--config", default="config/app/default.yaml",
        help="Application YAML config (default: config/app/default.yaml)",
    )
    parser.add_argument(
        "--venue", default=None,
        help="Venue YAML config. Overrides venue_config from the app config.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--stale-frames", type=int, default=10,
        help="Frame gap past which motion is excluded from congestion scoring",
    )
    parser.add_argument(
        "--alert-log", default="logs/alerts.jsonl",
        help="JSON Lines incident log. Pass an empty string to disable.",
    )
    parser.add_argument("--log-level", default="INFO", help="Root log level")
    parser.add_argument(
        "--no-cameras", action="store_true",
        help="Start with no feeds, ignoring any cameras in the config",
    )
    parser.add_argument(
        "--autostart", action="store_true",
        help="Start monitoring immediately instead of waiting for the Start button. "
             "For headless use; needs footage configured in the app config.",
    )
    return parser.parse_args()


def print_banner(orchestrator: Orchestrator, host: str, port: int) -> None:
    venue = orchestrator.venue
    models = orchestrator.models

    unmonitored = venue.unmonitored_gate_ids
    stride = orchestrator.config.ingestion.anomaly_stride

    lines = [
        "",
        "=" * 74,
        "  CrowdGuard - crowd management and anomaly detection",
        "=" * 74,
        f"  Venue        : {venue.venue_name} ({venue.gate_count} gates, "
        f"{venue.geometry_type} r={venue.radius_m:.0f}m)",
        f"  Gates        : {', '.join(venue.gate_ids)}",
        f"  Footage on   : {', '.join(venue.monitored_gate_ids) or 'none yet'}",
        f"  No footage   : {', '.join(unmonitored) or 'none'}",
        f"  Device       : {models.device}",
        f"  Density      : {'CSRNet @ ' + str(orchestrator.config.ingestion.density_fps) + ' fps' if models.csrnet_enabled else 'disabled'}",
        f"  Pose         : {orchestrator.config.models.posetrack.model_path if models.posetrack_enabled else 'disabled'}"
        f"{f' (every {stride} frame(s))' if models.posetrack_enabled else ''}",
        f"  Anomaly      : {models.classifier.model_name if models.classifier else 'rule-based fallback only'}",
        f"  Playback     : real-time at source fps, {orchestrator.config.display.delay_s:.1f}s behind live",
        f"  Session      : {'RUNNING' if orchestrator.session_running else 'setup - upload footage, then press Start'}",
        "",
        f"  Dashboard    : http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}",
        f"  Optional     : python scripts/open_dashboard.py"
        f"{' --url http://localhost:' + str(port) if port != 8000 else ''}",
        "                 (draws video on the integrated GPU, so the models keep the NVIDIA one)",
        "=" * 74,
        "",
    ]

    if stride > 1:
        lines.insert(-2, (
            f"  NOTE: anomaly_stride={stride} means the classifier sees every {stride}th\n"
            f"        frame. Its trajectory features were trained on consecutive frames,\n"
            f"        so sitting and counter-flow detection are approximate at this\n"
            f"        setting. Congestion management is unaffected.\n"
        ))

    print("\n".join(lines))


def _enable_crash_log() -> Path:
    """Record a stack trace of every thread if the process crashes natively.

    A fault inside OpenCV, PyTorch or the CUDA driver kills the process without
    any Python traceback, leaving nothing to diagnose. faulthandler writes the
    Python stack of every thread at the moment of the crash.
    """
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    crash_path = log_dir / "crash.log"
    # Kept open for the life of the process; faulthandler writes to it on a crash.
    handle = open(crash_path, "a", buffering=1)
    faulthandler.enable(file=handle, all_threads=True)
    return crash_path


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level.upper())
    load_dotenv()
    crash_path = _enable_crash_log()
    logger.info("Native crashes will be recorded in %s", crash_path)

    try:
        config = load_app_config(args.config)
    except Exception as exc:
        logger.error("Could not load %s: %s", args.config, exc)
        return 1

    if args.no_cameras:
        config = config.model_copy(update={"cameras": []})

    try:
        venue = load_venue_topology(args.venue or config.venue_config)
    except Exception as exc:
        logger.error("Could not load venue config: %s", exc)
        return 1

    alert_log = args.alert_log.strip() or None
    if alert_log and not Path(alert_log).is_absolute():
        alert_log = str(ROOT / alert_log)

    try:
        orchestrator = Orchestrator(
            config=config,
            venue=venue,
            decision_config=DecisionConfig(),
            alert_log_path=alert_log,
            stale_frame_gap=args.stale_frames,
        )
    except Exception as exc:
        logger.error("Could not initialise the pipeline: %s", exc)
        return 1

    orchestrator.start(autostart=args.autostart)
    print_banner(orchestrator, args.host, args.port)

    app = create_app(orchestrator)

    def shutdown(_signum, _frame):
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    try:
        signal.signal(signal.SIGTERM, shutdown)
    except (AttributeError, ValueError):
        pass  # not available on every platform

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        orchestrator.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
