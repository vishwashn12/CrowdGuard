"""Open the dashboard in a browser window that draws on the integrated GPU.

    python scripts/open_dashboard.py
    python scripts/open_dashboard.py --url http://localhost:8080

Optional. The dashboard plays three live video streams at full frame rate. A
browser drawing them on the same NVIDIA GPU that runs the models adds GPU
contention. On an RTX 3050 laptop the system keeps full pace either way since the
renderer was moved to one thread per camera (analysis 11.4 frames/s with the
browser on the NVIDIA GPU, 11.7 on the Intel one), so this is headroom rather
than a requirement - useful when the machine is also busy with other work.

The window gets its own browser profile. Chrome and Edge forward a launch to any
instance already running with the same profile and silently drop its flags, so
without a separate profile the GPU choice would be ignored whenever the
operator's normal browser is open.

To make the setting permanent for your normal browser instead: Windows Settings >
System > Display > Graphics, add Chrome or Edge, Options > Power saving.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / ".dashboard-profile"

_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browser() -> str | None:
    """First installed Chromium-based browser."""
    for candidate in _CANDIDATES:
        if candidate and Path(candidate).exists():
            return candidate
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge", "chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the CrowdGuard dashboard on the integrated GPU")
    parser.add_argument("--url", default="http://localhost:8000", help="Dashboard URL")
    parser.add_argument("--browser", default=None, help="Path to chrome.exe or msedge.exe")
    args = parser.parse_args()

    browser = args.browser or find_browser()
    if browser is None:
        print("No Chrome or Edge found. Open the URL yourself, and set your browser to the "
              "integrated GPU in Windows Settings > System > Display > Graphics.")
        return 1

    PROFILE_DIR.mkdir(exist_ok=True)
    command = [
        browser,
        f"--app={args.url}",                 # chromeless window for the operator view
        f"--user-data-dir={PROFILE_DIR}",    # own profile, so the flag below is honoured
        "--force_low_power_gpu",             # draw on the integrated GPU, not the NVIDIA one
        "--no-first-run",
        "--no-default-browser-check",
        "--window-size=1600,1000",
    ]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Opened {args.url} in {Path(browser).stem} on the integrated GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
