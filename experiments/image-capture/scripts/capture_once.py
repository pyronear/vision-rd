"""Capture the latest image for every pose on every configured camera.

Usage:
    uv run python scripts/capture_once.py
    uv run python scripts/capture_once.py --output-dir data/01_raw/images
    uv run python scripts/capture_once.py --config configs/cameras.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyro_camera_api_client import PyroCameraAPIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def capture_all(config_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text())
    port: int = config.get("api_port", 8081)
    sites: list[dict] = config["cameras"]

    now = datetime.now()
    batch_dir = output_dir / now.strftime("%Y-%m-%d") / now.strftime("%Hh")

    log.info("Starting capture — %d sites, batch dir: %s", len(sites), batch_dir)

    total_saved = 0
    for site_entry in sites:
        site: str = site_entry["site"]
        vpn_ip: str = site_entry["ip"]
        client = PyroCameraAPIClient(base_url=f"http://{vpn_ip}:{port}", timeout=10.0)

        try:
            infos = client.get_camera_infos()
        except Exception as exc:
            log.warning("[%s] Could not fetch camera infos: %s", site, exc)
            continue

        cameras = infos if isinstance(infos, list) else infos.get("cameras", [])
        if not cameras:
            log.warning("[%s] Empty camera_infos response", site)
            continue

        for cam in cameras:
            cam_ip: str = cam.get("ip") or cam.get("camera_id")
            cam_name: str = cam.get("name", cam_ip)
            poses = sorted(int(p) for p in cam.get("poses", []))

            if not poses:
                log.warning("[%s/%s] No poses, skipping", site, cam_name)
                continue

            cam_dir = batch_dir / site / cam_name
            cam_dir.mkdir(parents=True, exist_ok=True)

            for pose in poses:
                try:
                    image = client.get_latest_image(camera_ip=cam_ip, pose=pose)
                except Exception as exc:
                    log.error(
                        "[%s/%s] pose %d — request failed: %s",
                        site, cam_name, pose, exc,
                    )
                    continue

                if image is None:
                    log.warning(
                        "[%s/%s] pose %d — no image (204)", site, cam_name, pose
                    )
                    continue

                out_path = cam_dir / f"pose_{pose}.jpg"
                image.save(out_path, format="JPEG", quality=95)
                log.info("[%s/%s] pose %d — saved", site, cam_name, pose)
                total_saved += 1

    log.info("Capture complete — %d images saved", total_saved)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture latest images from all cameras"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "cameras.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "01_raw" / "images",
    )
    args = parser.parse_args()
    capture_all(args.config, args.output_dir)


if __name__ == "__main__":
    main()
