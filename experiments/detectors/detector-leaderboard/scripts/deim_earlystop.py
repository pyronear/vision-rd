"""Early-stop watcher for a DEIM/DEIMv2 training run (no native early stopping).

Polls the run's ``log.txt`` for per-epoch validation mAP
(``test_coco_eval_bbox[0]`` = COCO AP@[.5:.95]) and kills the training process
when mAP stops improving for ``--patience`` epochs past its best (the standard
early-stop / overfitting signal). DEIM keeps ``best_stg*.pth`` continuously, so
the peak checkpoint is preserved regardless of when we stop.

    uv run python scripts/deim_earlystop.py \
        --log deimv2_repo/outputs/deimv2_s_smoke_resume/log.txt \
        --pid <torchrun_pid> --patience 15 --min-epochs 25 --poll-seconds 120
"""

import argparse
import contextlib
import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def read_epochs(log_path: Path) -> list[tuple[int, float, float]]:
    """Return [(epoch, val_mAP, train_loss)] parsed from the DEIM log.txt."""
    rows = []
    if not log_path.is_file():
        return rows
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = d.get("test_coco_eval_bbox")
        if m and "epoch" in d:
            loss = float(d.get("train_loss", "nan"))
            rows.append((int(d["epoch"]), float(m[0]), loss))
    return rows


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _stop(pid: int, match: str | None) -> None:
    # SIGTERM the launcher; also pkill children by config match (torchrun spawns).
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)
    if match:
        subprocess.run(["pkill", "-TERM", "-f", match], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="DEIM overfitting early-stop watcher")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--match", type=str, default=None, help="pkill -f pattern")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument(
        "--stop-marker",
        type=Path,
        default=None,
        help="file touched when the early-stop fires (so an orchestrator can tell "
        "an intentional stop from an external kill / crash).",
    )
    args = parser.parse_args()

    best_map, best_epoch = -1.0, -1
    logger.info(
        "watching %s (pid %d, patience %d, min-epochs %d)",
        args.log,
        args.pid,
        args.patience,
        args.min_epochs,
    )
    while True:
        time.sleep(args.poll_seconds)
        rows = read_epochs(args.log)
        if rows:
            for epoch, val_map, _ in rows:
                if val_map > best_map:
                    best_map, best_epoch = val_map, epoch
            cur_epoch, cur_map, cur_loss = rows[-1]
            since = cur_epoch - best_epoch
            logger.info(
                "epoch %d: val_mAP=%.4f (best %.4f @%d, %d since) train_loss=%.3f",
                cur_epoch,
                cur_map,
                best_map,
                best_epoch,
                since,
                cur_loss,
            )
            if cur_epoch >= args.min_epochs and since >= args.patience:
                logger.info(
                    "val mAP has not improved for %d epochs (best %.4f @epoch %d) — "
                    "overfitting/plateau; stopping training.",
                    since,
                    best_map,
                    best_epoch,
                )
                if args.stop_marker is not None:
                    args.stop_marker.write_text(
                        f"early-stop best={best_map:.4f}@{best_epoch}\n"
                    )
                _stop(args.pid, args.match)
                break
        if not _alive(args.pid):
            logger.info("training process ended on its own; watcher exiting.")
            break


if __name__ == "__main__":
    main()
