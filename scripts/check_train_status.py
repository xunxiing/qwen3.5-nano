from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_latest_metrics(log_path: Path) -> dict | None:
    latest = None
    if not log_path.exists():
        return None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in payload:
                latest = payload
    return latest


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    log_path = Path(args.log)
    output_dir = Path(args.output_dir)

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    latest = load_latest_metrics(log_path)
    checkpoints = sorted(output_dir.glob("decoder_step_*.pt"))

    report: dict[str, object] = {
        "config": str(config_path),
        "log": str(log_path),
        "output_dir": str(output_dir),
        "max_steps": int(config["train"]["max_steps"]),
        "checkpoint_count": len(checkpoints),
        "latest_checkpoint": checkpoints[-1].name if checkpoints else None,
    }

    if latest is None:
        report["status"] = "no_metrics_yet"
    else:
        step = int(latest["step"])
        max_steps = int(config["train"]["max_steps"])
        report["status"] = "running" if step < max_steps else "finished"
        report["step"] = step
        report["progress"] = round(step / max_steps, 4)
        report["loss"] = latest.get("loss")
        report["grad_norm"] = latest.get("grad_norm")
        report["gpu_mem_alloc_gb"] = latest.get("gpu_mem_alloc_gb")
        report["gpu_mem_peak_gb"] = latest.get("gpu_mem_peak_gb")
        report["recent_window_sec"] = latest.get("step_time_sec")
        if latest.get("step_time_sec") is not None and step > 0:
            sec_per_step = float(latest["step_time_sec"]) / 10.0
            remaining_steps = max_steps - step
            eta_sec = remaining_steps * sec_per_step
            report["eta_hours"] = round(eta_sec / 3600.0, 2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
