import argparse
import os
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
MODEL_WEIGHTS = PROJECT_ROOT / "weights" / "yolo26m.pt"
RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26m on the microgel dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Use -1 to let Ultralytics estimate a safe batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to an existing last.pt checkpoint",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {description}: {path}")


def main() -> None:
    args = parse_args()

    if args.resume is not None:
        checkpoint = args.resume.expanduser().resolve()
        require_file(checkpoint, "resume checkpoint")

        print(f"Resuming training from: {checkpoint}")
        model = YOLO(str(checkpoint))
        model.train(resume=True)
        return

    require_file(DATA_YAML, "dataset configuration")
    require_file(MODEL_WEIGHTS, "pretrained model")

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        run_name = f"microgel_yolo26m_{job_id}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"microgel_yolo26m_local_{timestamp}"

    print(f"Dataset: {DATA_YAML}")
    print(f"Model: {MODEL_WEIGHTS}")
    print(f"Run name: {run_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch: {args.batch}")
    print(f"Workers: {args.workers}")

    model = YOLO(str(MODEL_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        optimizer="auto",
        patience=20,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        amp=True,
        cache=False,
        seed=0,
        deterministic=True,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
