import argparse
import csv
import gc
import re
import traceback
from pathlib import Path

import find_best_yolo_thresholds_one_box as core
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

DETAIL_FIELDS = [
    "model",
    "model_path",
    "conf",
    "iou",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "f1",
    "one_box_rate",
    "duplicate_particle_rate",
    "duplicate_box_rate",
    "selection_score",
    "ground_truth",
    "predictions",
    "true_positives",
    "false_positives",
    "false_negatives",
    "one_box_particles",
    "duplicate_particles",
    "duplicate_boxes",
]

SUMMARY_FIELDS = [
    "status",
    "model",
    "model_path",
    "data_path",
    "selection_metric",
    "conf",
    "iou",
    "map50",
    "map50_95",
    "precision",
    "recall",
    "f1",
    "one_box_rate",
    "duplicate_particle_rate",
    "duplicate_box_rate",
    "selection_score",
    "detail_csv",
    "run_signature",
    "error",
]


def portable_path_parts(value):
    """Split Windows or POSIX paths into portable components."""
    text = str(value).strip().replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", text):
        text = text[2:]
    return tuple(part for part in text.split("/") if part not in {"", "."})


def has_expected_type(path, expected_type):
    if expected_type == "file":
        return path.is_file()
    if expected_type == "dir":
        return path.is_dir()
    return path.exists()


def resolve_portable_path(value, base_dir, search_roots=(), expected_type="any"):
    """
    Resolve a local/HCP path, falling back to matching trailing names.

    This lets an absolute Windows path in a dataset YAML relocate to an HCP
    directory with the same final directory names.
    """
    base_dir = Path(base_dir).expanduser().resolve()
    roots = [base_dir]
    roots.extend(Path(root).expanduser().resolve() for root in search_roots)

    unique_roots = []
    seen_roots = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen_roots:
            seen_roots.add(key)
            unique_roots.append(root)

    raw_path = Path(str(value)).expanduser()
    normalized_path = Path(str(value).strip().replace("\\", "/")).expanduser()
    direct_candidates = []
    for path in (raw_path, normalized_path):
        if path.is_absolute():
            direct_candidates.append(path)
        else:
            direct_candidates.append(base_dir / path)

    parts = portable_path_parts(value)
    for root in unique_roots:
        for start_index in range(len(parts)):
            direct_candidates.append(root.joinpath(*parts[start_index:]))

    seen_candidates = set()
    for candidate in direct_candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        key = str(candidate).lower()
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        if has_expected_type(candidate, expected_type):
            return candidate

    if not parts:
        raise FileNotFoundError(f"Cannot resolve empty path: {value!r}")

    target_name = parts[-1]
    ranked_matches = []
    for root in unique_roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob(target_name):
            if not has_expected_type(candidate, expected_type):
                continue

            candidate_parts = tuple(part.lower() for part in candidate.resolve().parts)
            wanted_parts = tuple(part.lower() for part in parts)
            suffix_score = 0
            for candidate_part, wanted_part in zip(
                reversed(candidate_parts), reversed(wanted_parts)
            ):
                if candidate_part != wanted_part:
                    break
                suffix_score += 1

            if suffix_score:
                ranked_matches.append((suffix_score, candidate.resolve()))

    if ranked_matches:
        best_score = max(score for score, _ in ranked_matches)
        best_matches = sorted(
            {
                path
                for score, path in ranked_matches
                if score == best_score
            },
            key=lambda path: str(path).lower(),
        )
        if len(best_matches) == 1:
            relocated = best_matches[0]
            print(f"Relocated path: {value} -> {relocated}")
            return relocated

        match_list = "\n  ".join(str(path) for path in best_matches)
        raise RuntimeError(
            f"Path '{value}' is ambiguous. Matching paths:\n  {match_list}"
        )

    searched = ", ".join(str(root) for root in unique_roots)
    raise FileNotFoundError(f"Could not locate '{value}' below: {searched}")


def resolve_model_spec(model_spec, model_root):
    parts = portable_path_parts(model_spec)
    if not parts:
        raise FileNotFoundError("An empty --model value was supplied.")

    if parts[-1].lower().endswith(".pt"):
        weight_spec = model_spec
    elif parts[-1].lower() == "weights":
        weight_spec = str(model_spec).rstrip("/\\") + "/best.pt"
    else:
        weight_spec = str(model_spec).rstrip("/\\") + "/weights/best.pt"

    return resolve_portable_path(
        value=weight_spec,
        base_dir=model_root,
        search_roots=(model_root, ROOT),
        expected_type="file",
    )


def discover_models(model_root):
    """Find every actual YOLO best.pt below the selected root."""
    models = {
        candidate.resolve()
        for candidate in model_root.rglob("best.pt")
        if candidate.is_file() and candidate.parent.name.lower() == "weights"
    }
    return sorted(models, key=lambda path: str(path).lower())


def model_identifier(model_path, model_root):
    model_dir = model_path.parent.parent
    try:
        relative_dir = model_dir.relative_to(model_root)
        parts = relative_dir.parts
    except ValueError:
        parts = (model_dir.name,)

    readable = "__".join(parts) or model_dir.name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", readable).strip("_")


def ranking_key(row, selection_metric):
    """Choose a meaningful confidence threshold as well as an NMS IoU."""
    if selection_metric == "one_box":
        return (
            row["selection_score"],
            row["f1"],
            row["map50_95"],
            -row["duplicate_particle_rate"],
        )
    if selection_metric == "f1":
        return (
            row["f1"],
            row["map50_95"],
            -row["duplicate_particle_rate"],
            row["precision"],
        )

    # Preserve the original priority: select NMS IoU by mAP50-95. Because
    # standard AP uses a low confidence floor, use the one-box score as the
    # tie-breaker that selects the deployable confidence threshold.
    return (
        row["map50_95"],
        row["selection_score"],
        row["f1"],
        -row["duplicate_particle_rate"],
    )


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def release_gpu_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def search_one_model(
    model_path,
    model_name,
    data_path,
    image_paths,
    ground_truth,
    confidence_values,
    nms_iou_values,
    args,
    detail_path,
):
    print("\n" + "#" * 78)
    print(f"MODEL: {model_name}")
    print(f"WEIGHTS: {model_path}")
    print("#" * 78)

    inference_confidence = min(min(confidence_values), args.map_conf)
    rows = []
    best = None
    model = None

    try:
        model = YOLO(str(model_path))

        for iou_index, nms_iou in enumerate(nms_iou_values, start=1):
            print(
                f"\n[{model_name}] inference for NMS IoU {nms_iou:.2f} "
                f"({iou_index}/{len(nms_iou_values)})"
            )
            records = core.collect_predictions(
                model=model,
                image_paths=image_paths,
                ground_truth=ground_truth,
                confidence=inference_confidence,
                nms_iou=nms_iou,
                image_size=args.imgsz,
                max_detections=args.max_det,
                device=args.device,
                batch=args.batch,
            )
            map_metrics = core.calculate_map_metrics(
                records=records,
                confidence=args.map_conf,
            )

            for confidence in confidence_values:
                metrics = core.calculate_one_box_metrics(
                    records=records,
                    confidence=confidence,
                    match_iou=args.match_iou,
                    duplicate_weight=args.duplicate_weight,
                )
                row = {
                    "model": model_name,
                    "model_path": str(model_path),
                    "conf": confidence,
                    "iou": nms_iou,
                    **map_metrics,
                    **metrics,
                }
                rows.append(row)

                if best is None or ranking_key(
                    row, args.selection_metric
                ) > ranking_key(best, args.selection_metric):
                    best = row.copy()

                print(
                    f"conf={confidence:.2f} iou={nms_iou:.2f} "
                    f"P={row['precision']:.4f} R={row['recall']:.4f} "
                    f"F1={row['f1']:.4f} one-box={row['one_box_rate']:.4f} "
                    f"dup={row['duplicate_particle_rate']:.4f} "
                    f"mAP50={row['map50']:.4f} "
                    f"mAP50-95={row['map50_95']:.4f}"
                )

            # Preserve completed IoU rows even if the Slurm job ends later.
            write_csv(detail_path, DETAIL_FIELDS, rows)
    finally:
        if model is not None:
            del model
        release_gpu_memory()

    if best is None:
        raise RuntimeError(f"No threshold result was produced for {model_name}.")

    print("\n" + "=" * 78)
    print(f"BEST FOR {model_name}")
    print(f"conf={best['conf']:.2f}, iou={best['iou']:.2f}")
    print(
        f"P={best['precision']:.4f}, R={best['recall']:.4f}, "
        f"F1={best['f1']:.4f}, mAP50={best['map50']:.4f}, "
        f"mAP50-95={best['map50_95']:.4f}"
    )
    print(f"Detail CSV: {detail_path}")
    print("=" * 78)
    return best


def load_existing_summary(summary_path):
    if not summary_path.is_file():
        return []
    with summary_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def make_run_signature(args, data_path):
    return "|".join(
        [
            f"data={data_path}",
            f"split={args.split}",
            f"imgsz={args.imgsz}",
            f"conf={args.conf}",
            f"iou={args.iou}",
            f"match_iou={args.match_iou}",
            f"map_conf={args.map_conf}",
            f"max_det={args.max_det}",
            f"selection={args.selection_metric}",
            f"duplicate_weight={args.duplicate_weight}",
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automatically find YOLO best.pt files, search each model's "
            "confidence/NMS-IoU thresholds, and continue to the next model."
        )
    )
    parser.add_argument(
        "--model-root",
        default=ROOT,
        help="Root searched recursively for */weights/best.pt.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "Optional model directory/name/best.pt. Repeat --model to select "
            "several models. If omitted, every best.pt below --model-root is used."
        ),
    )
    parser.add_argument(
        "--data",
        default="microgel_dataset_clean/data_new.yaml",
        help="Dataset YAML; local Windows paths are relocated by matching names.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--conf", default="0.05:0.80:0.05")
    parser.add_argument("--iou", default="0.30:0.90:0.05")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--map-conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--duplicate-weight",
        type=float,
        default=0.25,
        help="Penalty for particles receiving duplicate boxes.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["map50_95", "f1", "one_box"],
        default="map50_95",
        help=(
            "map50_95 selects IoU by mAP then conf by one-box score; f1 or "
            "one_box can instead be the primary ranking metric."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT / "threshold_search_results",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Best-parameter summary CSV (default: OUTPUT_DIR/best_thresholds_summary.csv).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models already completed with exactly the same settings.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed model and continue with the next model.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Resolve and print the model list without running inference.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.match_iou <= 1.0:
        parser.error("--match-iou must be between 0 and 1.")
    if not 0.0 <= args.map_conf <= 1.0:
        parser.error("--map-conf must be between 0 and 1.")
    if args.max_det <= 0:
        parser.error("--max-det must be greater than zero.")
    if args.duplicate_weight < 0.0:
        parser.error("--duplicate-weight must not be negative.")

    model_root = resolve_portable_path(
        value=args.model_root,
        base_dir=ROOT,
        search_roots=(ROOT,),
        expected_type="dir",
    )
    data_path = resolve_portable_path(
        value=args.data,
        base_dir=ROOT,
        search_roots=(model_root, ROOT),
        expected_type="file",
    )
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()

    if args.summary_out is None:
        summary_path = output_dir / "best_thresholds_summary.csv"
    else:
        summary_path = Path(args.summary_out).expanduser()
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        summary_path = summary_path.resolve()

    if args.models:
        model_paths = []
        for model_spec in args.models:
            resolved_model = resolve_model_spec(model_spec, model_root)
            if resolved_model not in model_paths:
                model_paths.append(resolved_model)
    else:
        model_paths = discover_models(model_root)

    if not model_paths:
        raise FileNotFoundError(
            f"No */weights/best.pt files were found below {model_root}"
        )

    print("=" * 78)
    print(f"Model root: {model_root}")
    print(f"Data YAML:  {data_path}")
    print(f"Models:     {len(model_paths)}")
    for index, model_path in enumerate(model_paths, start=1):
        print(f"  {index:02d}. {model_path}")
    print("=" * 78)

    if args.list_models:
        return

    confidence_values = core.parse_range(args.conf)
    nms_iou_values = core.parse_range(args.iou)

    # Patch the core YAML resolver at runtime so Windows paths inside data.yaml
    # can be relocated below the HCP model root.
    core.resolve_dataset_path = lambda value, base_dir: resolve_portable_path(
        value=value,
        base_dir=base_dir,
        search_roots=(model_root, ROOT, data_path.parent),
        expected_type="any",
    )
    image_paths = core.load_split_images(data_path, args.split)
    ground_truth, total_objects, missing_label_files = core.load_ground_truth(
        image_paths
    )
    print(f"Images: {len(image_paths)}")
    print(f"Ground-truth objects: {total_objects}")
    print(f"Missing label files: {missing_label_files}")

    run_signature = make_run_signature(args, data_path)
    summary_rows = load_existing_summary(summary_path)
    completed = {
        (row.get("model_path", ""), row.get("run_signature", ""))
        for row in summary_rows
        if row.get("status") == "completed"
        and row.get("detail_csv")
        and Path(row["detail_csv"]).is_file()
    }

    for model_index, model_path in enumerate(model_paths, start=1):
        model_name = model_identifier(model_path, model_root)
        detail_path = output_dir / f"{model_name}_threshold_search.csv"
        resume_key = (str(model_path), run_signature)

        print("\n" + "*" * 78)
        print(f"MODEL {model_index}/{len(model_paths)}: {model_name}")
        print("*" * 78)

        if args.resume and resume_key in completed:
            print("Already completed with these settings; skipping.")
            continue

        try:
            best = search_one_model(
                model_path=model_path,
                model_name=model_name,
                data_path=data_path,
                image_paths=image_paths,
                ground_truth=ground_truth,
                confidence_values=confidence_values,
                nms_iou_values=nms_iou_values,
                args=args,
                detail_path=detail_path,
            )
            summary_row = {
                "status": "completed",
                "model": model_name,
                "model_path": str(model_path),
                "data_path": str(data_path),
                "selection_metric": args.selection_metric,
                "conf": best["conf"],
                "iou": best["iou"],
                "map50": best["map50"],
                "map50_95": best["map50_95"],
                "precision": best["precision"],
                "recall": best["recall"],
                "f1": best["f1"],
                "one_box_rate": best["one_box_rate"],
                "duplicate_particle_rate": best["duplicate_particle_rate"],
                "duplicate_box_rate": best["duplicate_box_rate"],
                "selection_score": best["selection_score"],
                "detail_csv": str(detail_path),
                "run_signature": run_signature,
                "error": "",
            }
        except Exception as exc:
            traceback.print_exc()
            summary_row = {
                "status": "failed",
                "model": model_name,
                "model_path": str(model_path),
                "data_path": str(data_path),
                "selection_metric": args.selection_metric,
                "conf": "",
                "iou": "",
                "map50": "",
                "map50_95": "",
                "precision": "",
                "recall": "",
                "f1": "",
                "one_box_rate": "",
                "duplicate_particle_rate": "",
                "duplicate_box_rate": "",
                "selection_score": "",
                "detail_csv": str(detail_path),
                "run_signature": run_signature,
                "error": str(exc).replace("\n", " "),
            }

        summary_rows.append(summary_row)
        write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
        print(f"Summary updated: {summary_path}")

        if summary_row["status"] == "failed" and not args.continue_on_error:
            raise RuntimeError(
                f"Threshold search failed for {model_name}: {summary_row['error']}"
            )

    completed_count = sum(
        row.get("status") == "completed"
        and row.get("run_signature") == run_signature
        for row in summary_rows
    )
    failed_count = sum(
        row.get("status") == "failed"
        and row.get("run_signature") == run_signature
        for row in summary_rows
    )
    print("\n" + "=" * 78)
    print("ALL MODELS PROCESSED")
    print(f"Completed: {completed_count}")
    print(f"Failed:    {failed_count}")
    print(f"Summary:   {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
