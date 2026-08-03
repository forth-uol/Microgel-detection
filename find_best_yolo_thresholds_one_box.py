import argparse
import csv
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = (
    ROOT
    / "microgel_finetune_mosaic_flip0_1920-2-2"
    / "weights"
    / "best.pt"
)
DEFAULT_DATA = ROOT / "microgel_dataset_clean" / "data_new.yaml"
DEFAULT_OUT = ROOT / "val_threshold_search_one_box.csv"

IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

MAP_IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


def parse_range(text):
    """Parse a start:stop:step string into a list of floats."""
    try:
        start, stop, step = (float(value) for value in text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid range: {text}. Expected start:stop:step."
        ) from exc

    if step <= 0:
        raise argparse.ArgumentTypeError("The step value must be greater than zero.")

    if start > stop:
        raise argparse.ArgumentTypeError(
            "The start value must not be greater than the stop value."
        )

    values = []
    current = start

    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step

    return values


def resolve_path(value):
    """Resolve a path relative to the script directory."""
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def resolve_dataset_path(value, base_dir):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def collect_images_from_path(path):
    """Expand an image, directory, or image-list text file."""
    if path.is_dir():
        return sorted(
            candidate.resolve()
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        )

    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path.resolve()]

    if path.is_file() and path.suffix.lower() == ".txt":
        images = []
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value:
                continue

            image_path = Path(value).expanduser()
            if not image_path.is_absolute():
                image_path = path.parent / image_path

            images.extend(collect_images_from_path(image_path.resolve()))
        return images

    raise FileNotFoundError(f"Dataset split path not found or unsupported: {path}")


def load_split_images(data_path, split):
    """Read the requested image split from a YOLO dataset YAML."""
    with data_path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset YAML: {data_path}")

    if split not in data:
        raise KeyError(f"Dataset YAML has no '{split}' split: {data_path}")

    dataset_root = data_path.parent
    if data.get("path") is not None:
        dataset_root = resolve_dataset_path(data["path"], data_path.parent)

    split_entries = data[split]
    if not isinstance(split_entries, list):
        split_entries = [split_entries]

    images = []
    for entry in split_entries:
        split_path = resolve_dataset_path(entry, dataset_root)
        images.extend(collect_images_from_path(split_path))

    unique_images = []
    seen = set()
    for image_path in images:
        key = str(image_path)
        if key not in seen:
            seen.add(key)
            unique_images.append(image_path)

    if not unique_images:
        raise RuntimeError(f"No images found for split '{split}' in {data_path}")

    return unique_images


def find_label_path(image_path):
    """Find a YOLO label beside the image or under a parallel labels directory."""
    beside_image = image_path.with_suffix(".txt")
    if beside_image.is_file():
        return beside_image

    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            label_parts = parts.copy()
            label_parts[index] = "labels"
            return Path(*label_parts).with_suffix(".txt")

    return beside_image


def load_ground_truth(image_paths):
    """Load normalized YOLO labels and convert them to pixel xyxy boxes."""
    ground_truth = {}
    missing_label_files = 0
    total_objects = 0

    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        label_path = find_label_path(image_path)
        boxes = []

        if label_path.is_file():
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                values = line.split()
                if not values:
                    continue
                if len(values) < 5:
                    raise ValueError(
                        f"Invalid YOLO label at {label_path}:{line_number}"
                    )

                class_id = int(float(values[0]))
                center_x, center_y, width, height = (
                    float(value) for value in values[1:5]
                )

                x1 = (center_x - width / 2.0) * image_width
                y1 = (center_y - height / 2.0) * image_height
                x2 = (center_x + width / 2.0) * image_width
                y2 = (center_y + height / 2.0) * image_height
                boxes.append((x1, y1, x2, y2, class_id))
        else:
            missing_label_files += 1

        total_objects += len(boxes)
        ground_truth[str(image_path.resolve())] = boxes

    if total_objects == 0:
        raise RuntimeError(
            "No ground-truth objects were loaded. Check the dataset YAML and "
            "the images/labels directory structure."
        )

    return ground_truth, total_objects, missing_label_files


def box_iou(box_a, box_b):
    """Calculate IoU for two xyxy boxes."""
    intersection_x1 = max(box_a[0], box_b[0])
    intersection_y1 = max(box_a[1], box_b[1])
    intersection_x2 = min(box_a[2], box_b[2])
    intersection_y2 = min(box_a[3], box_b[3])

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )
    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )
    union = area_a + area_b - intersection_area

    return 0.0 if union <= 0.0 else intersection_area / union


def collect_predictions(
    model,
    image_paths,
    ground_truth,
    confidence,
    nms_iou,
    image_size,
    max_detections,
    device,
    batch,
):
    """Run the YOLO26 one-to-many head so the NMS IoU is effective."""
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=confidence,
        iou=nms_iou,
        end2end=False,
        imgsz=image_size,
        max_det=max_detections,
        device=device,
        batch=batch,
        half=False,
        dnn=False,
        agnostic_nms=False,
        classes=None,
        save=False,
        save_txt=False,
        verbose=False,
        stream=True,
    )

    records = []
    result_iterator = iter(results)

    # Ultralytics may replace the original filename with synthetic names such
    # as image0.jpg when it processes an in-memory image. Prediction results
    # are returned in source order, so bind each result to the corresponding
    # input path instead of relying on result.path.
    for image_index, image_path in enumerate(image_paths, start=1):
        try:
            result = next(result_iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Expected predictions for {len(image_paths)} images, "
                f"but received only {image_index - 1}."
            ) from exc

        image_key = str(image_path.resolve())

        predictions = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()

            for coordinates_row, score, class_id in zip(
                coordinates,
                confidences,
                classes,
            ):
                predictions.append(
                    (
                        float(coordinates_row[0]),
                        float(coordinates_row[1]),
                        float(coordinates_row[2]),
                        float(coordinates_row[3]),
                        float(score),
                        int(class_id),
                    )
                )

        predictions.sort(key=lambda prediction: prediction[4], reverse=True)
        records.append((predictions, ground_truth[image_key]))

    try:
        next(result_iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError(
            f"Expected predictions for {len(image_paths)} images, "
            "but the model returned additional results."
        )

    return records


def calculate_one_box_metrics(records, confidence, match_iou, duplicate_weight):
    """
    Match every prediction to at most one ground-truth particle.

    The first prediction assigned to a particle is a true positive. Additional
    predictions assigned to the same particle are duplicate false positives.
    """
    total_ground_truth = 0
    total_predictions = 0
    true_positives = 0
    duplicate_boxes = 0
    duplicate_particles = 0
    one_box_particles = 0

    for all_predictions, ground_truth in records:
        predictions = [
            prediction
            for prediction in all_predictions
            if prediction[4] >= confidence
        ]

        hits_per_ground_truth = [0] * len(ground_truth)

        for prediction in predictions:
            best_iou = 0.0
            best_ground_truth_index = None

            for ground_truth_index, target in enumerate(ground_truth):
                if prediction[5] != target[4]:
                    continue

                overlap = box_iou(prediction, target)
                if overlap > best_iou:
                    best_iou = overlap
                    best_ground_truth_index = ground_truth_index

            if (
                best_ground_truth_index is not None
                and best_iou >= match_iou
            ):
                hits_per_ground_truth[best_ground_truth_index] += 1

        detected_particles = sum(
            hit_count >= 1 for hit_count in hits_per_ground_truth
        )
        one_box_particles += sum(
            hit_count == 1 for hit_count in hits_per_ground_truth
        )
        duplicate_particles += sum(
            hit_count >= 2 for hit_count in hits_per_ground_truth
        )
        duplicate_boxes += sum(
            max(0, hit_count - 1) for hit_count in hits_per_ground_truth
        )

        true_positives += detected_particles
        total_ground_truth += len(ground_truth)
        total_predictions += len(predictions)

    false_positives = total_predictions - true_positives
    false_negatives = total_ground_truth - true_positives

    precision = (
        true_positives / total_predictions
        if total_predictions > 0
        else 0.0
    )
    recall = (
        true_positives / total_ground_truth
        if total_ground_truth > 0
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )

    duplicate_particle_rate = (
        duplicate_particles / true_positives
        if true_positives > 0
        else 0.0
    )
    duplicate_box_rate = (
        duplicate_boxes / total_predictions
        if total_predictions > 0
        else 0.0
    )
    one_box_rate = (
        one_box_particles / total_ground_truth
        if total_ground_truth > 0
        else 0.0
    )

    selection_score = f1 - duplicate_weight * duplicate_particle_rate

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "one_box_rate": one_box_rate,
        "duplicate_particle_rate": duplicate_particle_rate,
        "duplicate_box_rate": duplicate_box_rate,
        "selection_score": selection_score,
        "ground_truth": total_ground_truth,
        "predictions": total_predictions,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "one_box_particles": one_box_particles,
        "duplicate_particles": duplicate_particles,
        "duplicate_boxes": duplicate_boxes,
    }


def calculate_precision_recall(records, confidence, match_iou):
    """Calculate standard one-to-one detection precision and recall."""
    total_ground_truth = 0
    total_predictions = 0
    true_positives = 0

    for all_predictions, ground_truth in records:
        predictions = [
            prediction
            for prediction in all_predictions
            if prediction[4] >= confidence
        ]
        matched_ground_truth = set()

        for prediction in predictions:
            best_iou = 0.0
            best_ground_truth_index = None

            for ground_truth_index, target in enumerate(ground_truth):
                if ground_truth_index in matched_ground_truth:
                    continue
                if prediction[5] != target[4]:
                    continue

                overlap = box_iou(prediction, target)
                if overlap > best_iou:
                    best_iou = overlap
                    best_ground_truth_index = ground_truth_index

            if (
                best_ground_truth_index is not None
                and best_iou >= match_iou
            ):
                matched_ground_truth.add(best_ground_truth_index)
                true_positives += 1

        total_ground_truth += len(ground_truth)
        total_predictions += len(predictions)

    false_positives = total_predictions - true_positives
    false_negatives = total_ground_truth - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall}


def calculate_interpolated_ap(true_positive_flags, false_positive_flags, num_gt):
    """Calculate COCO-style 101-point interpolated average precision."""
    if num_gt <= 0 or not true_positive_flags:
        return 0.0

    recalls = []
    precisions = []
    cumulative_true_positives = 0
    cumulative_false_positives = 0

    for is_true_positive, is_false_positive in zip(
        true_positive_flags,
        false_positive_flags,
    ):
        cumulative_true_positives += is_true_positive
        cumulative_false_positives += is_false_positive
        recalls.append(cumulative_true_positives / num_gt)
        precisions.append(
            cumulative_true_positives
            / (cumulative_true_positives + cumulative_false_positives)
        )

    # Convert the raw precision curve into the monotonically decreasing
    # precision envelope used by interpolated AP.
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    interpolated_precision_sum = 0.0
    for recall_index in range(101):
        recall_threshold = recall_index / 100.0
        precision_at_recall = 0.0

        for recall, precision in zip(recalls, precisions):
            if recall >= recall_threshold:
                precision_at_recall = precision
                break

        interpolated_precision_sum += precision_at_recall

    return interpolated_precision_sum / 101.0


def calculate_class_ap(records, class_id, confidence, match_iou):
    """
    Calculate AP for one class and one prediction-to-target IoU threshold.

    Predictions are processed globally in descending confidence order. Each
    target can be matched once; later predictions on the same target are false
    positives, as required for AP calculation.
    """
    ground_truth_by_image = []
    ranked_predictions = []
    num_gt = 0

    for image_index, (all_predictions, ground_truth) in enumerate(records):
        class_ground_truth = [
            target
            for target in ground_truth
            if target[4] == class_id
        ]
        ground_truth_by_image.append(class_ground_truth)
        num_gt += len(class_ground_truth)

        for prediction in all_predictions:
            if prediction[5] == class_id and prediction[4] >= confidence:
                ranked_predictions.append(
                    (prediction[4], image_index, prediction)
                )

    ranked_predictions.sort(key=lambda item: item[0], reverse=True)
    matched_ground_truth = [set() for _ in ground_truth_by_image]
    true_positive_flags = []
    false_positive_flags = []

    for _, image_index, prediction in ranked_predictions:
        best_iou = 0.0
        best_ground_truth_index = None

        for ground_truth_index, target in enumerate(
            ground_truth_by_image[image_index]
        ):
            if ground_truth_index in matched_ground_truth[image_index]:
                continue

            overlap = box_iou(prediction, target)
            if overlap > best_iou:
                best_iou = overlap
                best_ground_truth_index = ground_truth_index

        if (
            best_ground_truth_index is not None
            and best_iou >= match_iou
        ):
            matched_ground_truth[image_index].add(best_ground_truth_index)
            true_positive_flags.append(1)
            false_positive_flags.append(0)
        else:
            true_positive_flags.append(0)
            false_positive_flags.append(1)

    return calculate_interpolated_ap(
        true_positive_flags=true_positive_flags,
        false_positive_flags=false_positive_flags,
        num_gt=num_gt,
    )


def calculate_map_metrics(records, confidence):
    """
    Calculate mAP50 and mAP50-95 over classes that occur in ground truth.

    mAP50 uses a match IoU of 0.50. mAP50-95 averages AP at IoU thresholds
    0.50, 0.55, ..., 0.95. Each AP uses 101-point interpolation.
    """
    class_ids = sorted(
        {
            target[4]
            for _, ground_truth in records
            for target in ground_truth
        }
    )

    if not class_ids:
        return {"map50": 0.0, "map50_95": 0.0}

    map_by_iou = []

    for match_iou in MAP_IOU_THRESHOLDS:
        class_aps = [
            calculate_class_ap(
                records=records,
                class_id=class_id,
                confidence=confidence,
                match_iou=match_iou,
            )
            for class_id in class_ids
        ]
        map_by_iou.append(sum(class_aps) / len(class_aps))

    return {
        "map50": map_by_iou[0],
        "map50_95": sum(map_by_iou) / len(map_by_iou),
    }


def ranking_key(row):
    """Select the best NMS setting using mAP50-95 only."""
    return (row["map50_95"],)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Search YOLO26 confidence and effective NMS IoU thresholds while "
            "penalizing multiple predictions on the same particle."
        )
    )

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument(
        "--conf",
        default="0.05:0.80:0.05",
        help="Confidence threshold range in start:stop:step format.",
    )
    parser.add_argument(
        "--map-conf",
        type=float,
        default=0.001,
        help="Low prediction confidence floor used to build standard AP curves.",
    )
    parser.add_argument(
        "--iou",
        default="0.20:0.70:0.05",
        help="Traditional NMS IoU range in start:stop:step format.",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help="Minimum prediction-to-ground-truth IoU for a match.",
    )
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--out", default=DEFAULT_OUT)

    args = parser.parse_args()

    if not 0.0 <= args.match_iou <= 1.0:
        parser.error("--match-iou must be between 0 and 1.")
    if not 0.0 <= args.map_conf <= 1.0:
        parser.error("--map-conf must be between 0 and 1.")
    if args.max_det <= 0:
        parser.error("--max-det must be greater than zero.")

    model_path = resolve_path(args.model)
    data_path = resolve_path(args.data)
    out_path = resolve_path(args.out)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML file not found: {data_path}")

    confidence_values = parse_range(args.conf)
    nms_iou_values = parse_range(args.iou)
    inference_confidence = min(min(confidence_values), args.map_conf)

    image_paths = load_split_images(data_path, args.split)
    ground_truth, total_objects, missing_label_files = load_ground_truth(
        image_paths
    )

    print("=" * 78)
    print(f"Model:                 {model_path}")
    print(f"Data:                  {data_path}")
    print(f"Split:                 {args.split}")
    print(f"Images:                {len(image_paths)}")
    print(f"Ground-truth objects:  {total_objects}")
    print(f"Missing label files:   {missing_label_files}")
    print(f"Confidence values:     {confidence_values}")
    print(f"NMS IoU values:        {nms_iou_values}")
    print(f"Match IoU:             {args.match_iou:.2f}")
    print(f"mAP confidence floor:  {args.map_conf:.3f}")
    print("Select best by:        mAP50-95 only")
    print("=" * 78)

    model = YOLO(str(model_path))
    rows = []
    best = None

    for nms_iou_index, nms_iou in enumerate(nms_iou_values, start=1):
        print(
            f"\nRunning inference for NMS IoU {nms_iou:.2f} "
            f"({nms_iou_index}/{len(nms_iou_values)})"
        )

        records = collect_predictions(
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

        map_metrics = calculate_map_metrics(
            records=records,
            confidence=args.map_conf,
        )

        for confidence in confidence_values:
            metrics = calculate_precision_recall(
                records=records,
                confidence=confidence,
                match_iou=args.match_iou,
            )

            row = {
                "conf": confidence,
                "iou": nms_iou,
                **map_metrics,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
            rows.append(row)

            if best is None or ranking_key(row) > ranking_key(best):
                best = row.copy()

            print(
                f"conf={confidence:.2f} "
                f"iou={nms_iou:.2f} "
                f"P={row['precision']:.4f} "
                f"R={row['recall']:.4f} "
                f"mAP50={row['map50']:.4f} "
                f"mAP50-95={row['map50_95']:.4f}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "conf",
        "iou",
        "map50",
        "map50_95",
        "precision",
        "recall",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print("Best NMS IoU selected by mAP50-95 only")
    print("=" * 78)
    print(f"NMS IoU:                 {best['iou']:.2f}")
    print(f"mAP50:                   {best['map50']:.4f}")
    print(f"mAP50-95:                {best['map50_95']:.4f}")
    print("-" * 78)
    print("Metrics at the best NMS IoU for every confidence threshold")
    for row in rows:
        if row["iou"] != best["iou"]:
            continue
        print(
            f"conf={row['conf']:.2f} "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"mAP50={row['map50']:.4f} "
            f"mAP50-95={row['map50_95']:.4f}"
        )
    print(f"CSV saved to:            {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
