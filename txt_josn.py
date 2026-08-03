import json
from pathlib import Path

IMG_W = 1920
IMG_H = 1200
LABEL_NAME = "microgel"

txt_path = Path(r"C:\Users\34553\runs\detect\predict-4\labels\Sample 2 - 10x.txt")
image_name = "Sample 2 - 10x.jpg"

shapes = []

with open(txt_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue

        class_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        score = float(parts[5]) if len(parts) >= 6 else None

        # YOLO归一化坐标 -> 像素坐标
        cx = x_center * IMG_W
        cy = y_center * IMG_H
        w = width * IMG_W
        h = height * IMG_H

        # X-AnyLabeling 的 circle:
        # 第一个点是圆心，第二个点是圆上一点
        radius = max(w, h) / 2

        p1 = [cx, cy]
        p2 = [cx + radius, cy]

        shape = {
            "kie_linking": [],
            "label": LABEL_NAME,
            "score": score,
            "points": [p1, p2],
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "circle",
            "flags": {},
            "attributes": {}
        }

        shapes.append(shape)

data = {
    "version": "3.3.7",
    "flags": {},
    "shapes": shapes,
    "imagePath": image_name,
    "imageData": None,
    "imageHeight": IMG_H,
    "imageWidth": IMG_W,
    "description": ""
}

json_path = txt_path.with_suffix(".json")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("转换完成：", json_path)
print("数量：", len(shapes))