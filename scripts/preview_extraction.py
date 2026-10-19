import csv, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, ".")
from scripts import run_demo as rd

SRC = Path("data/test_images")
OUT = Path("gallery/extraction")
OUT.mkdir(parents=True, exist_ok=True)
wc = rd.DEFAULT_WORKSPACE_CONFIG
rows = []

for path in sorted(p for p in SRC.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")):
    src = cv2.imread(str(path))
    if src is None:
        print("cannot read", path.name); continue
    panels = [cv2.resize(src, (240, 240))]
    ok_modes = []
    for mode in ("skeleton", "canny", "boundary"):
        try:
            ir = rd.ImageProcessor(rd.VisionConfig(simplification_epsilon_px=1.5)) \
                   .process_image(str(path), mode=mode)
        except Exception as e:
            print(path.name, mode, "FAILED:", e); continue
        strokes = [s for s in ir.strokes if len(s) >= 2]
        pts = sum(len(s) for s in strokes)
        valid = 0
        for s in strokes:
            a = rd.image_to_arm_space(s, wc.image_width, wc.image_height,
                                      wc.draw_origin_x, wc.draw_origin_y,
                                      wc.draw_width_cm, wc.draw_height_cm)
            if sum(rd._valid_pt(float(p[0]), float(p[1])) for p in a) >= 2:
                valid += 1
        ok = 3 <= len(strokes) <= 80 and pts <= 1500
        if ok:
            ok_modes.append((len(strokes), mode))
        canvas = np.full((480, 480, 3), 255, np.uint8)
        for s in strokes:
            cv2.polylines(canvas, [np.int32(np.asarray(s)).reshape(-1, 1, 2)],
                          False, (200, 0, 0), 2)
        canvas = cv2.resize(canvas, (240, 240))
        cv2.putText(canvas, "%s %d strokes" % (mode, len(strokes)), (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        panels.append(canvas)
        rows.append({"image": path.name, "mode": mode, "strokes": len(strokes),
                     "points": pts, "valid_in_workspace": valid,
                     "usable": "yes" if ok else "no"})
    cv2.imwrite(str(OUT / (path.stem + ".png")), np.hstack(panels))
    best = min(ok_modes)[1] if ok_modes else "none suitable"
    print("%-28s suggested mode: %s" % (path.name, best))

with open("gallery/extraction_report.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image", "mode", "strokes", "points",
                                      "valid_in_workspace", "usable"])
    w.writeheader(); w.writerows(rows)
print("done: gallery\\extraction\\ and gallery\\extraction_report.csv")