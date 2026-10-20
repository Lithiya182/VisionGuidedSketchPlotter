import sys
from pathlib import Path
import cv2, numpy as np

if len(sys.argv) < 2:
    sys.exit("usage: python scripts/make_lineart_variants.py <image>")
src = Path(sys.argv[1])
img = cv2.imread(str(src))
if img is None:
    sys.exit("cannot read image")
s = 480 / max(img.shape[:2])
img = cv2.resize(img, None, fx=s, fy=s,
                 interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bg = cv2.GaussianBlur(g, (0, 0), 25)
norm = cv2.divide(g, bg, scale=255)          # flat paper becomes uniform white
norm = cv2.bilateralFilter(norm, 7, 40, 40)

out = Path("data/uploads")
out.mkdir(parents=True, exist_ok=True)
stem = src.stem.replace(" ", "_")
tiles = [cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)]
for tag, pct in (("A", 4), ("B", 7), ("C", 11)):
    ink = (norm <= np.percentile(norm, pct)).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 35:
            keep[lab == i] = 255
    page = 255 - keep
    cv2.imwrite(str(out / ("%s_line%s.png" % (stem, tag))), page)
    tile = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    cv2.putText(tile, tag, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
    tiles.append(tile)
cv2.imwrite(str(out / (stem + "_variants.png")), np.hstack(tiles))
print("saved %s_lineA/B/C.png and %s_variants.png in data\\uploads" % (stem, stem))