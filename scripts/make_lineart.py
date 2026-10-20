import sys
from pathlib import Path
import cv2, numpy as np

if len(sys.argv) < 2:
    sys.exit("usage: python scripts/make_lineart.py <image> [C]  (C: 7 default; raise to 10-14 for fewer lines)")
src = Path(sys.argv[1])
C = float(sys.argv[2]) if len(sys.argv) > 2 else 7
img = cv2.imread(str(src))
if img is None:
    sys.exit("cannot read image")
s = 480 / max(img.shape[:2])
img = cv2.resize(img, None, fx=s, fy=s,
                 interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
g = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 9, 50, 50)
ink = 255 - cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, C)
n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
keep = np.zeros_like(ink)
for i in range(1, n):
    if st[i, cv2.CC_STAT_AREA] >= 40:
        keep[lab == i] = 255
out = Path("data/uploads") / (src.stem.replace(" ", "_") + "_lineart.png")
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), 255 - keep)
print("saved", out)