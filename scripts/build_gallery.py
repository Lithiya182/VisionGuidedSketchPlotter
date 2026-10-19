"""Build a gallery of finished drawings from logs/demo (safe to re-run)."""
import csv, html, re, shutil
from datetime import datetime
from pathlib import Path

SRC = Path("logs/demo")
OUT = Path("gallery")
IMG = OUT / "images"
PAT = re.compile(
    r"^(?P<image>.+)_(?P<extract>skeleton|canny|boundary)"
    r"_(?P<dist>none|mild|moderate|severe)"
    r"_(?P<order>naive|optimized)_(?P<control>open_loop|cl_prepass)_s(?P<seed>\d+)$")
FIELDS = ["run_id", "image", "extract", "disturbance", "order", "control", "seed",
          "evaluated", "rmse_mm", "chamfer_mm", "tol_iou_pct", "plain_iou_pct",
          "endpoint_error_mm", "pen_up_dist_cm", "exec_time_s",
          "drawing_open_loop", "drawing_closed_loop", "comparison", "date"]

def copy(src):
    if not src.exists():
        return ""
    shutil.copy2(src, IMG / src.name)
    return "images/" + src.name

def cell(path):
    return f'<a href="{path}"><img src="{path}" width="200"></a>' if path else ""

IMG.mkdir(parents=True, exist_ok=True)
rows = []
for mfile in sorted(SRC.glob("*_metrics.csv")):
    run_id = mfile.name[: -len("_metrics.csv")]
    m = PAT.match(run_id)
    if not m:
        print("skipped (name not recognised):", mfile.name)
        continue
    g = m.groupdict()
    ol = copy(SRC / (run_id + "_top_down_ol.png"))
    cl = copy(SRC / (run_id + "_top_down_cl.png"))
    cmp_ = copy(SRC / (run_id + "_comparison.png"))
    when = datetime.fromtimestamp(mfile.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    with open(mfile, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "run_id": run_id, "image": g["image"], "extract": g["extract"],
                "disturbance": g["dist"], "order": g["order"],
                "control": g["control"], "seed": g["seed"],
                "evaluated": r.pop("mode"), **r,
                "drawing_open_loop": ol, "drawing_closed_loop": cl,
                "comparison": cmp_, "date": when})

rows.sort(key=lambda x: x["date"], reverse=True)
with open(OUT / "runs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

body = []
for r in rows:
    info = html.escape(f'{r["image"]} | {r["extract"]} | {r["disturbance"]} | '
                       f'{r["order"]} | {r["control"]} | s{r["seed"]}')
    body.append(
        f'<tr><td>{info}<br><b>{html.escape(r["evaluated"])}</b><br>{r["date"]}</td>'
        f'<td>{cell(r["drawing_open_loop"])}</td><td>{cell(r["drawing_closed_loop"])}</td>'
        f'<td>{cell(r["comparison"])}</td>'
        f'<td>RMSE {r["rmse_mm"]} mm<br>Chamfer {r["chamfer_mm"]} mm<br>'
        f'Tol-IoU {r["tol_iou_pct"]}%</td></tr>')
page = ('<!doctype html><meta charset="utf-8"><title>Sketch Plotter gallery</title>'
        '<style>body{font-family:sans-serif}td{vertical-align:top;padding:6px;'
        'border-bottom:1px solid #ccc}</style><h2>Drawing gallery</h2>'
        '<table><tr><th>Run</th><th>Open-loop</th><th>Closed-loop</th>'
        '<th>Comparison</th><th>Metrics</th></tr>' + "".join(body) + "</table>")
(OUT / "index.html").write_text(page, encoding="utf-8")
print("runs:", len(rows), " | files in gallery\\")