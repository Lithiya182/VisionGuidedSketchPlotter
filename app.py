import json, subprocess, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, streamlit as st

sys.path.insert(0, ".")
GAL = Path("gallery")
st.set_page_config(page_title="Vision-Guided Sketch Plotter", layout="wide")
st.title("Vision-Guided Sketch Plotter")
page = st.sidebar.radio("Page", ["Gallery", "Learning", "Preview an image", "Run a drawing"])


def images_in(*folders):
    out = []
    for f in folders:
        p = Path(f)
        if p.exists():
            out += sorted(x for x in p.iterdir()
                          if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    return out


if page == "Gallery":
    f = GAL / "runs.csv"
    if not f.exists():
        st.info("No runs yet. Run a drawing, then python scripts/build_gallery.py")
        st.stop()
    df = pd.read_csv(f)
    c1, c2, c3 = st.columns(3)
    img = c1.multiselect("Image", sorted(df.image.unique()), default=sorted(df.image.unique()))
    dis = c2.multiselect("Disturbance", sorted(df.disturbance.unique()), default=sorted(df.disturbance.unique()))
    ev = c3.multiselect("Evaluated", sorted(df.evaluated.unique()), default=sorted(df.evaluated.unique()))
    d = df[df.image.isin(img) & df.disturbance.isin(dis) & df.evaluated.isin(ev)]
    st.dataframe(d[["image", "extract", "disturbance", "order", "control", "evaluated",
                    "rmse_mm", "chamfer_mm", "tol_iou_pct", "date"]])
    for _, r in d.iterrows():
        with st.expander("%s | %s | %s | %s | IoU %.1f%%" % (
                r.image, r.disturbance, r.order, r.evaluated, r.tol_iou_pct)):
            cols = st.columns(3)
            for col, key, cap in zip(cols, ["drawing_open_loop", "drawing_closed_loop", "comparison"],
                                     ["Open-loop", "Closed-loop", "Comparison"]):
                v = r[key]
                if isinstance(v, str) and v:
                    col.image(str(GAL / v), caption=cap)

elif page == "Learning":
    folder = GAL / "learning"
    curves = sorted(folder.glob("*_curve.png")) if folder.exists() else []
    if not curves:
        st.info("No learning runs yet. Run scripts/learn_loop.py")
    for c in curves:
        st.subheader(c.stem.replace("_curve", ""))
        st.image(str(c))
        base = str(c)[: -len("_curve.png")]
        cols = st.columns(2)
        for col, suf, cap in zip(cols, ["_pass0.png", "_final.png"],
                                 ["Pass 0 (no correction)", "After learning"]):
            if Path(base + suf).exists():
                col.image(base + suf, caption=cap)
    m = Path("learned_memory.json")
    if m.exists():
        h = json.loads(m.read_text()).get("history", [])
        if h:
            st.subheader("What the arm has learned so far")
            st.dataframe(pd.DataFrame(h))

elif page == "Preview an image":
    up = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp"])
    if up:
        Path("data/uploads").mkdir(parents=True, exist_ok=True)
        path = Path("data/uploads") / Path(up.name).name.replace(" ", "_")
        path.write_bytes(up.getvalue())
        from scripts import run_demo as rd
        src = cv2.imread(str(path))
        if src is None:
            st.error("Cannot read that image.")
            st.stop()
        cols = st.columns(4)
        cols[0].image(src, channels="BGR", caption="Original")
        for col, mode in zip(cols[1:], ["skeleton", "canny", "boundary"]):
            try:
                ir = rd.ImageProcessor(rd.VisionConfig(simplification_epsilon_px=1.5)) \
                       .process_image(str(path), mode=mode)
                strokes = [s for s in ir.strokes if len(s) >= 2]
                canvas = np.full((480, 480, 3), 255, np.uint8)
                for s in strokes:
                    cv2.polylines(canvas, [np.int32(np.asarray(s)).reshape(-1, 1, 2)],
                                  False, (200, 0, 0), 2)
                col.image(canvas, channels="BGR", caption="%s: %d strokes, %d points" % (
                    mode, len(strokes), sum(len(s) for s in strokes)))
            except Exception as e:
                col.error("%s failed: %s" % (mode, e))
        st.caption("Choose the mode that gives a clean drawing with fewer than about 80 strokes.")

else:
    imgs = images_in("data/test_images", "data/uploads")
    if not imgs:
        st.info("Put images in data/test_images")
        st.stop()
    choice = st.selectbox("Image", [str(p) for p in imgs])
    c = st.columns(5)
    mode = c[0].selectbox("Extraction", ["skeleton", "canny", "boundary"])
    dist = c[1].selectbox("Disturbance", ["none", "mild", "moderate", "severe"], index=2)
    order = c[2].selectbox("Order", ["optimized", "naive"])
    control = c[3].selectbox("Control", ["cl_prepass", "open_loop"])
    quality = c[4].selectbox("Quality", ["hq", "std"])
    seed = int(st.number_input("Seed", value=42, step=1))
    st.warning("CoppeliaSim must be open with the scene loaded and the simulation stopped.")
    if st.button("Run drawing"):
        cmd = [sys.executable, "scripts/run_demo.py", "--image", choice, "--mode", mode,
               "--disturbance", dist, "--order", order, "--control", control,
               "--seed", str(seed), "--quality", quality]
        with st.spinner("The arm is drawing..."):
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        keep = [l for l in p.stdout.splitlines()
                if any(k in l for k in ("RMSE", "WARN", "Valid", "ERROR"))]
        st.code("\n".join(keep + p.stderr.splitlines()[-5:]) or "(no output)")
        if p.returncode != 0:
            st.error("The run failed. See the lines above.")
            st.stop()
        subprocess.run([sys.executable, "scripts/build_gallery.py"], capture_output=True, text=True)
        rid = "%s%s_%s_%s_%s_%s_s%d" % (Path(choice).stem, "-hq" if quality == "hq" else "",
                                        mode, dist, order, control, seed)
        cols = st.columns(3)
        for col, suf, cap in zip(cols, ["_top_down_ol.png", "_top_down_cl.png", "_comparison.png"],
                                 ["Open-loop", "Closed-loop", "Comparison"]):
            f = Path("logs/demo") / (rid + suf)
            if f.exists():
                col.image(str(f), caption=cap)