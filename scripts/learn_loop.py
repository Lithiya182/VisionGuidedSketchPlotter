"""Learning loop: draw, look at the result with the camera, correct, redraw."""
import argparse, csv, dataclasses, json, sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from scripts import run_demo as rd

PX_MM = 120.0 / 512.0            # 12 cm field of view over 512 px
MEM = Path("learned_memory.json")
OUT = Path("gallery/learning")
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def blur(m):
    return cv2.GaussianBlur((m > 0).astype(np.float32), (0, 0), 1.5)


def est_shift(a, b):
    """Shift d=(dx,dy) in px such that b(x) = a(x - d)."""
    c = np.real(np.fft.ifft2(np.fft.fft2(blur(b)) * np.conj(np.fft.fft2(blur(a)))))
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    h, w = c.shape

    def vertex(lo, mid, hi):
        den = lo - 2 * mid + hi
        return 0.0 if abs(den) < 1e-12 else 0.5 * (lo - hi) / den

    dx = ix + vertex(c[iy, (ix - 1) % w], c[iy, ix], c[iy, (ix + 1) % w])
    dy = iy + vertex(c[(iy - 1) % h, ix], c[iy, ix], c[(iy + 1) % h, ix])
    if dx > w / 2: dx -= w
    if dy > h / 2: dy -= h
    return np.array([dx, dy])


def selftest():
    rng = np.random.default_rng(0)
    m = np.zeros((512, 512), np.uint8)
    for _ in range(40):
        p = rng.integers(100, 400, 4)
        cv2.line(m, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 255, 3)
    d = est_shift(m, np.roll(m, (-4, 7), axis=(0, 1)))
    assert np.abs(d - np.array([7, -4])).max() < 0.6, "shift estimator broken: %s" % d
    print("self-test ok (estimated shift %s)" % np.round(d, 2))


def tol_iou(a, b):
    A = cv2.dilate((a > 0).astype(np.uint8), K5) > 0
    B = cv2.dilate((b > 0).astype(np.uint8), K5) > 0
    return 100.0 * (A & B).sum() / max(1, (A | B).sum())


def mean_dist_mm(a, b):
    A, B = a > 0, b > 0
    dA = cv2.distanceTransform((~A).astype(np.uint8), cv2.DIST_L2, 3)
    dB = cv2.distanceTransform((~B).astype(np.uint8), cv2.DIST_L2, 3)
    return PX_MM * 0.5 * (dB[A].mean() + dA[B].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", default="skeleton", choices=["skeleton", "canny", "boundary"])
    ap.add_argument("--disturbance", default="moderate", choices=["mild", "moderate", "severe"])
    ap.add_argument("--quality", default="std", choices=["std", "hq"])
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--gain", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    selftest()

    rd._QUALITY = args.quality
    wc, cam = rd.DEFAULT_WORKSPACE_CONFIG, rd.DEFAULT_CAMERA_CONFIG
    if args.quality == "hq":
        wc = dataclasses.replace(wc, draw_origin_x=2.75, draw_origin_y=13.75,
                                 draw_width_cm=8.5, draw_height_cm=8.5)
        rd._run_one.__defaults__ = (0.1,)
    raw = rd._load_or_extract_strokes(args.image, args.mode, wc)
    if not raw:
        sys.exit("no valid strokes")
    tour = rd.order_strokes_2opt(raw, rd.DEFAULT_STROKE_ORDER_CONFIG)
    base = [np.array(s.points.tolist() if hasattr(s, "points") else s, dtype=float)
            for s in tour]

    def valid(strokes):
        return all(rd._valid_pt(float(p[0]), float(p[1])) for s in strokes for p in s)

    def draw(shift, seed):
        sh = np.asarray(shift, float)
        for _ in range(8):
            if valid([s + sh for s in base]):
                break
            sh = sh * 0.8
            print("  note: shift reduced to stay inside the workspace")
        else:
            sys.exit("shifted strokes leave the workspace")
        return rd._run_one([s + sh for s in base], args.disturbance, seed,
                           "none", wc, cam)

    # pass 0: no correction. Its target mask is the ORIGINAL target.
    res0 = draw((0.0, 0.0), args.seed)
    tgt, obs0 = res0["tgt_mask"] > 0, res0["obs_mask"] > 0
    if tgt.ndim != 2 or tgt.shape != obs0.shape or obs0.sum() < 200:
        sys.exit("unexpected masks: %s %s ink=%d" % (tgt.shape, obs0.shape, obs0.sum()))
    print("check: this script's Tol-IoU %.1f%% vs run_demo's %.1f%% (should be close)"
          % (tol_iou(obs0, tgt), res0["tol_iou_pct"]))

    mem = json.loads(MEM.read_text()) if MEM.exists() else {}
    key = "%s|%s" % (args.disturbance, args.quality)

    if "M" in mem:
        M = np.array(mem["M"], float)
    else:
        cols = []
        for axis in (0, 1):
            c = np.zeros(2)
            for sign in (1, -1):
                c[:] = 0.0
                c[axis] = 0.3 * sign
                if valid([s + c for s in base]):
                    break
            else:
                sys.exit("probe leaves the workspace")
            r = draw(c, args.seed)
            cols.append(est_shift(obs0, r["obs_mask"] > 0) / c[axis])
        M = np.array(cols).T
    print("camera-to-arm map M (px per cm):\n", np.round(M, 1))
    if abs(np.linalg.det(M)) < 500 or np.linalg.cond(M) > 3:
        sys.exit("camera map looks wrong; not continuing")

    warm = key in mem.get("shifts", {})
    if warm:
        shift = np.array(mem["shifts"][key], float)
        print("warm start from memory: shift (%.2f, %.2f) mm" % tuple(shift * 10))
    else:
        shift = args.gain * np.linalg.solve(M, est_shift(obs0, tgt))
        print("cold start: first correction (%.2f, %.2f) mm" % tuple(shift * 10))

    rows = [dict(p=0, iou=tol_iou(obs0, tgt), dist=mean_dist_mm(obs0, tgt),
                 res=np.linalg.norm(est_shift(obs0, tgt)) * PX_MM)]
    last = res0
    for k in range(1, args.passes + 1):
        res = draw(shift, args.seed + k)
        obs = res["obs_mask"] > 0
        d = est_shift(obs, tgt)
        rows.append(dict(p=k, iou=tol_iou(obs, tgt), dist=mean_dist_mm(obs, tgt),
                         res=np.linalg.norm(d) * PX_MM))
        shift = np.clip(shift + args.gain * np.linalg.solve(M, d), -0.6, 0.6)
        last = res

    print("\npass | Tol-IoU % | mean distance mm | leftover shift mm")
    for r in rows:
        print("%4d | %9.1f | %16.3f | %16.3f" % (r["p"], r["iou"], r["dist"], r["res"]))

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "%s_%s_%s" % (Path(args.image).stem, args.quality, args.disturbance)
    cv2.imwrite(str(OUT / (stem + "_pass0.png")), res0["cam_bgr"])
    cv2.imwrite(str(OUT / (stem + "_final.png")), last["cam_bgr"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    xs = [r["p"] for r in rows]
    ax[0].plot(xs, [r["iou"] for r in rows], "o-"); ax[0].set_title("Tolerance IoU (%)")
    ax[1].plot(xs, [r["dist"] for r in rows], "o-"); ax[1].set_title("Mean distance to target (mm)")
    for a in ax:
        a.set_xlabel("pass (0 = no correction)"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / (stem + "_curve.png"), dpi=130)

    mem["M"] = M.tolist()
    mem.setdefault("shifts", {})[key] = shift.tolist()
    mem.setdefault("history", []).append(dict(
        image=Path(args.image).name, mode=args.mode, key=key, warm_start=warm,
        iou_before=rows[0]["iou"], iou_after=rows[-1]["iou"],
        dist_before=rows[0]["dist"], dist_after=rows[-1]["dist"]))
    MEM.write_text(json.dumps(mem, indent=2, default=float))
    print("\nsaved: gallery\\learning\\%s_*.png and learned_memory.json" % stem)


main()