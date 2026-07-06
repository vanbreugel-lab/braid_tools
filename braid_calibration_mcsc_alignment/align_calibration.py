#!/usr/bin/env python3
"""Align an MCSC multi-camera calibration to arena geometry (automated step 5).

Loads an unaligned <multi_camera_reconstructor> XML, a .braidz arena-tracing
dataset, and a stimxml arena box; auto-fits a similarity transform (scale,
rotation, translation; reflections allowed) mapping the wand data onto the
box surface; shows a live 3D view (arena wireframe, data, camera frusta,
+X/+Y/+Z axes) with manual fine-tuning — including axis flips and 90-degree
hops, since a square arena cross-section leaves an axis ambiguity only your
drawn arrows can resolve — and saves the aligned calibration XML plus a
transform sidecar JSON.

Example:
  python3 align_calibration.py \\
      --unaligned-xml example_calibration_data/20251120_calibration/unaligned/20251121_134739-unaligned.xml \\
      --braidz example_calibration_data/20251120_calibration/aligned/20260706_120045.braidz \\
      --stimxml example_calibration_data/bigwindtunnel_stim.xml
"""

import argparse
import datetime
import json
import os
import sys

os.environ.setdefault("QT_API", "pyqt5")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align_core as core

EPILOG = """If the embedded 3D view misrenders on Wayland (Ubuntu default),
relaunch with:  QT_QPA_PLATFORM=xcb python3 align_calibration.py ..."""

FRUSTUM_LENGTH = 0.15  # m
MAX_DISPLAY_POINTS = 50000
RESIDUAL_CLIM = [0.0, 0.05]  # m, colormap range


# ---------------------------------------------------------------------------
# Paths and sidecar
# ---------------------------------------------------------------------------


def default_output_path(unaligned_xml):
    d, base = os.path.split(os.path.abspath(unaligned_xml))
    stem = base[:-4] if base.endswith(".xml") else base
    if stem.endswith("-unaligned"):
        stem = stem[: -len("-unaligned")]
    return os.path.join(d, stem + "-aligned.xml")


def save_all(cams, M, X, out_xml, meta, force=False):
    """Run numeric checks, write aligned XML + sidecar JSON. Returns file list."""
    if os.path.exists(out_xml) and not force:
        raise FileExistsError(f"{out_xml} exists (use --force / pick another path)")
    worst_px = core.run_save_checks(cams, M, X)
    out_cams = [dict(c, P=core.transform_P(c["P"], M)) for c in cams]
    core.write_calibration_xml(out_cams, out_xml)
    core.check_roundtrip(out_cams, out_xml)
    d = core.decompose_M(M)
    from scipy.spatial.transform import Rotation

    rotvec_deg = np.degrees(
        Rotation.from_matrix(
            d["Q"] @ (np.diag([-1.0, 1.0, 1.0]) if d["det_sign"] < 0 else np.eye(3))
        ).as_rotvec()
    )
    sidecar = {
        "created": datetime.datetime.now().astimezone().isoformat(),
        "M": M.tolist(),
        "decomposition": {
            "scale": d["s"],
            "det_sign": d["det_sign"],
            "rotvec_deg": rotvec_deg.tolist(),
            "translation": d["t"].tolist(),
        },
        "reprojection_check_worst_px": worst_px,
        **meta,
    }
    json_path = out_xml[:-4] + ".transform.json" if out_xml.endswith(".xml") else out_xml + ".transform.json"
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    return [out_xml, json_path]


# ---------------------------------------------------------------------------
# Scene (shared between GUI and headless snapshot)
# ---------------------------------------------------------------------------


def box_wireframe_polydata(verts):
    import pyvista as pv

    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    lines = np.hstack([[2, a, b] for a, b in edges])
    return pv.PolyData(np.asarray(verts, dtype=float), lines=lines)


def frusta_polydata(cams, centers, Ms_P):
    """One PolyData holding rectangle+cone line models for all cameras."""
    import pyvista as pv

    pts, lines = [], []
    for i, (cam, C, Pp) in enumerate(zip(cams, centers, Ms_P)):
        corners = core.frustum_points(Pp, C, cam["resolution"], FRUSTUM_LENGTH)
        base = 5 * i
        pts.extend([C, *corners])
        for k in range(4):
            lines.append([2, base, base + 1 + k])                    # spokes (cone)
            lines.append([2, base + 1 + k, base + 1 + (k + 1) % 4])  # rectangle
    return pv.PolyData(np.asarray(pts), lines=np.hstack(lines))


class Scene:
    """Owns the pyvista actors; update(M) moves points and frusta in place."""

    def __init__(self, plotter, box, X_disp, cams):
        import pyvista as pv

        self.plotter = plotter
        self.box = box
        self.X_disp = X_disp
        self.cams = cams
        self.centers0 = np.array([core.camera_center(c["P"]) for c in cams])
        self.label_actor = None
        self.show_labels = True

        plotter.set_background("#2a2a2e")
        plotter.add_mesh(box_wireframe_polydata(box["verts"]), color="white",
                         line_width=2, name="arena")
        # arena axes: what the user's drawn wand-arrows must match
        for d, color, lbl in [((1, 0, 0), "red", "+X"), ((0, 1, 0), "green", "+Y"),
                              ((0, 0, 1), "blue", "+Z")]:
            plotter.add_mesh(pv.Arrow(start=(0, 0, 0), direction=d, scale=0.3),
                             color=color, name=f"axis{lbl}")
            plotter.add_point_labels([np.asarray(d) * 0.36], [lbl], text_color=color,
                                     font_size=18, shape=None, show_points=False,
                                     name=f"axislbl{lbl}")
        plotter.add_axes()

        self.points_poly = pv.PolyData(X_disp.copy())
        self.points_poly["dist_to_surface_m"] = np.zeros(len(X_disp))
        self.points_actor = plotter.add_mesh(
            self.points_poly, scalars="dist_to_surface_m", clim=RESIDUAL_CLIM,
            cmap="viridis", point_size=4, render_points_as_spheres=True,
            name="wandpoints",
            scalar_bar_args={"title": "dist to surface (m)", "color": "white"},
        )
        self.frusta_poly = None
        self.frusta_actor = None

    def update(self, M, color_by_residual=True, show_frusta=True, point_size=4):
        Xt = core.transform_points(M, self.X_disp)
        self.points_poly.points[:] = Xt
        self.points_poly["dist_to_surface_m"] = core.box_surface_distance(
            Xt, self.box["lo"], self.box["hi"])
        self.points_actor.mapper.scalar_visibility = color_by_residual
        self.points_actor.prop.color = "orange"
        self.points_actor.prop.point_size = point_size

        centers = core.transform_points(M, self.centers0)
        Ps = [core.transform_P(c["P"], M) for c in self.cams]
        poly = frusta_polydata(self.cams, centers, Ps)
        if self.frusta_poly is None:
            self.frusta_poly = poly
            self.frusta_actor = self.plotter.add_mesh(
                poly, color="#ffcc66", line_width=1.5, name="frusta")
        else:
            self.frusta_poly.points[:] = poly.points
        self.frusta_actor.visibility = show_frusta

        if self.label_actor is not None:
            self.plotter.remove_actor(self.label_actor, render=False)
            self.label_actor = None
        if self.show_labels and show_frusta:
            self.label_actor = self.plotter.add_point_labels(
                centers, [c["cam_id"] for c in self.cams], font_size=10,
                text_color="#ffcc66", shape=None, show_points=False,
                name="camlabels")
        self.plotter.render()


def snapshot_png(box, X_disp, cams, M, png_path):
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
    scene = Scene(plotter, box, X_disp, cams)
    scene.update(M)
    plotter.camera_position = "iso"
    plotter.screenshot(png_path)
    plotter.close()


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


def run_gui(box, X, X_disp, cams, fit_result, args):
    from qtpy import QtCore, QtWidgets
    from pyvistaqt import QtInteractor
    from scipy.spatial.transform import Rotation

    c_box = (box["lo"] + box["hi"]) / 2.0

    class AlignWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("braid calibration alignment")
            self.resize(1500, 950)
            self.M_fit0 = fit_result["M"].copy()
            self.M_fit = fit_result["M"].copy()
            self.R_coarse = np.eye(3)
            self._updating = False

            central = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(central)
            self.setCentralWidget(central)

            panel = QtWidgets.QWidget()
            panel.setFixedWidth(330)
            form = QtWidgets.QVBoxLayout(panel)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidget(panel)
            scroll.setWidgetResizable(True)
            scroll.setFixedWidth(350)
            layout.addWidget(scroll)

            self.interactor = QtInteractor(central)
            layout.addWidget(self.interactor, stretch=1)
            self.scene = Scene(self.interactor, box, X_disp, cams)

            def group(title):
                g = QtWidgets.QGroupBox(title)
                v = QtWidgets.QVBoxLayout(g)
                form.addWidget(g)
                return v

            # --- autofit ---
            g = group("Autofit")
            self.btn_refit = QtWidgets.QPushButton("Re-run autofit (seeded from current)")
            self.btn_reset_manual = QtWidgets.QPushButton("Reset manual adjustments")
            self.btn_reset_all = QtWidgets.QPushButton("Reset all (original autofit)")
            for b in (self.btn_refit, self.btn_reset_manual, self.btn_reset_all):
                g.addWidget(b)
            self.btn_refit.clicked.connect(self.on_refit)
            self.btn_reset_manual.clicked.connect(self.reset_manual)
            self.btn_reset_all.clicked.connect(self.reset_all)

            def dspin(lo, hi, step, dec, val=0.0):
                s = QtWidgets.QDoubleSpinBox()
                s.setRange(lo, hi)
                s.setSingleStep(step)
                s.setDecimals(dec)
                s.setValue(val)
                s.setKeyboardTracking(False)
                s.valueChanged.connect(self.on_changed)
                return s

            # --- scale ---
            g = group("Scale (x total)")
            self.spin_scale = dspin(0.2, 5.0, 0.001, 5, 1.0)
            g.addWidget(self.spin_scale)

            # --- rotation ---
            g = group("Rotation (deg, about arena center)")
            self.spin_rot = []
            for ax in "XYZ":
                row = QtWidgets.QHBoxLayout()
                row.addWidget(QtWidgets.QLabel(f"r{ax.lower()}"))
                s = dspin(-180.0, 180.0, 0.1, 2)
                self.spin_rot.append(s)
                row.addWidget(s)
                g.addLayout(row)
            row = QtWidgets.QHBoxLayout()
            for i, ax in enumerate("XYZ"):
                b = QtWidgets.QPushButton(f"{ax}+90°")
                b.clicked.connect(lambda _=False, k=i: self.on_rot90(k))
                row.addWidget(b)
            g.addLayout(row)

            # --- flips ---
            g = group("Flip axes (mirror about arena center)")
            self.chk_flip = []
            row = QtWidgets.QHBoxLayout()
            for ax in "XYZ":
                c = QtWidgets.QCheckBox(f"Flip {ax}")
                c.toggled.connect(self.on_changed)
                self.chk_flip.append(c)
                row.addWidget(c)
            g.addLayout(row)

            # --- translation ---
            g = group("Translation (m, arena axes)")
            self.spin_trans = []
            for ax in "xyz":
                row = QtWidgets.QHBoxLayout()
                row.addWidget(QtWidgets.QLabel(f"t{ax}"))
                s = dspin(-2.0, 2.0, 0.001, 4)
                self.spin_trans.append(s)
                row.addWidget(s)
                g.addLayout(row)

            # --- display ---
            g = group("Display")
            self.chk_residual = QtWidgets.QCheckBox("Color by distance to surface")
            self.chk_residual.setChecked(True)
            self.chk_residual.toggled.connect(self.on_changed)
            g.addWidget(self.chk_residual)
            self.chk_frusta = QtWidgets.QCheckBox("Show cameras")
            self.chk_frusta.setChecked(True)
            self.chk_frusta.toggled.connect(self.on_changed)
            g.addWidget(self.chk_frusta)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("point size"))
            self.spin_ptsize = QtWidgets.QSpinBox()
            self.spin_ptsize.setRange(1, 12)
            self.spin_ptsize.setValue(4)
            self.spin_ptsize.valueChanged.connect(self.on_changed)
            row.addWidget(self.spin_ptsize)
            g.addLayout(row)

            # --- metrics ---
            g = group("Fit quality (all points)")
            self.lbl_metrics = QtWidgets.QLabel()
            self.lbl_metrics.setStyleSheet("font-family: monospace;")
            g.addWidget(self.lbl_metrics)

            # --- save ---
            g = group("Save")
            self.edit_out = QtWidgets.QLineEdit(args.output)
            g.addWidget(self.edit_out)
            self.btn_save = QtWidgets.QPushButton("Save aligned XML")
            self.btn_save.clicked.connect(self.on_save)
            g.addWidget(self.btn_save)
            self.lbl_status = QtWidgets.QLabel()
            self.lbl_status.setWordWrap(True)
            g.addWidget(self.lbl_status)

            form.addStretch(1)

            self.metrics_timer = QtCore.QTimer(self)
            self.metrics_timer.setSingleShot(True)
            self.metrics_timer.setInterval(50)
            self.metrics_timer.timeout.connect(self.update_metrics)

            self.on_changed()
            self.interactor.reset_camera()

        # ---- transform state ----
        def manual_params(self):
            return {
                "scale": self.spin_scale.value(),
                "rot_deg": [s.value() for s in self.spin_rot],
                "trans": [s.value() for s in self.spin_trans],
                "flips": [-1.0 if c.isChecked() else 1.0 for c in self.chk_flip],
            }

        def current_M(self):
            p = self.manual_params()
            return core.compose_manual(
                self.M_fit, c_box, scale=p["scale"], rot_deg=p["rot_deg"],
                trans=p["trans"], flips=p["flips"], R_coarse=self.R_coarse)

        def on_changed(self, *_):
            if self._updating:
                return
            M = self.current_M()
            self.scene.update(
                M, color_by_residual=self.chk_residual.isChecked(),
                show_frusta=self.chk_frusta.isChecked(),
                point_size=self.spin_ptsize.value())
            self.metrics_timer.start()

        def update_metrics(self):
            M = self.current_M()
            m = core.compute_metrics(X, box["lo"], box["hi"], M)
            d = core.decompose_M(M)
            self.lbl_metrics.setText(
                f"median |d|  : {m['median_abs_dist_m']*1000:7.2f} mm\n"
                f"p90 |d|     : {m['p90_abs_dist_m']*1000:7.2f} mm\n"
                f"inliers 1cm : {m['inlier_frac_1cm']*100:6.1f} %\n"
                f"inliers 2cm : {m['inlier_frac_2cm']*100:6.1f} %\n"
                f"total scale : {d['s']:9.5f}\n"
                f"det sign    : {d['det_sign']:+d}"
                + ("  (mirrored)" if d["det_sign"] < 0 else ""))

        def _reset_widgets(self):
            self._updating = True
            self.spin_scale.setValue(1.0)
            for s in self.spin_rot + self.spin_trans:
                s.setValue(0.0)
            for c in self.chk_flip:
                c.setChecked(False)
            self.R_coarse = np.eye(3)
            self._updating = False

        def on_rot90(self, axis):
            rv = np.zeros(3)
            rv[axis] = np.pi / 2
            self.R_coarse = Rotation.from_rotvec(rv).as_matrix() @ self.R_coarse
            self.on_changed()

        def on_refit(self):
            try:
                res = core.refit_from_M(X, box["lo"], box["hi"], self.current_M(),
                                        max_fit_points=args.max_fit_points,
                                        seed=args.seed)
            except RuntimeError as e:
                self.set_status(str(e), error=True)
                return
            self.M_fit = res["M"]
            self._reset_widgets()
            self.on_changed()
            self.set_status("re-fit done")

        def reset_manual(self):
            self._reset_widgets()
            self.on_changed()

        def reset_all(self):
            self.M_fit = self.M_fit0.copy()
            self._reset_widgets()
            self.on_changed()

        def set_status(self, text, error=False):
            self.lbl_status.setStyleSheet(
                "color: #ff6666;" if error else "color: #88cc88;")
            self.lbl_status.setText(text)

        def on_save(self):
            M = self.current_M()
            meta = {
                "unaligned_xml": os.path.abspath(args.unaligned_xml),
                "braidz": os.path.abspath(args.braidz),
                "stimxml": os.path.abspath(args.stimxml),
                "fit_metrics": core.compute_metrics(X, box["lo"], box["hi"], M),
                "manual_adjustments": dict(self.manual_params(),
                                           coarse_rot=self.R_coarse.tolist()),
            }
            try:
                written = save_all(cams, M, X, self.edit_out.text().strip(),
                                   meta, force=args.force)
            except (AssertionError, ValueError, FileExistsError, OSError) as e:
                self.set_status(f"NOT saved: {e}", error=True)
                return
            args.force = True  # re-saving over our own output is fine
            self.set_status("saved:\n" + "\n".join(written))

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = AlignWindow()
    win.show()

    shot = os.environ.get("ALIGN_GUI_SCREENSHOT")
    if shot:  # automated smoke test: screenshot the window, then quit

        def _grab():
            win.grab().save(shot)  # Qt widgets (GL viewport shows as garbage here)
            gl = shot.rsplit(".", 1)[0] + "-viewport.png"
            win.interactor.screenshot(gl)  # the actual VTK framebuffer
            print(f"screenshots saved to {shot} and {gl}")
            app.quit()

        QtCore.QTimer.singleShot(4000, _grab)
    print("GUI checklist: floor at z=0; long axis along ±X; your drawn arrows")
    print("must point along the +X/+Y/+Z arena arrows (use Flip / +90° buttons);")
    print("cameras around the arena pointing inward; walls dark in the residual")
    print("colormap, interior arrows bright. Then Save.")
    sys.exit(app.exec_())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unaligned-xml", required=True,
                    help="unaligned multi_camera_reconstructor XML from mcsc")
    ap.add_argument("--braidz", required=True,
                    help=".braidz arena-tracing dataset (recorded with the unaligned cal)")
    ap.add_argument("--stimxml", required=True, help="arena geometry stimxml")
    ap.add_argument("--output", default=None,
                    help="aligned XML output path (default: <input>-aligned.xml)")
    ap.add_argument("--no-gui", action="store_true",
                    help="headless: autofit, run checks, save XML + JSON + PNG snapshot")
    ap.add_argument("--transform-json", default=None,
                    help="skip autofit; load M from a previously saved sidecar JSON")
    ap.add_argument("--max-fit-points", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    args = ap.parse_args()

    if args.output is None:
        args.output = default_output_path(args.unaligned_xml)

    print(f"loading calibration : {args.unaligned_xml}")
    cams = core.load_calibration_xml(args.unaligned_xml)
    print(f"  {len(cams)} cameras")
    print(f"loading arena       : {args.stimxml}")
    box = core.load_stim_box(args.stimxml)
    print(f"  box lo={box['lo']} hi={box['hi']}")
    print(f"loading braidz      : {args.braidz}")
    X = core.load_wand_points(args.braidz)
    print(f"  {len(X)} 3D points")

    if args.transform_json:
        with open(args.transform_json) as f:
            M = np.array(json.load(f)["M"])
        core.check_similarity(M)
        fit_result = {"M": M, "metrics": core.compute_metrics(X, box["lo"], box["hi"], M)}
        print("loaded transform from JSON (autofit skipped)")
    else:
        print("running autofit (16 symmetry starts)...")
        fit_result = core.autofit(X, box["lo"], box["hi"],
                                  max_fit_points=args.max_fit_points, seed=args.seed)
    m = fit_result["metrics"]
    print(f"  median |d| = {m['median_abs_dist_m']*1000:.2f} mm, "
          f"p90 = {m['p90_abs_dist_m']*1000:.2f} mm, "
          f"inliers@1cm = {m['inlier_frac_1cm']*100:.1f}%, "
          f"@2cm = {m['inlier_frac_2cm']*100:.1f}%")
    d = core.decompose_M(fit_result["M"])
    print(f"  scale = {d['s']:.5f}, det sign = {d['det_sign']:+d}")

    rng = np.random.default_rng(args.seed)
    X_disp = (X if len(X) <= MAX_DISPLAY_POINTS
              else X[rng.choice(len(X), MAX_DISPLAY_POINTS, replace=False)])

    if args.no_gui:
        meta = {
            "unaligned_xml": os.path.abspath(args.unaligned_xml),
            "braidz": os.path.abspath(args.braidz),
            "stimxml": os.path.abspath(args.stimxml),
            "fit_metrics": m,
            "manual_adjustments": None,
        }
        written = save_all(cams, fit_result["M"], X, args.output, meta,
                           force=args.force)
        png = args.output[:-4] + ".png" if args.output.endswith(".xml") else args.output + ".png"
        try:
            snapshot_png(box, X_disp, cams, fit_result["M"], png)
            written.append(png)
        except Exception as e:  # headless rendering can fail without a display
            print(f"warning: could not render PNG snapshot: {e}")
        print("wrote:")
        for p in written:
            print(f"  {p}")
        print("NOTE: geometry-only autofit — axis directions/flips are arbitrary up")
        print("to box symmetry. Run the GUI to check your arrows before trusting axes.")
        return

    run_gui(box, X, X_disp, cams, fit_result, args)


if __name__ == "__main__":
    main()
