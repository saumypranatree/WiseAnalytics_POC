#!/usr/bin/env python3
"""Step 2: COLMAP SfM (pycolmap, CPU SIFT) on the selected keyframes.

usage: 02_sfm.py WORK_DIR [--max-image-size 1600] [--overlap 15]
Writes WORK_DIR/sfm/{database.db, images/, sparse/0, sfm_summary.json}.
"""
import argparse, json, os, shutil, time
import pycolmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--max-image-size", type=int, default=1600)
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL")
    ap.add_argument("--camera-params", default="auto",
                    help="prior intrinsics 'f,cx,cy,k' for SIMPLE_RADIAL; 'auto' = f 0.833 x long "
                         "side (iPhone 1x wide, 26 mm-equiv, video crop), centred; '' = COLMAP default")
    ap.add_argument("--focal-factor", type=float, default=0.833)
    ap.add_argument("--fix-focal", action="store_true", help="do not refine the focal in BA")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--name", default="sfm", help="output subdir")
    a = ap.parse_args()
    t0 = time.time()
    info = json.load(open(os.path.join(a.work, "frames.json")))
    sfm = os.path.join(a.work, a.name)
    img_dir = os.path.join(sfm, "images")
    if os.path.exists(sfm):
        shutil.rmtree(sfm)
    os.makedirs(img_dir)
    for n in info["keyframes"]:
        shutil.copy(os.path.join(a.work, "frames", n), os.path.join(img_dir, n))
    db = os.path.join(sfm, "database.db")

    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = a.max_image_size
    ext.sift.max_num_features = a.max_features
    ext.num_threads = a.threads
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = a.camera_model
    if a.camera_params == "auto":
        w, h = info["frame_size_wh"]
        a.camera_params = f"{a.focal_factor * max(w, h):.1f},{w / 2},{h / 2},0"
    if a.camera_params:
        reader.camera_params = a.camera_params      # sets has_prior_focal_length
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext,
                              device=pycolmap.Device.cpu)
    t1 = time.time()
    pair = pycolmap.SequentialPairingOptions()
    pair.overlap = a.overlap
    pair.quadratic_overlap = True
    pair.loop_detection = False
    pycolmap.match_sequential(db, pairing_options=pair, device=pycolmap.Device.cpu)
    t2 = time.time()
    out = os.path.join(sfm, "sparse")
    os.makedirs(out, exist_ok=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = a.threads
    opts.mapper.num_threads = a.threads
    if a.fix_focal:
        opts.ba_refine_focal_length = False
    recs = pycolmap.incremental_mapping(db, img_dir, out, options=opts)
    t3 = time.time()
    summ = {"args": vars(a), "num_keyframes": len(info["keyframes"]), "models": {}}
    for k, r in recs.items():
        summ["models"][str(k)] = {"registered": r.num_reg_images(), "points3D": r.num_points3D(),
                                  "mean_reproj_error": r.compute_mean_reprojection_error(),
                                  "cameras": {str(c): {"model": v.model_name, "params": list(map(float, v.params))}
                                              for c, v in r.cameras.items()}}
    # Later steps read sparse/0, but COLMAP does not number its models by size (IMG_5200:
    # model 0 had 2 images, model 1 had 122). Make the largest model sparse/0.
    if recs:
        best = max(recs, key=lambda k: recs[k].num_reg_images())
        if str(best) != "0" and os.path.isdir(os.path.join(out, str(best))):
            a0, ab, tmp = (os.path.join(out, n) for n in ("0", str(best), "_swap"))
            os.rename(a0, tmp); os.rename(ab, a0); os.rename(tmp, ab)
        n_best = recs[best].num_reg_images()
        summ["used_model"] = {"original_id": str(best), "saved_as": "sparse/0", "registered": n_best,
                              "coverage": round(n_best / max(1, len(info["keyframes"])), 3)}
        if n_best < 0.6 * len(info["keyframes"]):
            print(f"WARNING: partial reconstruction: the largest model has {n_best} of "
                  f"{len(info['keyframes'])} keyframes; the scan covers only part of the video", flush=True)
    summ["seconds"] = {"extract": round(t1 - t0, 1), "match": round(t2 - t1, 1),
                       "map": round(t3 - t2, 1)}
    json.dump(summ, open(os.path.join(sfm, "sfm_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
