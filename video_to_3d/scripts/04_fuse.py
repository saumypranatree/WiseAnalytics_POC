#!/usr/bin/env python3
"""Step 4: TSDF-fuse the aligned depth maps (Open3D, CPU), extract a mesh, remove
small components / floaters, decimate. Units: SfM units (arbitrary scale).

usage: 04_fuse.py WORK_DIR [--voxel-frac 0.006] [--target-tris 400000]
Writes WORK_DIR/mesh/fused_raw.ply, WORK_DIR/mesh/fused.ply
"""
import argparse, json, os, time
import numpy as np, cv2, open3d as o3d


def edge_mask(depth, thr=0.04):
    """Drop pixels at depth discontinuities (mono depth smears them into floaters)."""
    d = depth.copy()
    d[d <= 0] = np.nan
    gx = np.abs(cv2.Sobel(np.nan_to_num(d), cv2.CV_32F, 1, 0, ksize=3)) / 8
    gy = np.abs(cv2.Sobel(np.nan_to_num(d), cv2.CV_32F, 0, 1, ksize=3)) / 8
    rel = np.maximum(gx, gy) / np.maximum(np.nan_to_num(d, nan=1e9), 1e-6)
    bad = (rel > thr) | ~np.isfinite(d)
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return ~bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--voxel-frac", type=float, default=0.006,
                    help="voxel size as a fraction of the median scene depth")
    ap.add_argument("--trunc-vox", type=float, default=5.0)
    ap.add_argument("--max-depth-x", type=float, default=2.0, help="x median depth")
    ap.add_argument("--target-tris", type=int, default=400000)
    ap.add_argument("--block-count", type=int, default=12000, help="hash blocks (16^3 voxels, ~80 KB each)")
    ap.add_argument("--min-weight", type=float, default=2.0, help="min fused frames per voxel")
    ap.add_argument("--min-component-frac", type=float, default=0.01)
    a = ap.parse_args()
    t0 = time.time()
    dense = os.path.join(a.work, "dense")
    cams = json.load(open(os.path.join(dense, "cameras.json")))
    K = np.array(cams["K"]); W, H = cams["width"], cams["height"]
    med = float(np.median([f["depth_median"] for f in cams["frames"]]))
    vox = a.voxel_frac * med
    import open3d.core as o3c
    dev = o3c.Device("CPU:0")        # Open3D's CUDA kernels hang on sm_120 (Blackwell)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=vox, block_resolution=16,
        block_count=a.block_count, device=dev)
    Kt = o3c.Tensor(K, o3c.float64)
    dmax = a.max_depth_x * med
    kept = []
    for f in cams["frames"]:
        rgb = cv2.imread(os.path.join(dense, "rgb", f["name"] + ".jpg"))[..., ::-1]
        d = np.load(os.path.join(dense, "depth", f["name"] + ".npy"))
        m = edge_mask(d)
        d = np.where(m, d, 0).astype(np.float32)
        kept.append(float(m.mean()))
        T = o3c.Tensor(np.array(f["T_cam_from_world"]), o3c.float64)
        dimg = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(d[..., None])))
        cimg = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(rgb.astype(np.float32) / 255)))
        blk = vbg.compute_unique_block_coordinates(dimg, Kt, T, 1.0, dmax,
                                                   trunc_voxel_multiplier=a.trunc_vox)
        vbg.integrate(blk, dimg, cimg, Kt, Kt, T, 1.0, dmax, trunc_voxel_multiplier=a.trunc_vox)
    mesh = vbg.extract_triangle_mesh(weight_threshold=a.min_weight).to_legacy()
    t1 = time.time()
    os.makedirs(os.path.join(a.work, "mesh"), exist_ok=True)
    o3d.io.write_triangle_mesh(os.path.join(a.work, "mesh", "fused_raw.ply"), mesh)
    n_raw = len(mesh.triangles)
    # keep large connected components
    cl, cnt, _ = mesh.cluster_connected_triangles()
    cl, cnt = np.asarray(cl), np.asarray(cnt)
    keep = cnt[cl] >= a.min_component_frac * cnt.max()
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    n_clean = len(mesh.triangles)
    if n_clean > a.target_tris:
        mesh = mesh.simplify_quadric_decimation(a.target_tris)
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices(); mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(a.work, "mesh", "fused.ply"), mesh)
    info = {"median_depth_sfm": med, "voxel_sfm": vox, "frames": len(cams["frames"]),
            "mean_pixels_kept_after_edge_filter": float(np.mean(kept)),
            "tris_raw": n_raw, "tris_after_components": n_clean,
            "tris_final": len(mesh.triangles), "verts_final": len(mesh.vertices),
            "seconds_fuse": round(t1 - t0, 1), "seconds_total": round(time.time() - t0, 1)}
    json.dump(info, open(os.path.join(a.work, "mesh", "fuse_info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
