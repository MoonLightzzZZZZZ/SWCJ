import os
import csv
import struct
import argparse
from itertools import combinations

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from skimage import measure, morphology, draw


# ============================================================
# 1. 基础工具
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_label(label_path):
    nii = nib.load(label_path)
    data = nii.get_fdata().astype(np.int16)
    spacing = tuple(float(x) for x in nii.header.get_zooms()[:3])
    affine = nii.affine
    return data, spacing, affine


def save_nifti(mask, affine, path):
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(nii, path)


def make_sparse_mask(mask, sparse_step=2, keep_first_last=True):
    mask = mask.astype(bool)

    if sparse_step <= 1:
        return mask.copy()

    out = np.zeros_like(mask, dtype=bool)
    z_dim = mask.shape[2]

    keep_z = list(range(0, z_dim, sparse_step))

    if keep_first_last:
        positive_z = np.where(mask.sum(axis=(0, 1)) > 0)[0]
        if len(positive_z) > 0:
            keep_z.append(int(positive_z[0]))
            keep_z.append(int(positive_z[-1]))

    keep_z = sorted(set([z for z in keep_z if 0 <= z < z_dim]))

    for z in keep_z:
        out[:, :, z] = mask[:, :, z]

    return out


# ============================================================
# 2. 评价指标
# ============================================================

def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / denom)


def iou_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(inter / union)


def volume_error(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    v_pred = pred.sum()
    v_gt = gt.sum()
    return 0.0 if v_gt == 0 else float(abs(v_pred - v_gt) / v_gt * 100.0)


def get_surface(mask):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask)
    return mask & (~eroded)


def surface_distances(pred, gt, spacing):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return None, None

    s_pred = get_surface(pred)
    s_gt = get_surface(gt)

    if s_pred.sum() == 0 or s_gt.sum() == 0:
        return None, None

    dt_gt = ndi.distance_transform_edt(~s_gt, sampling=spacing)
    dt_pred = ndi.distance_transform_edt(~s_pred, sampling=spacing)

    d_pred_to_gt = dt_gt[s_pred]
    d_gt_to_pred = dt_pred[s_gt]

    return d_pred_to_gt, d_gt_to_pred


def hd95(pred, gt, spacing):
    d1, d2 = surface_distances(pred, gt, spacing)
    if d1 is None:
        return np.nan
    all_d = np.concatenate([d1, d2])
    return float(np.percentile(all_d, 95)) if len(all_d) > 0 else np.nan


def assd(pred, gt, spacing):
    d1, d2 = surface_distances(pred, gt, spacing)
    if d1 is None or len(d1) == 0 or len(d2) == 0:
        return np.nan
    return float((d1.mean() + d2.mean()) / 2.0)


def connected_component_metrics(mask):
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return 0, 0.0

    lab, num = ndi.label(mask)
    counts = np.bincount(lab.ravel())

    if len(counts) <= 1:
        return 0, 0.0

    counts[0] = 0
    largest = counts.max()
    largest_ratio = float(largest / (mask.sum() + 1e-8))

    return int(num), largest_ratio


def compute_metrics(pred, gt, spacing, name="object"):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    cc_num, lcc_ratio = connected_component_metrics(pred)

    return {
        "structure": name,
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "hd95_mm": hd95(pred, gt, spacing),
        "assd_mm": assd(pred, gt, spacing),
        "volume_error_percent": volume_error(pred, gt),
        "pred_voxels": int(pred.sum()),
        "gt_voxels": int(gt.sum()),
        "connected_components": cc_num,
        "largest_component_ratio": lcc_ratio,
    }


def save_metrics(metrics, csv_path):
    keys = list(metrics[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


# ============================================================
# 3. Signed Distance Field 插值
# ============================================================

def signed_distance_2d(mask2d, spacing_xy=(1.0, 1.0)):
    mask2d = mask2d.astype(bool)
    h, w = mask2d.shape
    far = float(max(h, w))

    if mask2d.sum() == 0:
        return np.ones_like(mask2d, dtype=np.float32) * (-far)

    if mask2d.sum() == mask2d.size:
        return np.ones_like(mask2d, dtype=np.float32) * far

    inside = ndi.distance_transform_edt(mask2d, sampling=spacing_xy)
    outside = ndi.distance_transform_edt(~mask2d, sampling=spacing_xy)

    return (inside - outside).astype(np.float32)


def sdf_threshold_by_target_area(sdf, target_area, base_threshold=0.0, tau_limit=2.0):
    """
    根据目标面积轻微调整 SDF 阈值。
    tau_limit 控制修正强度，避免病灶/血管被过度扩大或缩小。
    """
    target_area = int(max(target_area, 0))

    if target_area <= 0:
        return np.inf

    flat = sdf.ravel()

    if target_area >= flat.size:
        return -np.inf

    kth = flat.size - target_area
    tau = np.partition(flat, kth)[kth]

    tau = float(np.clip(tau, base_threshold - tau_limit, base_threshold + tau_limit))
    return tau


def distance_field_interpolate_sparse_mask(
    sparse_mask,
    spacing_xy=(1.0, 1.0),
    threshold=0.0,
    fill_holes=False,
    closing=False
):
    sparse_mask = sparse_mask.astype(bool)
    out = sparse_mask.copy()

    positive_z = np.where(sparse_mask.sum(axis=(0, 1)) > 0)[0]

    if len(positive_z) == 0:
        return out

    positive_z = sorted([int(z) for z in positive_z])

    for idx in range(len(positive_z) - 1):
        z0 = positive_z[idx]
        z1 = positive_z[idx + 1]

        if z1 <= z0 + 1:
            continue

        mask_a = sparse_mask[:, :, z0]
        mask_b = sparse_mask[:, :, z1]

        sdf_a = signed_distance_2d(mask_a, spacing_xy=spacing_xy)
        sdf_b = signed_distance_2d(mask_b, spacing_xy=spacing_xy)

        gap = z1 - z0

        for z in range(z0 + 1, z1):
            t = (z - z0) / float(gap)
            sdf_t = (1.0 - t) * sdf_a + t * sdf_b

            new_slice = sdf_t >= threshold

            if fill_holes:
                new_slice = ndi.binary_fill_holes(new_slice)

            if closing:
                new_slice = morphology.binary_closing(new_slice, morphology.disk(1))

            out[:, :, z] = new_slice

    return out.astype(bool)


# ============================================================
# 4. DF-MCCS：病灶距离场引导 + 面积趋势校正
# ============================================================

def df_mccs_tumor_interpolation(
    sparse_mask,
    spacing_xy=(1.0, 1.0),
    threshold=0.0,
    tau_limit=1.5,
    min_area_2d=10,
    fill_holes=True,
    closing=True
):
    """
    改进版病灶重建：
    先用距离场保证整体连续性，再用面积趋势做轻微阈值校正。
    """
    sparse_mask = sparse_mask.astype(bool)
    out = sparse_mask.copy()

    positive_z = np.where(sparse_mask.sum(axis=(0, 1)) > 0)[0]

    if len(positive_z) == 0:
        return out

    positive_z = sorted([int(z) for z in positive_z])
    inserted = 0

    for idx in range(len(positive_z) - 1):
        z0 = positive_z[idx]
        z1 = positive_z[idx + 1]

        if z1 <= z0 + 1:
            continue

        mask_a = sparse_mask[:, :, z0]
        mask_b = sparse_mask[:, :, z1]

        area_a = float(mask_a.sum())
        area_b = float(mask_b.sum())

        sdf_a = signed_distance_2d(mask_a, spacing_xy=spacing_xy)
        sdf_b = signed_distance_2d(mask_b, spacing_xy=spacing_xy)

        gap = z1 - z0

        for z in range(z0 + 1, z1):
            t = (z - z0) / float(gap)

            sdf_t = (1.0 - t) * sdf_a + t * sdf_b
            target_area = (1.0 - t) * area_a + t * area_b

            tau = sdf_threshold_by_target_area(
                sdf_t,
                target_area=target_area,
                base_threshold=threshold,
                tau_limit=tau_limit
            )

            new_slice = sdf_t >= tau

            if fill_holes:
                new_slice = ndi.binary_fill_holes(new_slice)

            if closing:
                new_slice = morphology.binary_closing(new_slice, morphology.disk(1))

            if new_slice.sum() < min_area_2d:
                new_slice = np.zeros_like(new_slice, dtype=bool)

            out[:, :, z] = new_slice
            inserted += 1

    print("DF-MCCS inserted tumor slices:", inserted)
    return out.astype(bool)


# ============================================================
# 5. P-BVCI 局部分叉修正
# ============================================================

def remove_small_2d(mask2d, min_size=5):
    mask2d = mask2d.astype(bool)
    lab, num = ndi.label(mask2d)

    if num == 0:
        return mask2d

    out = np.zeros_like(mask2d, dtype=bool)
    counts = np.bincount(lab.ravel())

    for i in range(1, num + 1):
        if counts[i] >= min_size:
            out |= lab == i

    return out


def get_components_2d(mask2d, min_size=5):
    mask2d = mask2d.astype(bool)
    lab, num = ndi.label(mask2d)

    comps = []

    if num == 0:
        return comps

    counts = np.bincount(lab.ravel())

    for i in range(1, num + 1):
        if counts[i] < min_size:
            continue

        comp = lab == i
        coords = np.argwhere(comp)

        if coords.size == 0:
            continue

        area = int(comp.sum())
        center = coords.mean(axis=0).astype(np.float32)

        comps.append({
            "mask": comp,
            "center": center,
            "area": area,
            "label": i
        })

    return comps


def equivalent_radius(area):
    return float(np.sqrt(max(float(area), 1.0) / np.pi))


def angle_between_deg(v1, v2):
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return None

    cosv = float(np.dot(v1, v2) / (n1 * n2 + 1e-8))
    cosv = np.clip(cosv, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosv)))


def interval_penalty(v, low, high):
    if v < low:
        return float(low - v)
    if v > high:
        return float(v - high)
    return 0.0


def score_bifurcation(
    parent,
    child1,
    child2,
    theta_range=(35.0, 52.0),
    ratio_range=(1.15, 1.40),
    max_dist=28.0
):
    cp = parent["center"]
    c1 = child1["center"]
    c2 = child2["center"]

    v1 = c1 - cp
    v2 = c2 - cp

    d1 = float(np.linalg.norm(v1))
    d2 = float(np.linalg.norm(v2))

    if d1 > max_dist or d2 > max_dist:
        return None

    opening_angle = angle_between_deg(v1, v2)

    if opening_angle is None:
        return None

    theta = opening_angle / 2.0

    rp = equivalent_radius(parent["area"])
    r1 = equivalent_radius(child1["area"])
    r2 = equivalent_radius(child2["area"])
    mean_child = 0.5 * (r1 + r2)

    ratio = rp / (mean_child + 1e-8)

    # 极端情况拒绝，正常范围用软约束打分
    if theta < 15.0 or theta > 80.0:
        return None

    if ratio < 0.60 or ratio > 2.60:
        return None

    balance = abs(r1 - r2) / (mean_child + 1e-8)
    dist_pen = (d1 + d2) / (2.0 * max_dist + 1e-8)
    theta_pen = interval_penalty(theta, theta_range[0], theta_range[1])
    ratio_pen = interval_penalty(ratio, ratio_range[0], ratio_range[1])

    score = dist_pen + 0.5 * balance + 0.06 * theta_pen + 1.8 * ratio_pen

    return {
        "score": float(score),
        "theta": float(theta),
        "ratio": float(ratio),
        "parent_radius": float(rp),
        "child_radii": (float(r1), float(r2)),
    }


def find_bifurcations(
    comps_parent,
    comps_child,
    max_dist=28.0,
    theta_range=(35.0, 52.0),
    ratio_range=(1.15, 1.40),
    max_children_per_parent=6,
    max_groups=10
):
    candidates = []

    if len(comps_parent) == 0 or len(comps_child) < 2:
        return []

    for ip, parent in enumerate(comps_parent):
        nearby = []

        for jc, child in enumerate(comps_child):
            d = float(np.linalg.norm(parent["center"] - child["center"]))
            if d <= max_dist:
                nearby.append((d, jc))

        nearby = sorted(nearby, key=lambda x: x[0])[:max_children_per_parent]
        child_ids = [x[1] for x in nearby]

        if len(child_ids) < 2:
            continue

        for j1, j2 in combinations(child_ids, 2):
            info = score_bifurcation(
                parent,
                comps_child[j1],
                comps_child[j2],
                theta_range=theta_range,
                ratio_range=ratio_range,
                max_dist=max_dist
            )

            if info is None:
                continue

            candidates.append((info["score"], ip, j1, j2, info))

    candidates.sort(key=lambda x: x[0])

    used_parent = set()
    used_child = set()
    groups = []

    for _, ip, j1, j2, info in candidates:
        if ip in used_parent:
            continue
        if j1 in used_child or j2 in used_child:
            continue

        groups.append({
            "parent": ip,
            "children": [j1, j2],
            "info": info
        })

        used_parent.add(ip)
        used_child.add(j1)
        used_child.add(j2)

        if len(groups) >= max_groups:
            break

    return groups


def draw_tapered_bridge(shape, p0, p1, r0, r1, scale=0.60):
    out = np.zeros(shape, dtype=bool)

    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)

    length = float(np.linalg.norm(p1 - p0))

    if length < 1e-6:
        rr, cc = draw.disk(
            (float(p0[0]), float(p0[1])),
            radius=max(1.0, r0 * scale),
            shape=shape
        )
        out[rr, cc] = True
        return out

    n_steps = max(6, int(np.ceil(length * 1.5)))

    for s in np.linspace(0.0, 1.0, n_steps):
        p = (1.0 - s) * p0 + s * p1
        r = (1.0 - s) * r0 + s * r1
        r = max(1.0, float(r) * scale)

        rr, cc = draw.disk(
            (float(p[0]), float(p[1])),
            radius=r,
            shape=shape
        )
        out[rr, cc] = True

    return out


def create_split_bridge(shape, parent, children, t, bridge_scale=0.60):
    """
    只在 DT 结果基础上添加局部桥接，不替代原始 DT 结构。
    """
    out = np.zeros(shape, dtype=bool)

    parent_center = parent["center"]
    parent_r = equivalent_radius(parent["area"])

    child_centers = [c["center"] for c in children]
    child_rs = [equivalent_radius(c["area"]) for c in children]

    mean_child_center = np.mean(np.stack(child_centers, axis=0), axis=0)
    mean_child_r = float(np.mean(child_rs))

    branch_center = (1.0 - t) * parent_center + t * mean_child_center
    branch_r = (1.0 - t) * parent_r + t * mean_child_r

    rr, cc = draw.disk(
        (float(branch_center[0]), float(branch_center[1])),
        radius=max(1.0, branch_r * bridge_scale),
        shape=shape
    )
    out[rr, cc] = True

    for child_center, child_r in zip(child_centers, child_rs):
        end_center = (1.0 - t) * parent_center + t * child_center
        end_r = (1.0 - t) * parent_r + t * child_r

        out |= draw_tapered_bridge(
            shape,
            branch_center,
            end_center,
            branch_r,
            end_r,
            scale=bridge_scale
        )

    return out


def df_pbvci_vessel_interpolation(
    sparse_mask,
    spacing_xy=(1.0, 1.0),
    threshold=0.0,
    min_area_2d=5,
    max_match_dist=28.0,
    theta_range=(35.0, 52.0),
    ratio_range=(1.15, 1.40),
    bridge_scale=0.60,
    max_groups_per_gap=10,
    clean_small=True
):
    """
    改进版血管重建：
    1. 先用 DT 插值保证整体连续；
    2. 只在分叉/合并候选区域添加局部生理桥接；
    3. 不再用 P-BVCI 替代整层血管，避免错连和团块化。
    """
    sparse_mask = sparse_mask.astype(bool)

    base = distance_field_interpolate_sparse_mask(
        sparse_mask,
        spacing_xy=spacing_xy,
        threshold=threshold,
        fill_holes=False,
        closing=False
    )

    out = base.copy()

    positive_z = np.where(sparse_mask.sum(axis=(0, 1)) > 0)[0]

    if len(positive_z) == 0:
        return out

    positive_z = sorted([int(z) for z in positive_z])

    inserted = 0
    total_split = 0
    total_merge = 0
    total_added = 0

    for idx in range(len(positive_z) - 1):
        z0 = positive_z[idx]
        z1 = positive_z[idx + 1]

        if z1 <= z0 + 1:
            continue

        slice_a = remove_small_2d(sparse_mask[:, :, z0], min_size=min_area_2d)
        slice_b = remove_small_2d(sparse_mask[:, :, z1], min_size=min_area_2d)

        comps_a = get_components_2d(slice_a, min_size=min_area_2d)
        comps_b = get_components_2d(slice_b, min_size=min_area_2d)

        split_groups = find_bifurcations(
            comps_a,
            comps_b,
            max_dist=max_match_dist,
            theta_range=theta_range,
            ratio_range=ratio_range,
            max_groups=max_groups_per_gap
        )

        merge_groups = find_bifurcations(
            comps_b,
            comps_a,
            max_dist=max_match_dist,
            theta_range=theta_range,
            ratio_range=ratio_range,
            max_groups=max_groups_per_gap
        )

        total_split += len(split_groups)
        total_merge += len(merge_groups)

        gap = z1 - z0

        for z in range(z0 + 1, z1):
            t = (z - z0) / float(gap)

            add_slice = np.zeros_like(slice_a, dtype=bool)

            for g in split_groups:
                parent = comps_a[g["parent"]]
                children = [comps_b[j] for j in g["children"]]

                add_slice |= create_split_bridge(
                    slice_a.shape,
                    parent,
                    children,
                    t=t,
                    bridge_scale=bridge_scale
                )

            for g in merge_groups:
                parent = comps_b[g["parent"]]
                children = [comps_a[j] for j in g["children"]]

                add_slice |= create_split_bridge(
                    slice_a.shape,
                    parent,
                    children,
                    t=1.0 - t,
                    bridge_scale=bridge_scale
                )

            before = out[:, :, z].sum()
            out[:, :, z] = np.logical_or(out[:, :, z], add_slice)
            after = out[:, :, z].sum()

            total_added += int(after - before)
            inserted += 1

            if clean_small:
                out[:, :, z] = remove_small_2d(
                    out[:, :, z],
                    min_size=max(2, min_area_2d // 2)
                )

    print("DF-P-BVCI inserted vessel slices:", inserted)
    print("DF-P-BVCI split groups:", total_split)
    print("DF-P-BVCI merge groups:", total_merge)
    print("DF-P-BVCI added voxels:", total_added)

    return out.astype(bool)


# ============================================================
# 6. 三维重建
# ============================================================

def bbox_from_mask(mask, margin=6):
    coords = np.argwhere(mask)

    if coords.size == 0:
        raise ValueError("mask 为空，无法计算 bbox")

    mn = coords.min(axis=0)
    mx = coords.max(axis=0) + 1

    mn = np.maximum(mn - margin, 0)
    mx = np.minimum(mx + margin, mask.shape)

    return mn.astype(int), mx.astype(int)


def remove_small_3d(mask, min_size=100):
    if min_size <= 0:
        return mask.astype(bool)
    return morphology.remove_small_objects(mask.astype(bool), min_size=min_size)


def keep_largest_component_3d(mask):
    lab, num = ndi.label(mask.astype(bool))

    if num == 0:
        return mask.astype(bool)

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest = counts.argmax()

    return lab == largest


def build_vertex_adjacency(n_vertices, faces):
    adjacency = [set() for _ in range(n_vertices)]

    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])

        adjacency[a].add(b)
        adjacency[a].add(c)

        adjacency[b].add(a)
        adjacency[b].add(c)

        adjacency[c].add(a)
        adjacency[c].add(b)

    return [np.asarray(list(x), dtype=np.int32) for x in adjacency]


def laplacian_step(vertices, adjacency, lam):
    new_vertices = vertices.copy()

    for i, neigh in enumerate(adjacency):
        if len(neigh) == 0:
            continue

        avg = vertices[neigh].mean(axis=0)
        new_vertices[i] = vertices[i] + lam * (avg - vertices[i])

    return new_vertices


def taubin_smooth(vertices, faces, iterations=10, lamb=0.45, mu=-0.50):
    if iterations <= 0:
        return vertices.astype(np.float32)

    vertices = vertices.astype(np.float32).copy()
    adjacency = build_vertex_adjacency(vertices.shape[0], faces)

    for _ in range(iterations):
        vertices = laplacian_step(vertices, adjacency, lamb)
        vertices = laplacian_step(vertices, adjacency, mu)

    return vertices.astype(np.float32)


def mask_to_mesh_mc(
    mask,
    spacing,
    name="object",
    sigma_mm=0.5,
    level=0.25,
    min_size=100,
    keep_largest=False,
    closing_radius=0,
    crop_margin=6,
    smooth_iter=10
):
    mask = mask.astype(bool)

    if min_size > 0:
        mask = remove_small_3d(mask, min_size=min_size)

    if keep_largest:
        mask = keep_largest_component_3d(mask)

    if closing_radius > 0:
        mask = morphology.binary_closing(mask, morphology.ball(closing_radius))

    if mask.sum() == 0:
        raise ValueError(f"{name} mask 为空，无法重建")

    mn, mx = bbox_from_mask(mask, margin=crop_margin)

    crop = mask[
        mn[0]:mx[0],
        mn[1]:mx[1],
        mn[2]:mx[2]
    ]

    pad = 3
    crop_pad = np.pad(crop.astype(np.float32), pad, mode="constant", constant_values=0)

    sigma_vox = [
        max(float(sigma_mm) / float(spacing[0]), 0.01),
        max(float(sigma_mm) / float(spacing[1]), 0.01),
        max(float(sigma_mm) / float(spacing[2]), 0.01),
    ]

    print(f"\n{name} crop shape:", crop.shape)
    print(f"{name} sigma_mm={sigma_mm}, sigma_vox={sigma_vox}, level={level}")

    field = ndi.gaussian_filter(crop_pad, sigma=sigma_vox)

    if field.max() <= level:
        level = float(field.max() * 0.45)
        print(f"{name} 自动降低 marching cubes level 到 {level:.4f}")

    verts, faces, normals, values = measure.marching_cubes(
        field,
        level=level,
        spacing=spacing
    )

    offset_index = mn - pad
    offset_phys = np.asarray([
        offset_index[0] * spacing[0],
        offset_index[1] * spacing[1],
        offset_index[2] * spacing[2]
    ], dtype=np.float32)

    verts = verts + offset_phys[None, :]

    print(f"{name} marching cubes vertices={verts.shape[0]}, faces={faces.shape[0]}")

    verts = taubin_smooth(
        verts,
        faces,
        iterations=smooth_iter,
        lamb=0.45,
        mu=-0.50
    )

    print(f"{name} after smoothing vertices={verts.shape[0]}, faces={faces.shape[0]}")

    return verts.astype(np.float32), faces.astype(np.int32)


# ============================================================
# 7. STL / PLY 输出
# ============================================================

def write_binary_stl(path, vertices, faces, name="mesh"):
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)

    with open(path, "wb") as f:
        header = (name[:80]).ljust(80, " ").encode("ascii", errors="ignore")
        f.write(header)
        f.write(struct.pack("<I", faces.shape[0]))

        for tri in faces:
            v0, v1, v2 = vertices[tri]

            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)

            if norm > 1e-8:
                n = n / norm
            else:
                n = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


def write_colored_ply(path, vertices, faces, colors):
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)
    colors = colors.astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")

        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")

        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")

        f.write("end_header\n")

        for v, c in zip(vertices, colors):
            f.write(
                f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def save_models(
    out_dir,
    tumor_vertices,
    tumor_faces,
    vessel_vertices,
    vessel_faces
):
    tumor_stl = os.path.join(out_dir, "tumor_Ours_DF_MCCS.stl")
    vessel_stl = os.path.join(out_dir, "vessel_Ours_DF_PBVCI.stl")

    write_binary_stl(tumor_stl, tumor_vertices, tumor_faces, name="tumor_Ours_DF_MCCS")
    write_binary_stl(vessel_stl, vessel_vertices, vessel_faces, name="vessel_Ours_DF_PBVCI")

    tumor_color = np.tile(
        np.array([[40, 220, 45]], dtype=np.uint8),
        (tumor_vertices.shape[0], 1)
    )

    vessel_color = np.tile(
        np.array([[220, 40, 35]], dtype=np.uint8),
        (vessel_vertices.shape[0], 1)
    )

    tumor_ply = os.path.join(out_dir, "tumor_Ours_DF_MCCS_green.ply")
    vessel_ply = os.path.join(out_dir, "vessel_Ours_DF_PBVCI_red.ply")
    combined_ply = os.path.join(out_dir, "tumor_vessel_Ours_DF_MCCS_DF_PBVCI_colored.ply")

    write_colored_ply(tumor_ply, tumor_vertices, tumor_faces, tumor_color)
    write_colored_ply(vessel_ply, vessel_vertices, vessel_faces, vessel_color)

    combined_vertices = np.concatenate([tumor_vertices, vessel_vertices], axis=0)

    combined_faces = np.concatenate(
        [
            tumor_faces,
            vessel_faces + tumor_vertices.shape[0]
        ],
        axis=0
    )

    combined_colors = np.concatenate([tumor_color, vessel_color], axis=0)

    write_colored_ply(combined_ply, combined_vertices, combined_faces, combined_colors)

    return tumor_stl, vessel_stl, combined_ply


# ============================================================
# 8. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="/data/LLM/CT3D_project/datasets/3Dircadb_msd_like")
    parser.add_argument("--case", type=str, default="3Dircadb1.1")
    parser.add_argument("--out_root", type=str, default="/data/LLM/CT3D_project/results")

    parser.add_argument("--tumor_label", type=int, default=2)
    parser.add_argument("--vessel_label", type=int, default=1)

    parser.add_argument("--sparse_step", type=int, default=2)
    parser.add_argument("--keep_first_last", action="store_true")

    # DF-MCCS tumor
    parser.add_argument("--tumor_threshold", type=float, default=0.0)
    parser.add_argument("--tumor_tau_limit", type=float, default=1.5)
    parser.add_argument("--tumor_min_area_2d", type=int, default=10)

    # DF-P-BVCI vessel
    parser.add_argument("--vessel_threshold", type=float, default=0.0)
    parser.add_argument("--vessel_min_area_2d", type=int, default=5)
    parser.add_argument("--vessel_match_dist", type=float, default=28.0)

    parser.add_argument("--physio_theta_min", type=float, default=35.0)
    parser.add_argument("--physio_theta_max", type=float, default=52.0)
    parser.add_argument("--physio_ratio_min", type=float, default=1.15)
    parser.add_argument("--physio_ratio_max", type=float, default=1.40)
    parser.add_argument("--physio_bridge_scale", type=float, default=0.60)
    parser.add_argument("--max_groups_per_gap", type=int, default=10)

    # Mesh params
    parser.add_argument("--tumor_sigma", type=float, default=0.55)
    parser.add_argument("--tumor_level", type=float, default=0.28)
    parser.add_argument("--tumor_min_size_3d", type=int, default=200)
    parser.add_argument("--tumor_smooth_iter", type=int, default=12)

    parser.add_argument("--vessel_sigma", type=float, default=0.30)
    parser.add_argument("--vessel_level", type=float, default=0.18)
    parser.add_argument("--vessel_min_size_3d", type=int, default=25)
    parser.add_argument("--vessel_smooth_iter", type=int, default=10)

    args = parser.parse_args()

    label_path = os.path.join(args.root, "labelsTr", args.case + ".nii.gz")

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"找不到 label 文件: {label_path}")

    out_dir = os.path.join(
        args.out_root,
        f"{args.case}_Ours_DF_MCCS_DF_PBVCI_sparse{args.sparse_step}"
    )
    ensure_dir(out_dir)

    print("读取 label:", label_path)

    label, spacing, affine = load_label(label_path)

    print("label shape:", label.shape)
    print("spacing:", spacing)

    tumor_gt = label == args.tumor_label
    vessel_gt = label == args.vessel_label

    print("GT tumor voxels:", int(tumor_gt.sum()))
    print("GT vessel voxels:", int(vessel_gt.sum()))

    tumor_sparse = make_sparse_mask(
        tumor_gt,
        sparse_step=args.sparse_step,
        keep_first_last=args.keep_first_last
    )

    vessel_sparse = make_sparse_mask(
        vessel_gt,
        sparse_step=args.sparse_step,
        keep_first_last=args.keep_first_last
    )

    print("Sparse tumor voxels:", int(tumor_sparse.sum()))
    print("Sparse vessel voxels:", int(vessel_sparse.sum()))

    spacing_xy = (float(spacing[0]), float(spacing[1]))

    print("\n===== Ours: DF-MCCS tumor interpolation =====")
    tumor_ours = df_mccs_tumor_interpolation(
        tumor_sparse,
        spacing_xy=spacing_xy,
        threshold=args.tumor_threshold,
        tau_limit=args.tumor_tau_limit,
        min_area_2d=args.tumor_min_area_2d,
        fill_holes=True,
        closing=True
    )

    print("\n===== Ours: DF-P-BVCI vessel interpolation =====")
    vessel_ours = df_pbvci_vessel_interpolation(
        vessel_sparse,
        spacing_xy=spacing_xy,
        threshold=args.vessel_threshold,
        min_area_2d=args.vessel_min_area_2d,
        max_match_dist=args.vessel_match_dist,
        theta_range=(args.physio_theta_min, args.physio_theta_max),
        ratio_range=(args.physio_ratio_min, args.physio_ratio_max),
        bridge_scale=args.physio_bridge_scale,
        max_groups_per_gap=args.max_groups_per_gap,
        clean_small=True
    )

    print("Ours tumor voxels:", int(tumor_ours.sum()))
    print("Ours vessel voxels:", int(vessel_ours.sum()))

    # 保存 masks
    np.save(os.path.join(out_dir, "tumor_gt_mask.npy"), tumor_gt.astype(np.uint8))
    np.save(os.path.join(out_dir, "vessel_gt_mask.npy"), vessel_gt.astype(np.uint8))
    np.save(os.path.join(out_dir, "tumor_sparse_mask.npy"), tumor_sparse.astype(np.uint8))
    np.save(os.path.join(out_dir, "vessel_sparse_mask.npy"), vessel_sparse.astype(np.uint8))
    np.save(os.path.join(out_dir, "tumor_Ours_DF_MCCS_mask.npy"), tumor_ours.astype(np.uint8))
    np.save(os.path.join(out_dir, "vessel_Ours_DF_PBVCI_mask.npy"), vessel_ours.astype(np.uint8))

    save_nifti(tumor_ours, affine, os.path.join(out_dir, "tumor_Ours_DF_MCCS.nii.gz"))
    save_nifti(vessel_ours, affine, os.path.join(out_dir, "vessel_Ours_DF_PBVCI.nii.gz"))

    # 计算指标
    print("\n===== 计算 Ours 指标 =====")

    tumor_metrics = compute_metrics(
        pred=tumor_ours,
        gt=tumor_gt,
        spacing=spacing,
        name="tumor"
    )

    vessel_metrics = compute_metrics(
        pred=vessel_ours,
        gt=vessel_gt,
        spacing=spacing,
        name="vessel"
    )

    metrics = [tumor_metrics, vessel_metrics]

    csv_path = os.path.join(out_dir, "ours_df_mccs_df_pbvci_metrics.csv")
    save_metrics(metrics, csv_path)

    txt_path = os.path.join(out_dir, "ours_df_mccs_df_pbvci_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Ours: DF-MCCS + DF-P-BVCI report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"case: {args.case}\n")
        f.write(f"spacing: {spacing}\n")
        f.write(f"sparse_step: {args.sparse_step}\n")
        f.write(f"keep_first_last: {args.keep_first_last}\n\n")

        for row in metrics:
            f.write(f"[{row['structure']}]\n")
            for k, v in row.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    print("保存指标:", csv_path)
    print("保存报告:", txt_path)

    # 三维重建
    print("\n===== Ours 三维重建：肿瘤 =====")
    tumor_vertices, tumor_faces = mask_to_mesh_mc(
        tumor_ours,
        spacing=spacing,
        name="tumor_Ours_DF_MCCS",
        sigma_mm=args.tumor_sigma,
        level=args.tumor_level,
        min_size=args.tumor_min_size_3d,
        keep_largest=True,
        closing_radius=0,
        crop_margin=8,
        smooth_iter=args.tumor_smooth_iter
    )

    print("\n===== Ours 三维重建：血管 =====")
    vessel_vertices, vessel_faces = mask_to_mesh_mc(
        vessel_ours,
        spacing=spacing,
        name="vessel_Ours_DF_PBVCI",
        sigma_mm=args.vessel_sigma,
        level=args.vessel_level,
        min_size=args.vessel_min_size_3d,
        keep_largest=False,
        closing_radius=0,
        crop_margin=6,
        smooth_iter=args.vessel_smooth_iter
    )

    tumor_stl, vessel_stl, combined_ply = save_models(
        out_dir,
        tumor_vertices,
        tumor_faces,
        vessel_vertices,
        vessel_faces
    )

    info_path = os.path.join(out_dir, "ours_df_mccs_df_pbvci_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"case: {args.case}\n")
        f.write(f"label_path: {label_path}\n")
        f.write(f"spacing: {spacing}\n")
        f.write(f"sparse_step: {args.sparse_step}\n")
        f.write(f"tumor_gt_voxels: {int(tumor_gt.sum())}\n")
        f.write(f"vessel_gt_voxels: {int(vessel_gt.sum())}\n")
        f.write(f"tumor_sparse_voxels: {int(tumor_sparse.sum())}\n")
        f.write(f"vessel_sparse_voxels: {int(vessel_sparse.sum())}\n")
        f.write(f"tumor_ours_voxels: {int(tumor_ours.sum())}\n")
        f.write(f"vessel_ours_voxels: {int(vessel_ours.sum())}\n")
        f.write(f"tumor_stl: {tumor_stl}\n")
        f.write(f"vessel_stl: {vessel_stl}\n")
        f.write(f"combined_ply: {combined_ply}\n")

    print("\n全部完成！结果保存到:")
    print(out_dir)

    print("\n重点查看:")
    print(tumor_stl)
    print(vessel_stl)
    print(combined_ply)
    print(csv_path)

    print("\n3D Slicer 建议导入:")
    print("1. tumor_Ours_DF_MCCS.stl    颜色：绿色")
    print("2. vessel_Ours_DF_PBVCI.stl  颜色：红色")
    print("后续官网风格图再单独导出 liver / portalvein / venoussystem / artery。")


if __name__ == "__main__":
    main()
