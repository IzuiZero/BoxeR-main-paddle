import os
import pickle
import paddle
import numba
import numpy as np


def is_array_like(x):
    return isinstance(x, (list, tuple, np.ndarray))


def shape_mergeable(x, expected_shape):
    mergeable = True

    if is_array_like(x) and is_array_like(expected_shape):
        x = np.array(x)
        if len(x.shape) == len(expected_shape):
            for s, s_ex in zip(x.shape, expected_shape):
                if s_ex is not None and s != s_ex:
                    mergeable = False
                    break
    return mergeable


def mask_points_by_range(points, pc_range):
    mask = (
        (points[:, 0] >= pc_range[0])
        & (points[:, 0] <= pc_range[3])
        & (points[:, 1] >= pc_range[1])
        & (points[:, 1] <= pc_range[4])
        & (points[:, 2] >= pc_range[2])
        & (points[:, 2] <= pc_range[5])
    )
    return mask


def read_from_file(info, nsweeps=1):
    path = info["path"]
    with open(path, "rb") as f:
        obj = pickle.load(f)

    points = read_single_waymo(obj)
    times = None

    if nsweeps > 1:
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        assert (nsweeps - 1) == len(info["sweeps"]), "nsweeps {} should be equal to the list length {}.".format(nsweeps, len(info["sweeps"]))

        for i in range(nsweeps - 1):
            sweep = info["sweeps"][i]
            points_sweep, times_sweep = read_single_waymo_sweep(sweep)

            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

    if times is not None:
        points = np.hstack([points, times])

    return points


def read_pc_annotations(info):
    if "gt_boxes" in info:
        annos = {
            "boxes": info["gt_boxes"].astype(np.float32),
            "names": info["gt_names"],
            "difficulty": info["difficulty"].astype(np.int8),
            "num_points_in_gt": info["num_points_in_gt"].astype(np.int64),
        }
        return annos
    return None


def _read_file(path, num_points=4, painted=False):
    if painted:
        dir_path = os.path.join(*path.split("/")[:-2], "painted_" + path.split("/")[-2])
        painted_path = os.path.join(dir_path, path.split("/")[-1] + ".npy")
        points = np.load(painted_path)
        points = points[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]  # remove ring_index from features
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_points]
    return points


def _remove_close(points, radius: float):
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, painted=False):
    min_distance = 1.0
    points_sweep = _read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = _remove_close(points_sweep, min_distance)

    num_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

    cur_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    return points_sweep.T, cur_times.T


def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points = np.concatenate([points_xyz, points_feature], axis=-1)
    return points


def read_single_waymo_sweep(sweep):
    with open(sweep["path"], "rb") as f:
        obj = pickle.load(f)

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T

    num_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

    cur_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    return points_sweep.T, cur_times.T


def get_paddings_indicator(actual_num, max_num, axis=0):
    actual_num = paddle.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = paddle.arange(max_num, dtype=paddle.int64).reshape(max_num_shape)
    paddings_indicator = actual_num.cast(paddle.int64) > max_num
    return paddings_indicator


def points_to_voxel_cuda(
    points, voxel_size, coord_range, max_points=35, max_voxel=20000, reverse=True,
):
    grid_shape = (coord_range[3:] - coord_range[:3]) / voxel_size
    grid_shape = paddle.cast(paddle.round(grid_shape), dtype='int32')

    points_to_voxel_coord = paddle.floor(
        (paddle.to_tensor(points[:, :3]) - coord_range[:3]) / voxel_size
    ).astype('int32')
    assert (
        (points_to_voxel_coord >= 0).all() & (points_to_voxel_coord < grid_shape).all()
    ).numpy().all()

    points_to_voxel_idx = (
        points_to_voxel_coord[:, 0] * grid_shape[1] * grid_shape[2] +
        points_to_voxel_coord[:, 1] * grid_shape[2] +
        points_to_voxel_coord[:, 2]
    )
    sorted_indices = paddle.argsort(points_to_voxel_idx)
    points_to_voxel_idx = points_to_voxel_idx[sorted_indices]
    points_to_voxel_coord = points_to_voxel_coord[sorted_indices]
    points = points[sorted_indices]

    unique_voxel_idx, voxel_idx_count = paddle.unique(
        points_to_voxel_idx, return_counts=True, sorted=True
    )
    num_voxels = unique_voxel_idx.shape[0]
    num_voxels = min(max_voxel, num_voxels)

    voxels = paddle.zeros(
        (num_voxels, max_points, points.shape[-1]),
        dtype=points.dtype
    )
    coords = paddle.zeros((num_voxels, 3), dtype=paddle.int32)
    num_points_per_voxel = paddle.zeros((num_voxels,), dtype=paddle.int32)

    points_idx = 0
    for i in range(num_voxels):
        num_points = voxel_idx_count[i].item()

        num_points_in_voxel = min(num_points, max_points)
        voxels[i, :num_points_in_voxel] = points[
            points_idx : points_idx + num_points_in_voxel
        ]
        if reverse:
            coords[i] = paddle.flip(points_to_voxel_coord[points_idx], axis=[0])
        else:
            coords[i] = points_to_voxel_coord[points_idx]
        num_points_per_voxel[i] = num_points_in_voxel
        points_idx += num_points

    return voxels, coords, num_points_per_voxel