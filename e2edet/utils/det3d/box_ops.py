import numpy as np
import paddle

from e2edet.utils.det3d.geometry import points_in_convex_polygon_3d_jit


def box_cxcyczlwh_to_xyxyxy(x):
    x_c, y_c, z_c, l, w, h = paddle.unbind(x, axis=-1)
    b = [
        (x_c - 0.5 * l),
        (y_c - 0.5 * w),
        (z_c - 0.5 * h),
        (x_c + 0.5 * l),
        (y_c + 0.5 * w),
        (z_c + 0.5 * h),
    ]

    return paddle.stack(b, axis=-1)


def box_vol_wo_angle(boxes):
    vol = (
        (boxes[:, 3] - boxes[:, 0])
        * (boxes[:, 4] - boxes[:, 1])
        * (boxes[:, 5] - boxes[:, 2])
    )

    return vol


def box_intersect_wo_angle(boxes1, boxes2):
    ltb = paddle.maximum(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rbf = paddle.minimum(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    lwh = paddle.clip(rbf - ltb, min=0)  # [N,M,3]
    inter = lwh[:, :, 0] * lwh[:, :, 1] * lwh[:, :, 2]  # [N,M]

    return inter


def box_iou_wo_angle(boxes1, boxes2):
    vol1 = box_vol_wo_angle(boxes1)
    vol2 = box_vol_wo_angle(boxes2)
    inter = box_intersect_wo_angle(boxes1, boxes2)

    union = vol1[:, None] + vol2 - inter
    iou = inter / union

    return iou, union


def generalized_box3d_iou(boxes1, boxes2):
    assert paddle.all(boxes1[:, 3:] >= boxes1[:, :3])
    assert paddle.all(boxes2[:, 3:] >= boxes2[:, :3])

    iou, union = box_iou_wo_angle(boxes1, boxes2)

    ltb = paddle.minimum(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rbf = paddle.maximum(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    whl = paddle.clip(rbf - ltb, min=0)  # [N,M,3]
    vol = whl[:, :, 0] * whl[:, :, 1] * whl[:, :, 2]

    return iou - (vol - union) / vol


def rotate_points_along_z(points, angle):
    if isinstance(points, np.ndarray):
        points = paddle.to_tensor(points, dtype='float32')
        angle = paddle.to_tensor(angle, dtype='float32')

    cosa = paddle.cos(angle)
    sina = paddle.sin(angle)
    zeros = paddle.zeros_like(angle)
    ones = paddle.ones_like(angle)
    rot_matrix = paddle.stack(
        [cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones], axis=1
    ).reshape([-1, 3, 3]).astype('float32')
    points_rot = paddle.matmul(points[:, :, :3], rot_matrix)
    points_rot = paddle.concat([points_rot, points[:, :, 3:]], axis=-1)

    return points_rot.numpy()


def boxes_to_corners_3d(boxes3d):
    template = (
        np.array(
            (
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, 1, 1],
            )
        ).astype(boxes3d.dtype)
        / 2
    )

    corners3d = boxes3d[:, None, 3:6] * template[None, :, :]
    corners3d = rotate_points_along_z(
        corners3d.reshape(-1, 8, 3), boxes3d[:, 6]
    ).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, :3]

    return corners3d


def mask_boxes_outside_range(boxes, limit_range, min_num_corners=8):
    if boxes.shape[1] > 7:
        boxes = boxes[:, [0, 1, 2, 3, 4, 5, -1]]

    corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
    mask = ((corners >= limit_range[:3]) & (corners <= limit_range[3:])).all(axis=-1)
    mask = mask.sum(axis=1) >= min_num_corners

    return mask


def limit_period(val, offset=0.5, period=np.pi):
    if isinstance(val, np.ndarray):
        val = paddle.to_tensor(val, dtype='float32')

    val = val - paddle.floor(val / period + offset) * period

    if not paddle.all((val >= -offset * period) & (val <= offset * period)):
        val = paddle.clip(val, min=-offset * period, max=offset * period)

    return val.numpy()


def corners_nd(dims, origin=0.5):
    ndim = int(dims.shape[1])
    corners_norm = paddle.stack(
        paddle.unravel_index(paddle.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - paddle.to_tensor(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])

    return corners.numpy()


def rotation_2d(points, angles):
    rot_sin = paddle.sin(angles)
    rot_cos = paddle.cos(angles)

    rot_mat_T = paddle.stack([rot_cos, rot_sin, -rot_sin, rot_cos]).reshape([2, 2, -1])

    return paddle.einsum("aij,jka->aik", points, rot_mat_T).numpy()


def rotation_3d(points, angles):
    rot_sin = paddle.sin(angles)
    rot_cos = paddle.cos(angles)
    ones = paddle.ones_like(rot_cos)
    zeros = paddle.zeros_like(rot_cos)

    rot_mat_T = paddle.stack(
        [[rot_cos, rot_sin, zeros], [-rot_sin, rot_cos, zeros], [zeros, zeros, ones],]
    )

    return paddle.einsum("aij,jka->aik", points, rot_mat_T).numpy()


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    corners = corners_nd(dims, origin=origin)

    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])

    return corners


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 0.5, 0.5), axis=2):
    corners = corners_nd(dims, origin=origin)

    if angles is not None:
        corners = rotation_3d(corners, angles)
    corners += centers.reshape([-1, 1, 3])

    return corners


def corner_to_surfaces_3d(corners):
    surfaces = np.array(
        [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ]
    ).transpose([2, 0, 1, 3])
    return surfaces


def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)

    return indices
