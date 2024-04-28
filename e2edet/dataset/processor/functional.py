import numpy as np
import paddle
import paddle.nn.functional as F
from e2edet.utils.box_ops import box_xyxy_to_cxcywh
from e2edet.utils.det3d.box_ops import (
    mask_boxes_outside_range,
    rotate_points_along_z,
    limit_period,
)
from e2edet.utils.general import interpolate, normalize_period, data_to_tensor
from e2edet.utils.det3d.general import points_to_voxel, mask_points_by_range

# =========================== #
# --------- 2d ops ---------- #
# =========================== #


def resize_scale(sample, target, scale, target_height, target_width):
    def _get_resize(image_size, scale):
        w, h = image_size
        input_size = paddle.to_tensor([h, w])

        # Compute new target size given a scale
        target_size = paddle.to_tensor([target_height, target_width])
        target_scale_size = target_size * scale

        # Compute actual rescaling applied to input image and output size
        output_scale = paddle.minimum(
            target_scale_size[0] / input_size[0], target_scale_size[1] / input_size[1]
        )
        output_size = paddle.round(input_size * output_scale).astype('int32')
        oh, ow = output_size.tolist()

        return (ow, oh)

    size = _get_resize(sample["image"].shape[-2:], scale)

    return resize(sample, target, size)


def random_crop(sample, target, crop_size, is_fixed=True, pad_value=0):
    def _get_crop(image_size, crop_size, is_fixed):
        w, h = image_size
        ow, oh = crop_size

        # Add random crop if the image is scaled up
        max_offset = paddle.to_tensor([h, w]) - paddle.to_tensor([oh, ow])
        max_offset = paddle.clip(max_offset, min=0)

        offset = max_offset * paddle.uniform(0.0, 1.0)
        offset = paddle.round(offset).astype('int32').tolist()

        if is_fixed:
            return (offset[0], offset[1], oh, ow)

        return (offset[0], offset[1], min(oh, h), min(ow, w))

    size = _get_crop(sample["image"].shape[-2:], crop_size, is_fixed)

    if is_fixed:
        w, h = sample["image"].shape[-2:]
        ow, oh = crop_size

        pad_size = paddle.to_tensor([oh, ow]) - paddle.to_tensor([h, w])
        pad_size = paddle.clip(pad_size, min=0).tolist()
        sample, target = pad(
            sample, target, (pad_size[1], pad_size[0]), pad_value=pad_value
        )

    return crop(sample, target, size)


def crop(sample, target, region):
    """
    Crop region in the image. For 3D annotations, it considers their 2D projection on image.
    """
    # region: [y, x, h, w]
    cropped_image = F.crop(sample["image"], *region)

    i, j, h, w = region

    target = target.copy()
    target["size"] = paddle.to_tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]  # x1, y1, x2, y2
        max_size = paddle.to_tensor([w, h], dtype='float32')
        cropped_boxes = boxes - paddle.to_tensor([j, i, j, i])
        cropped_boxes = paddle.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = paddle.clip(cropped_boxes, min=0)

        area = (cropped_boxes[:, 1] - cropped_boxes[:, 0]).prod(axis=1)
        target["boxes"] = cropped_boxes.view(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target["boxes"].view(-1, 2, 2)
            keep = paddle.all(cropped_boxes[:, 1] > cropped_boxes[:, 0], axis=1)
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    sample["image"] = cropped_image

    return sample, target


def hflip(sample, target):
    flipped_image = F.hflip(sample["image"])

    w, h = sample["image"].shape[-2:]

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * paddle.to_tensor([-1, 1, -1, 1]) + paddle.to_tensor(
            [w, 0, w, 0]
        )
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = paddle.flip(target["masks"], [-1])

    sample["image"] = flipped_image

    return sample, target


def pad(sample, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(sample["image"], (0, 0, padding[0], padding[1]))

    if target is None:
        sample["image"] = padded_image

        return sample, None

    target = target.copy()
    target["size"] = paddle.to_tensor(padded_image.shape[-2:][::-1])
    if "masks" in target:
        target["masks"] = F.pad(
            target["masks"], (0, padding[0], 0, padding[1])
        )

    sample["image"] = padded_image

    return sample, target


def resize(sample, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def _get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_orig_size = float(min((w, h)))
            max_orig_size = float(max((w, h)))
            if max_orig_size / min_orig_size * size > max_size:
                size = int(round(max_size * min_orig_size / max_orig_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def _get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return _get_size_with_aspect_ratio(image_size, size, max_size)

    size = _get_size(sample["image"].shape[-2:], size, max_size)
    rescaled_image = F.resize(sample["image"], size)

    if target is None:
        sample["image"] = rescaled_image

        return sample, None

    ratios = tuple(
        float(s) / float(s_orig)
        for s, s_orig in zip(rescaled_image.shape[-2:], sample["image"].shape[-2:])
    )
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * paddle.to_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = paddle.to_tensor([h, w])

    if "masks" in target:
        target["masks"] = (
            interpolate(paddle.unsqueeze(target["masks"].astype('float32'), axis=1), size, mode="nearest")[:, 0]
            > 0.5
        )

    sample["image"] = rescaled_image

    return sample, target


def to_tensor(sample, target):
    sample["image"] = F.to_tensor(sample["image"])

    return sample, target


def normalize(sample, target, mean, std):
    sample["image"] = F.normalize(sample["image"], mean=mean, std=std)
    if target is None:
        return sample, None

    target = target.copy()
    h, w = sample["image"].shape[-2:]
    if "boxes" in target:
        boxes = target["boxes"]
        target["orig_boxes"] = boxes
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / paddle.to_tensor([w, h, w, h], dtype='float32')
        target["boxes"] = boxes

    return sample, target

# =========================== #
# --------- 3d ops ---------- #
# =========================== #

def double_flip(sample, target):
    # y flip
    points = sample["points"].copy()
    points[:, 1] = -points[:, 1]

    sample["yflip_points"] = points

    # x flip
    points = sample["points"].copy()
    points[:, 0] = -points[:, 0]

    sample["xflip_points"] = points

    # x y flip
    points = sample["points"].copy()
    points[:, 0] = -points[:, 0]
    points[:, 1] = -points[:, 1]

    sample["double_flip_points"] = points

    return sample, target


def global_rotation(sample, target, noise_rotation):
    sample["points"] = rotate_points_along_z(
        sample["points"][np.newaxis, :, :], np.array([noise_rotation])
    )[0]

    target = target.copy()
    if "boxes" in target:
        target["boxes"][:, :3] = rotate_points_along_z(
            target["boxes"][np.newaxis, :, :3], np.array([noise_rotation])
        )[0]
        target["boxes"][:, -1] += noise_rotation
        if target["boxes"].shape[1] > 7:
            target["boxes"][:, 6:8] = rotate_points_along_z(
                np.hstack(
                    [target["boxes"][:, 6:8], np.zeros((target["boxes"].shape[0], 1))]
                )[np.newaxis, :, :],
                np.array([noise_rotation]),
            )[0, :, :2]

    return sample, target


def global_scaling(sample, target, noise_scale):
    sample["points"][:, :3] *= noise_scale

    target = target.copy()
    if "boxes" in target:
        target["boxes"][:, :6] *= noise_scale

    return sample, target


def global_translate(sample, target, noise_translate):
    sample["points"][:, :3] += noise_translate

    target = target.copy()
    if "boxes" in target:
        target["boxes"][:, :3] += noise_translate

    return sample, target


def random_flip(sample, target, x_flip=False, y_flip=False):
    target = target.copy()
    if x_flip:
        sample["points"][:, 1] = -sample["points"][:, 1]

        if "boxes" in target:
            target["boxes"][:, 1] = -target["boxes"][:, 1]
            target["boxes"][:, -1] = -target["boxes"][:, -1]

            if target["boxes"].shape[1] > 7:
                target["boxes"][:, 7] = -target["boxes"][:, 7]

    if y_flip:
        sample["points"][:, 0] = -sample["points"][:, 0]

        if "boxes" in target:
            target["boxes"][:, 0] = -target["boxes"][:, 0]
            target["boxes"][:, -1] = -(target["boxes"][:, -1] + paddle.pi)

            if target["boxes"].shape[1] > 7:
                target["boxes"][:, 6] = -target["boxes"][:, 6]

    return sample, target


def shuffle_points(sample, target):
    np.random.shuffle(sample["points"])

    return sample, target


def voxelize(sample, target, voxel_size, pc_range, max_points_in_voxel, max_voxel_num):
    point_cloud_range = paddle.to_tensor(pc_range)
    voxel_size = paddle.to_tensor(voxel_size)
    grid_size = paddle.round((point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size).astype(paddle.int64)

    voxels, coordinates, num_points = points_to_voxel(
        sample["points"],
        voxel_size.numpy(),
        pc_range,
        max_points_in_voxel,
        True,
        max_voxel_num,
    )

    num_voxels = paddle.to_tensor([voxels.shape[0]])

    if "voxels" in sample:
        sample["coordinates"].append(coordinates)
        sample["voxels"].append(voxels)
        sample["num_voxels"].append(num_voxels)
        sample["num_points_per_voxel"].append(num_points)
        sample["grid_shape"].append(grid_size)
    else:
        sample.update(
            {
                "coordinates": [coordinates],
                "voxels": [voxels],
                "num_voxels": [num_voxels],
                "num_points_per_voxel": [num_points],
                "grid_shape": [grid_size],
            }
        )

    return sample, target


def filter_by_pc_range(sample, target, pc_range):
    target = target.copy()

    keep = mask_points_by_range(sample["points"], pc_range)
    sample["points"] = sample["points"][keep]

    if "boxes" in target:
        keep = mask_boxes_outside_range(target["boxes"], pc_range)
        target["boxes"] = target["boxes"][keep]
        target["labels"] = target["labels"][keep]

    return sample, target


def normalize3d(sample, target, pc_range, normalize_angle="sine"):
    target = target.copy()

    if "boxes" in target:
        pc_size = paddle.to_tensor(pc_range[3:] - pc_range[:3])

        target["boxes"][:, :3] -= pc_range[:3]
        target["boxes"][:, :3] /= pc_size
        target["boxes"][:, 3:6] /= pc_size
        target["boxes"][:, -1] = limit_period(
            target["boxes"][:, -1], offset=0.5, period=paddle.pi * 2
        )

        if normalize_angle == "sine":
            target["boxes"] = paddle.concat(
                [
                    target["boxes"][:, :6],
                    paddle.sin(target["boxes"][:, -1:]),
                    paddle.cos(target["boxes"][:, -1:]),
                ],
                axis=-1,
            )

            assert (
                ((target["boxes"][:, :6] >= 0) & (target["boxes"][:, :6] <= 1))
                .all()
                .numpy()
            )
        elif normalize_angle == "sigmoid":
            # TODO: temporarily remove velocity, need to add when move to tracking
            target["boxes"] = target["boxes"][:, [0, 1, 2, 3, 4, 5, -1]]

            target["boxes"][:, -1] = normalize_period(
                target["boxes"][:, -1], offset=0.5, period=paddle.pi * 2
            )

            assert (
                ((target["boxes"] >= 0) & (target["boxes"] <= 1)).all().numpy()
            ), "{}".format(
                target["boxes"][((target["boxes"] < 0) | (target["boxes"] > 1)).any(-1)]
            )

    return sample, target


def np_to_tensor(sample, target):
    sample = data_to_tensor(sample)
    target = data_to_tensor(target)

    return sample, target