import collections
import re
import paddle
import numpy as np

np_str_obj_array_pattern = re.compile(r"[SaUO]")

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)

def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, paddle.Tensor):
        out = None
        return paddle.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([paddle.to_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return paddle.to_tensor(batch)
    elif isinstance(elem, float):
        return paddle.to_tensor(batch, dtype='float64')
    elif isinstance(elem, int):
        return paddle.to_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def _collate_sample2d(sample):
    assert sample[0]["image"].ndim == 3

    if len(sample) == 1:
        return {"image": sample[0]["image"][None], "mask": None}

    new_sample = {}
    total_shape = (sample[i]["image"].shape for i in range(len(sample)))
    shape = (len(sample), *(max(elem) for elem in zip(*total_shape)))
    new_sample["image"] = paddle.zeros(shape)
    b, h, w = shape[0], shape[2], shape[3]
    new_sample["mask"] = paddle.ones((b, h, w)).astype('bool')

    for i, elem in enumerate(sample):
        c, h, w = elem["image"].shape
        new_sample["image"][i, :c, :h, :w].copy_(elem["image"])
        new_sample["mask"][i, :h, :w] = False

    return new_sample

def collate2d(batch):
    batch = list(zip(*batch))

    iter_per_update = batch[0][0].get("iter_per_update", 1)
    if iter_per_update == 1:
        new_batch = [_collate_sample2d(batch[0]), batch[1]]
    elif iter_per_update > 1:
        sample = batch[0]
        target = batch[1]

        batch_size = len(sample)

        assert batch_size % iter_per_update == 0
        split_size = batch_size // iter_per_update

        new_batch = [
            (
                _collate_sample2d(sample[i * split_size : (i + 1) * split_size]),
                target[i * split_size : (i + 1) * split_size],
            )
            for i in range(iter_per_update)
        ]
    else:
        raise ValueError("iter_per_update should be greater than or equal to 1")

    return new_batch

def _collate_sample3d(sample):
    new_sample = {}

    fields = sample[0].keys()
    num_grid = len(sample[0]["voxels"])
    for field in fields:
        if sample[0][field] is None:
            new_sample[field] = None
            continue

        if field in ("voxels", "num_points_per_voxel", "num_voxels"):
            for i in range(num_grid):
                if num_grid == 1:
                    new_sample[field] = paddle.concat(
                        [elem[field][i] for elem in sample], axis=0
                    )
                else:
                    new_sample[field + f"_{i}"] = paddle.concat(
                        [elem[field][i] for elem in sample], axis=0
                    )
        elif field == "coordinates":
            for i in range(num_grid):
                batch_idx = paddle.concat(
                    [
                        paddle.ones(elem[field][i].shape[0], dtype=elem[field][i].dtype)
                        * j
                        for j, elem in enumerate(sample)
                    ],
                    axis=0,
                ).unsqueeze(1)
                data = paddle.concat([elem[field][i] for elem in sample], axis=0)

                if num_grid == 1:
                    new_sample[field] = paddle.concat([batch_idx, data], axis=1)
                else:
                    new_sample[field + f"_{i}"] = paddle.concat([batch_idx, data], axis=1)
        elif field in ["points", "calib", "iter_per_update"]:
            continue
        elif field == "grid_shape":
            for i in range(num_grid):
                if num_grid == 1:
                    new_sample[field] = paddle.stack(
                        [elem[field][i] for elem in sample], axis=0
                    )
                else:
                    new_sample[field + f"_{i}"] = paddle.stack(
                        [elem[field][i] for elem in sample], axis=0
                    )
        else:
            new_sample[field] = paddle.stack([elem[field] for elem in sample], axis=0)
    new_sample["num_grid"] = num_grid
    new_sample["batch_size"] = len(sample)

    return new_sample

def collate3d(batch):
    batch = list(zip(*batch))

    iter_per_update = batch[0][0].get("iter_per_update", 1)
    if iter_per_update == 1:
        new_batch = [_collate_sample3d(batch[0]), batch[1]]
    elif iter_per_update > 1:
        sample = batch[0]
        target = batch[1]

        batch_size = len(sample)

        assert batch_size % iter_per_update == 0
        split_size = batch_size // iter_per_update

        new_batch = [
            (
                _collate_sample3d(sample[i * split_size : (i + 1) * split_size]),
                target[i * split_size : (i + 1) * split_size],
            )
            for i in range(iter_per_update)
        ]
    else:
        raise ValueError("iter_per_update should be greater than or equal to 1")

    return new_batch