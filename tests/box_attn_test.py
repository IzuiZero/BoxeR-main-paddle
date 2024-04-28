import paddle
import numpy as np

from e2edet.module.ops import BoxAttnFunction
from e2edet.utils.general import view_with_shape


def PlainBoxAttnFunction(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    Params:
    :value: (B, L2, C)
    :value_spatial_shapes: (N, 2)
    :sampling_locations: (B, L1, nheads, nlevels, npoints, 2)
    :attention_weights: (B, L1, nheads, nlevels, npoints)

    Return:
    :output: (B, L1, C)
    """
    b, l1, nheads, nlevels, npoints = attention_weights.shape
    value, _ = view_with_shape(value, None, value_spatial_shapes)

    v_samples = []
    for level in range(nlevels):
        h, w = value[level].shape[2:]

        sampled_v = value[level].reshape(b * nheads, -1, h, w)
        grid = sampling_locations[:, :, :, level].transpose(1, 2)
        grid = grid.contiguous().reshape(b * nheads, l1, npoints, 2)

        sampled_v = F.grid_sample(sampled_v, grid, align_corners=False)
        sampled_v = sampled_v.reshape(b, nheads, -1, l1, npoints).permute(0, 3, 1, 2, 4)
        sampled_v = (attention_weights[:, :, :, level].unsqueeze(-2) * sampled_v).sum(
            dim=-1
        )
        v_samples.append(sampled_v)
    v_samples = paddle.stack(v_samples, axis=1).sum(axis=1).contiguous()
    v_samples = v_samples.reshape(b, l1, -1)

    return v_samples


N, M, D = 1, 2, 2
Lq, L, P = 2, 2, 2
shapes = paddle.to_tensor([(6, 4), (3, 2)], dtype='int64').cuda()
level_start_index = paddle.concat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H * W).item() for H, W in shapes])


paddle.manual_seed(3)


@paddle.no_grad()
def check_forward(tensor_type="float"):
    value = paddle.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = paddle.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = paddle.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)

    im2col_step = 2
    if tensor_type == "double":
        value = value.astype('float64')
        sampling_locations = sampling_locations.astype('float64')
        attention_weights = attention_weights.astype('float64')

    output_pytorch = (
        PlainBoxAttnFunction(
            value.reshape(N, S, -1), shapes, 2 * sampling_locations - 1, attention_weights
        )
        .detach()
        .cpu()
    )
    output_cuda = (
        BoxAttnFunction.apply(
            value,
            shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        .detach()
        .cpu()
    )
    forward_check = paddle.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(
        f"{forward_check.numpy()} check_forward (tensor_type: {tensor_type}): max_abs_err {max_abs_err.numpy():.2e}, max_rel_err {max_rel_err.numpy():.2e}"
    )


def check_forward_and_backward(tensor_type="double"):
    raise ValueError("only work with double tensor type")


def check_gradient_numerical(
    channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True
):
    raise ValueError("not implemented for PaddlePaddle yet")


if __name__ == "__main__":
    try:
        for channels in [30, 32, 64, 71, 1025, 2048, 3096]:
            check_gradient_numerical(channels, True, True, True)
    except Exception as e:
        print(e)

    check_forward("float")
    check_forward("double")
    check_forward_and_backward()
