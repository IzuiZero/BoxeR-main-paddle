import paddle
from paddle.autograd import Function
from paddle.autograd.functional import once_differentiable
from paddle.fluid.layers import OpsElementwise

from e2edet import ops


class BoxAttnFunction(Function):
    @staticmethod
    @OpsElementwise.cast_inputs(torch.float32)
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = ops.box_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )

        return output

    @staticmethod
    @OpsElementwise.once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = ops.box_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class InstanceAttnFunction(Function):
    @staticmethod
    @OpsElementwise.cast_inputs(torch.float32)
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        spatial_attention_weights,
        level_attention_weights,
        mask_size,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output, mask_output = ops.instance_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
            im2col_step,
        )

        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
        )

        b, l, _, c = mask_output.shape
        mask_output = mask_output.reshape(b, l, mask_size, mask_size, c)

        return output, mask_output

    @staticmethod
    @OpsElementwise.once_differentiable
    def backward(ctx, grad_output, grad_mask_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if not grad_mask_output.is_contiguous():
            grad_mask_output = grad_mask_output.contiguous()

        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
        ) = ctx.saved_tensors

        (
            grad_value,
            grad_sampling_loc,
            grad_spatial_attn_weight,
            grad_level_attn_weight,
        ) = ops.instance_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            spatial_attention_weights,
            level_attention_weights,
            grad_output,
            grad_mask_output,
            ctx.im2col_step,
        )

        return (
            grad_value,
            None,
            None,
            grad_sampling_loc,
            grad_spatial_attn_weight,
            grad_level_attn_weight,
            None,
            None,
        )