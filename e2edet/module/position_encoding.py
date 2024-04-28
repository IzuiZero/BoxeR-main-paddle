import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from e2edet.utils.general import get_proposal_pos_embed


class FixedPositionEmbedding(nn.Layer):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super(FixedPositionEmbedding, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * 3.141592653589793
        self.scale = scale

    def forward(self, x, mask=None, ref_size=None):
        eps = 1e-6
        if mask is not None:
            not_mask = paddle.logical_not(mask)
            y_embed = not_mask.cumsum(axis=1, dtype=x.dtype)
            x_embed = not_mask.cumsum(axis=2, dtype=x.dtype)
        else:
            size_h, size_w = x.shape[-2:]
            y_embed = paddle.arange(1, size_h + 1, dtype=x.dtype)
            x_embed = paddle.arange(1, size_w + 1, dtype=x.dtype)
            y_embed, x_embed = paddle.meshgrid([y_embed, x_embed])
            x_embed = paddle.unsqueeze(x_embed, axis=0).tile([x.shape[0], 1, 1])
            y_embed = paddle.unsqueeze(y_embed, axis=0).tile([x.shape[0], 1, 1])

        if self.normalize:
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = paddle.arange(self.num_pos_feats, dtype=x.dtype)
        dim_t = self.temperature ** (
            2 * paddle.floor(dim_t.div(2) / self.num_pos_feats)
        )

        pos_x = x_embed.unsqueeze(3) / dim_t.unsqueeze(0).unsqueeze(0)
        pos_y = y_embed.unsqueeze(3) / dim_t.unsqueeze(0).unsqueeze(0)
        pos_x = paddle.concat(
            [paddle.sin(pos_x[:, :, :, 0::2]), paddle.cos(pos_x[:, :, :, 1::2])],
            axis=3,
        ).flatten(3)
        pos_y = paddle.concat(
            [paddle.sin(pos_y[:, :, :, 0::2]), paddle.cos(pos_y[:, :, :, 1::2])],
            axis=3,
        ).flatten(3)
        pos = paddle.concat([pos_x, pos_y], axis=3).transpose([0, 3, 1, 2])

        return pos


class FixedBoxEmbedding(nn.Layer):
    def __init__(self, hidden_dim, temperature=10000, normalize=False):
        super(FixedBoxEmbedding, self).__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, x, mask=None, ref_size=4):
        eps = 1e-6
        if mask is not None:
            not_mask = paddle.logical_not(mask)
            y_embed = not_mask.cumsum(axis=1, dtype=x.dtype)
            x_embed = not_mask.cumsum(axis=2, dtype=x.dtype)

            size_h = not_mask[:, :, 0].sum(axis=-1, dtype=x.dtype)
            size_w = not_mask[:, 0, :].sum(axis=-1, dtype=x.dtype)
        else:
            size_h, size_w = x.shape[-2:]
            y_embed = paddle.arange(1, size_h + 1, dtype=x.dtype)
            x_embed = paddle.arange(1, size_w + 1, dtype=x.dtype)
            y_embed, x_embed = paddle.meshgrid([y_embed, x_embed])
            x_embed = paddle.unsqueeze(x_embed, axis=0).tile([x.shape[0], 1, 1])
            y_embed = paddle.unsqueeze(y_embed, axis=0).tile([x.shape[0], 1, 1])

            size_h = paddle.full(
                [x.shape[0]], size_h, dtype=x.dtype
            )
            size_w = paddle.full(
                [x.shape[0]], size_w, dtype=x.dtype
            )

        if self.normalize:
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps)
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps)

        h_embed = ref_size / size_h
        w_embed = ref_size / size_w

        h_embed = paddle.unsqueeze(h_embed, axis=1).unsqueeze(2).tile_like(x_embed)
        w_embed = paddle.unsqueeze(w_embed, axis=1).unsqueeze(2).tile_like(x_embed)

        center_embed = paddle.stack([x_embed, y_embed], axis=-1)
        size_embed = paddle.stack([w_embed, h_embed], axis=-1)
        center = get_proposal_pos_embed(center_embed, self.hidden_dim)
        size = get_proposal_pos_embed(size_embed, self.hidden_dim)
        box = center + size

        return box.transpose([0, 3, 1, 2])


def build_position_encoding(position_embedding_type, hidden_dim):
    if position_embedding_type == "fixed":
        N_steps = hidden_dim // 2
        # TODO find a better way of exposing other arguments
        position_embedding = FixedPositionEmbedding(N_steps, normalize=True)
    elif position_embedding_type == "fixed_box":
        position_embedding = FixedBoxEmbedding(hidden_dim, normalize=True)
    else:
        raise ValueError(f"not supported {position_embedding_type}")

    return position_embedding