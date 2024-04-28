import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from e2edet.utils.det3d.general import get_paddings_indicator


class PointPillarsScatter(nn.Layer):
    def __init__(self, num_input_features=64):
        super().__init__()
        self.num_channels = num_input_features

    def forward(self, voxel_features, coords, batch_size, input_shape):
        self.nx = input_shape[0].item()
        self.ny = input_shape[1].item()

        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = paddle.zeros(
                [self.num_channels, self.nx * self.ny],
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            indices = (this_coords[:, 2] * self.nx + this_coords[:, 3]).astype('int64')
            voxels = voxel_features[batch_mask, :]
            voxels = paddle.transpose(voxels, perm=[1, 0])

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        batch_canvas = paddle.stack(batch_canvas, axis=0)
        batch_canvas = batch_canvas.view(
            batch_size, self.num_channels, self.ny, self.nx
        )

        return batch_canvas


class PFNLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, norm_layer=None, last_layer=False):
        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2

        self.linear = nn.Linear(in_channels, out_channels, bias_attr=False)

        if norm_layer is None:
            self.norm = nn.SyncBatchNorm(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.norm = norm_layer(out_channels)
        self._reset_parameters()

    def _reset_parameters(self):
        self.linear.weight.set_value(paddle.nn.initializer.XavierUniform())

    def forward(self, inputs):
        x = self.linear(inputs)
        paddle.set_device('cpu')
        x = self.norm(paddle.transpose(x, perm=[0, 2, 1])).transpose([0, 2, 1])
        paddle.set_device('gpu')
        x = F.relu(x)

        x_max = paddle.max(x, axis=1, keepdim=True)

        if self.last_vfe:
            return x_max

        x_repeat = paddle.tile(x_max, [1, inputs.shape[1], 1])
        x_concatenated = paddle.concat([x, x_repeat], axis=2)
        return x_concatenated


class PillarFeatureNet(nn.Layer):
    def __init__(
        self,
        num_input_features=4,
        num_filters=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        pc_range=(0, -40, -3, 70.4, 40, 1),
        norm_layer=None,
    ):
        super().__init__()
        assert len(num_filters) > 0

        self.num_input = num_input_features
        num_input_features += 5
        if with_distance:
            num_input_features += 1

        self.with_distance = with_distance

        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_layer=norm_layer,
                    last_layer=last_layer,
                )
            )
        self.pfn_layers = nn.LayerList(pfn_layers)

        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        points_mean = paddle.sum(features[:, :, :3], axis=1, keepdim=True) / paddle.cast(
            num_voxels.view(-1, 1, 1), features.dtype
        )
        f_cluster = features[:, :, :3] - points_mean

        f_center = paddle.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].astype(features.dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].astype(features.dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        features_ls = [features, f_cluster, f_center]
        if self.with_distance:
            points_dist = paddle.norm(features[:, :, :3], p=2, axis=2, keepdim=True)
            features_ls.append(points_dist)
        features = paddle.concat(features_ls, axis=-1)

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = paddle.unsqueeze(mask, axis=-1).astype(features.dtype)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()
