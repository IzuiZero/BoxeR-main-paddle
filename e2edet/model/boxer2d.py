import math
import paddle.nn as nn
import paddle.nn.functional as F

from e2edet.model import BaseDetectionModel, register_model
from e2edet.module import build_resnet, build_transformer, Detector
from e2edet.utils.general import filter_grads
from e2edet.utils.modeling import get_parameters


@register_model("boxer2d")
class BoxeR2D(BaseDetectionModel):
    def __init__(self, config, num_classes, **kwargs):
        super().__init__(config, num_classes, global_config=kwargs["global_config"])
        self.hidden_dim = config["hidden_dim"]
        self.deform_lr_multi = config["deform_lr_multi"]
        self.num_level = config["transformer"]["params"]["nlevel"]
        self.use_mask = config["use_mask"]
        aux_loss = config["aux_loss"]

        if self.use_mask:
            enc_mask_mode = "none"
            dec_mask_mode = "mask_v1"
        else:
            enc_mask_mode = "none"
            dec_mask_mode = "none"
        self.enc_detector = Detector(
            self.hidden_dim, 1, False, True, mask_mode=enc_mask_mode
        )
        self.detector = Detector(
            self.hidden_dim, num_classes, aux_loss, True, mask_mode=dec_mask_mode
        )

    def get_optimizer_parameters(self):
        backbone_groups = []
        transformer_groups = []

        backbone_param_group = {"params": filter_grads(self.backbone.parameters())}
        backbone_groups.append(backbone_param_group)

        transformer_param_group = get_parameters(
            self,
            lr_multi=self.deform_lr_multi,
            lr_module=["linear_box"],
            lr_except=["backbone"],
        )
        transformer_groups.extend(transformer_param_group)

        return (backbone_groups, transformer_groups)

    def _build(self):
        self.backbone = build_resnet(self.config.backbone)
        self.transformer = build_transformer(self.config.transformer)

        num_backbone_outs = len(self.backbone.return_layers)
        input_proj_list = []

        for i in range(num_backbone_outs):
            in_channels = self.backbone.num_channels[i]
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )
            )

        for _ in range(self.num_level - num_backbone_outs):
            input_proj_list.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1
                    ),
                    nn.GroupNorm(32, self.hidden_dim),
                )
            )
            in_channels = self.hidden_dim
        self.input_proj = nn.LayerList(input_proj_list)
        self._reset_parameters()

        self.transformer.encoder.detector = [self.enc_detector]

    def _reset_parameters(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        nn.initializer.Constant(self.detector.class_embed.bias, bias_value)
        nn.initializer.Constant(self.detector.bbox_embed.layers[-1].weight, 0)
        nn.initializer.Constant(self.detector.bbox_embed.layers[-1].bias, 0)

        nn.initializer.Constant(self.enc_detector.class_embed.bias, bias_value)
        nn.initializer.Constant(self.enc_detector.bbox_embed.layers[-1].weight, 0)
        nn.initializer.Constant(self.enc_detector.bbox_embed.layers[-1].bias, 0)

        for module in self.input_proj.sublayers():
            if isinstance(module, nn.Conv2D):
                nn.initializer.XavierUniform(module.weight, gain=1)
                if module.bias is not None:
                    nn.initializer.Constant(module.bias, 0)

    def forward(self, sample, target=None):
        out, pos = self.backbone(sample["image"], sample["mask"])

        if self.num_level < len(out):
            raise RuntimeError(
                "num_level should be greater than or equal to the number of output features "
                "(num_level: {}, # output features: {})".format(
                    self.num_level, len(out)
                )
            )

        features = []
        masks = []
        pos_encodings = []

        for i, (src, mask) in enumerate(out):
            features.append(self.input_proj[i](src))
            pos_encodings.append(pos[i])
            masks.append(mask)

        idx = len(features)
        for i in range(idx, self.num_level):
            if i == idx:
                feat = self.input_proj[i](out[-1][0])
            else:
                feat = self.input_proj[i](F.relu(features[-1]))

            mask = sample["mask"]
            if mask is not None:
                mask = F.interpolate(mask.unsqueeze(0).astype('float32'), size=feat.shape[-2:])[0].astype('bool')

            if self.backbone.position_encoding is not None:
                ref_size = self.backbone.ref_size
                pos_encodings.append(
                    self.backbone.position_encoding(feat, mask, ref_size).astype(feat.dtype)
                )
            else:
                pos_encodings.append(None)
            features.append(feat)
            masks.append(mask)

        outputs = self.transformer(features, masks, pos_encodings)
        hidden_state, roi, ref_windows, src_embed, src_ref_windows, src_mask = outputs

        out = (
            self.detector(hidden_state, ref_windows, roi)
            if self.use_mask
            else self.detector(hidden_state, ref_windows)
        )

        if not self.inferencing:
            ref_windows_valid = (
                (src_ref_windows[..., :2] > 0.01) & (src_ref_windows[..., :2] < 0.99)
            ).all(-1)
            if src_mask is not None:
                src_mask = src_mask | (~ref_windows_valid)
            else:
                src_mask = ~ref_windows_valid
            src_embed = src_embed.masked_fill(src_mask.unsqueeze(-1), 0.0)
            src_ref_windows = src_ref_windows.masked_fill(src_mask.unsqueeze(-1), 0.0)

            enc_out = self.enc_detector(
                src_embed.unsqueeze(0), src_ref_windows.unsqueeze(0), x_mask=src_mask.unsqueeze(0)
            )
            out["enc_outputs"] = [
                {
                    "pred_logits": enc_out["pred_logits"],
                    "pred_boxes": enc_out["pred_boxes"],
                }
            ]

        return out