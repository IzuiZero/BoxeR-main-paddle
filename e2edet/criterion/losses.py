import copy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from e2edet.utils import box_ops
from e2edet.utils.det3d import box_ops as box3d_ops
from e2edet.utils.distributed import get_world_size, is_dist_avail_and_initialized
from e2edet.module.matcher import build_matcher
from e2edet.utils.general import extract_grid

LOSS_REGISTRY = {}


def build_loss(loss_config, num_classes, iter_per_update):
    if loss_config["type"] not in LOSS_REGISTRY:
        raise ValueError("Loss ({}) is not found.".format(loss_config["type"]))

    loss_cls = LOSS_REGISTRY[loss_config["type"]]
    params = loss_config["params"]
    weight_dict = {
        "loss_ce": params["class_loss_coef"],
        "loss_bbox": params["bbox_loss_coef"],
        "loss_giou": params["giou_loss_coef"],
    }

    if loss_config["type"] == "detr":
        losses = ["boxes", "labels"]

        loss_param = dict(
            num_classes=num_classes,
            matcher=build_matcher(params["matcher"]),
            weight_dict=weight_dict,
            eos_coef=params["eos_coef"],
            losses=losses,
            iter_per_update=iter_per_update,
        )
    elif loss_config["type"] == "boxer2d":
        losses = ["boxes", "focal_labels"]
        if params["use_mask"]:
            weight_dict["loss_mask"] = params["mask_loss_coef"]
            weight_dict["loss_dice"] = params["dice_loss_coef"]
            losses.append("masks")

        loss_param = dict(
            num_classes=num_classes,
            matcher=build_matcher(params["matcher"]),
            weight_dict=weight_dict,
            losses=losses,
            iter_per_update=iter_per_update,
        )
    elif loss_config["type"] == "boxer3d":
        losses = ["boxes", "focal_labels"]
        weight_dict["loss_rad"] = params["rad_loss_coef"]

        loss_param = dict(
            num_classes=num_classes,
            matcher=build_matcher(params["matcher"]),
            weight_dict=weight_dict,
            losses=losses,
            iter_per_update=iter_per_update,
        )
    else:
        raise ValueError(
            "Only detr|boxer2d|boxer3d losses are supported (found {})".format(
                loss_config["type"]
            )
        )

    module_loss = loss_cls(**loss_param)

    return module_loss


def register_loss(name):
    def register_loss_cls(cls):
        if name in LOSS_REGISTRY:
            raise ValueError("Cannot register duplicate loss ({})".format(name))

        LOSS_REGISTRY[name] = cls
        return cls

    return register_loss_cls


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):
    prob = F.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return paddle.sum(loss) / num_boxes


def dice_loss(inputs, targets, num_boxes):
    inputs = F.sigmoid(inputs)
    inputs = paddle.flatten(inputs, start_axis=1)
    targets = paddle.flatten(targets, start_axis=1)

    numerator = 2 * paddle.sum(inputs * targets, axis=1)
    denominator = paddle.sum(inputs, axis=-1) + paddle.sum(targets, axis=-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return paddle.sum(loss) / num_boxes


class BaseLoss(nn.Layer):
    def __init__(self, name, params=None):
        super().__init__()
        self.name = name
        self.params = params
        for kk, vv in params.items():
            setattr(self, kk, vv)

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for kk, vv in self.params.items():
            format_string += "{}={},".format(kk, vv)
        format_string += ")"

        return format_string


class LabelLoss(BaseLoss):
    def __init__(self, num_classes, eos_coef, iter_per_update):
        defaults = dict(
            num_classes=num_classes, eos_coef=eos_coef, iter_per_update=iter_per_update
        )
        super().__init__("label_loss", defaults)

        empty_weight = paddle.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.target_classes = None
        self.src_logits = None

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = _get_src_permutation_idx(indices)
        target_classes_o = paddle.concat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        self.target_classes = target_classes_o
        self.src_logits = src_logits[idx]

        target_classes = paddle.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype='int64',
            device=src_logits.device,
        )  # batch_size x num_queries

        # assign correct classes to matched queries
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose([0, 2, 1]), target_classes, weight=self.empty_weight,
        )
        losses = {"loss_ce": (loss_ce / self.iter_per_update)}

        return losses


class FocalLabelLoss(BaseLoss):
    def __init__(self, num_classes, focal_alpha):
        defaults = dict(num_classes=num_classes, focal_alpha=focal_alpha,)
        super().__init__("focal_label_loss", defaults)

        self.target_classes = None
        self.src_logits = None

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = _get_src_permutation_idx(indices)
        target_classes_o = paddle.concat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        self.target_classes = target_classes_o
        self.src_logits = src_logits[idx]
        target_classes = paddle.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype='int64',
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = paddle.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            device=src_logits.device,
        )
        target_classes_onehot.scatter(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=2,
        )
        losses = {"loss_ce": loss_ce}

        return losses


class BoxesLoss(BaseLoss):
    def __init__(self):
        defaults = dict()
        super().__init__("boxes_loss", defaults)

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = _get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]  # batch_size * nb_target_boxes x 4
        target_boxes = paddle.concat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], axis=0
        )  # batch_size * nb_target_boxes x 4

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}

        loss_giou = 1 - paddle.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_bbox"] = paddle.sum(loss_bbox) / num_boxes
        losses["loss_giou"] = paddle.sum(loss_giou) / num_boxes

        return losses


class Boxes3DLoss(BaseLoss):
    def __init__(self):
        defaults = dict()
        super().__init__("boxes3d_loss", defaults)

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx = _get_src_permutation_idx(indices)
        src_boxes, src_rads = paddle.split(outputs["pred_boxes"][idx], 6, axis=-1)
        target_boxes = paddle.concat(
            [t["boxes"][i][..., :6] for t, (_, i) in zip(targets, indices)], axis=0
        )  # batch_size * nb_target_boxes x 6

        target_rads = paddle.concat(
            [t["boxes"][i][..., 6:] for t, (_, i) in zip(targets, indices)], axis=0
        )  # batch_size * nb_target_boxes x (1, 2)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_rad = F.l1_loss(src_rads, target_rads, reduction="none")

        losses = {}

        loss_giou = 1 - paddle.diag(
            box3d_ops.generalized_box3d_iou(
                box3d_ops.box_cxcyczlwh_to_xyxyxy(src_boxes),
                box3d_ops.box_cxcyczlwh_to_xyxyxy(target_boxes),
            )
        )
        losses["loss_bbox"] = paddle.sum(loss_bbox) / num_boxes
        losses["loss_giou"] = paddle.sum(loss_giou) / num_boxes
        losses["loss_rad"] = paddle.sum(loss_rad) / num_boxes

        return losses


class MaskLoss(BaseLoss):
    def __init__(self, mask_size):
        defaults = dict(mask_size=mask_size)
        super().__init__("mask_loss", defaults)

    def forward(self, outputs, targets, indices, num_boxes):
        assert "pred_masks" in outputs
        idx = _get_src_permutation_idx(indices)

        target_masks = paddle.concat(
            [t["instance_masks"][i] for t, (_, i) in zip(targets, indices)], axis=0
        )

        src_masks = outputs["pred_masks"]
        src_masks = paddle.flatten(src_masks[idx], start_axis=1)
        target_masks = paddle.flatten(target_masks, start_axis=1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes)
            / (self.mask_size ** 2),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }

        return losses


@register_loss("detr")
class DETRLoss(BaseLoss):
    def __init__(
        self, num_classes, matcher, weight_dict, eos_coef, losses, iter_per_update
    ):
        defaults = dict(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=losses,
            iter_per_update=iter_per_update,
        )
        super().__init__("detr", defaults)

        self.detr_losses = nn.LayerDict()
        for loss in losses:
            if loss == "boxes":
                self.detr_losses[loss] = BoxesLoss()
            elif loss == "labels":
                self.detr_losses[loss] = LabelLoss(
                    num_classes, eos_coef, iter_per_update
                )
            else:
                raise ValueError(
                    "Only boxes|labels|balanced_labels are supported for detr "
                    "losses. Found {}".format(loss)
                )

    def get_target_classes(self):
        for kk in self.detr_losses.keys():
            if "labels" in kk:
                return (
                    self.detr_losses[kk].src_logits,
                    self.detr_losses[kk].target_classes,
                )

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if "num_boxes" not in outputs:
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = paddle.to_tensor(
                [num_boxes],
                dtype=paddle.float32,
                device=next(iter(outputs.values())).device,
            )
            if is_dist_avail_and_initialized():
                paddle.distributed.all_reduce(num_boxes)
            num_boxes = paddle.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = outputs["num_boxes"].item()

        # Compute all the requested losses
        losses = {}

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.detr_losses[loss](
                        aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        for loss in self.losses:
            losses.update(self.detr_losses[loss](outputs, targets, indices, num_boxes))

        return losses


@register_loss("boxer2d")
class Boxer2DLoss(BaseLoss):
    def __init__(
        self, num_classes, matcher, weight_dict, losses, iter_per_update,
    ):
        defaults = dict(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            iter_per_update=iter_per_update,
        )
        if "masks" in losses:
            defaults["mask_size"] = 28

        super().__init__("boxer2d", defaults)

        self.boxer2d_losses = nn.LayerDict()
        self.boxer2d_enc_losses = nn.LayerDict()
        for loss in losses:
            if loss == "boxes":
                self.boxer2d_losses[loss] = BoxesLoss()
                self.boxer2d_enc_losses[loss + "_enc"] = BoxesLoss()
            elif loss == "focal_labels":
                self.boxer2d_losses[loss] = FocalLabelLoss(num_classes, 0.25)
                self.boxer2d_enc_losses[loss + "_enc"] = FocalLabelLoss(1, 0.25)
            elif loss == "masks":
                self.boxer2d_losses[loss] = MaskLoss(self.mask_size)
            else:
                raise ValueError(
                    "Only boxes|focal_labels|masks are supported for boxer2d "
                    "losses. Found {}".format(loss)
                )

    def get_target_classes(self):
        for kk in self.boxer2d_losses.keys():
            if "labels" in kk:
                return (
                    self.boxer2d_losses[kk].src_logits,
                    self.boxer2d_losses[kk].target_classes,
                )

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if "num_boxes" not in outputs:
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = paddle.to_tensor(
                [num_boxes],
                dtype=paddle.float32,
                device=next(iter(outputs.values())).device,
            )
            if is_dist_avail_and_initialized():
                paddle.distributed.all_reduce(num_boxes)
            num_boxes = paddle.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = outputs["num_boxes"].item()

        # Compute all the requested losses
        losses = {}

        if "enc_outputs" in outputs:
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = paddle.zeros_like(bt["labels"])

            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, bin_targets)
                for loss in self.losses:
                    if loss == "masks":
                        continue

                    l_dict = self.boxer2d_enc_losses[loss + "_enc"](
                        enc_outputs, bin_targets, indices, num_boxes
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            if "masks" in self.losses:
                with paddle.no_grad():
                    for t in targets:
                        instance_masks = extract_grid(
                            t["masks"][:, None].float(),
                            None,
                            t["boxes"][:, None],
                            self.mask_size,
                        )
                        instance_masks = (instance_masks.squeeze(1) >= 0.5).float()
                        t["instance_masks"] = instance_masks

            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.boxer2d_losses[loss](
                        aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        for loss in self.losses:
            losses.update(
                self.boxer2d_losses[loss](outputs, targets, indices, num_boxes)
            )

        return losses


@register_loss("boxer3d")
class Boxer3DLoss(BaseLoss):
    def __init__(self, num_classes, matcher, weight_dict, losses, iter_per_update):
        defaults = dict(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            iter_per_update=iter_per_update,
        )
        super().__init__("boxer3d", defaults)

        self.boxer3d_losses = nn.LayerDict()
        self.boxer3d_enc_losses = nn.LayerDict()
        for loss in losses:
            if loss == "boxes":
                self.boxer3d_losses[loss] = Boxes3DLoss()
                self.boxer3d_enc_losses[loss + "_enc"] = Boxes3DLoss()
            elif loss == "focal_labels":
                self.boxer3d_losses[loss] = FocalLabelLoss(num_classes, 0.25)
                self.boxer3d_enc_losses[loss + "_enc"] = FocalLabelLoss(1, 0.25)
            else:
                raise ValueError(
                    "Only boxes|focal_labels are supported for boxer3d "
                    "losses. Found {}".format(loss)
                )

    def get_target_classes(self):
        for kk in self.boxer3d_losses.keys():
            if "labels" in kk:
                return (
                    self.boxer3d_losses[kk].src_logits,
                    self.boxer3d_losses[kk].target_classes,
                )

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if "num_boxes" not in outputs:
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = paddle.to_tensor(
                [num_boxes],
                dtype=paddle.float32,
                device=next(iter(outputs.values())).device,
            )
            if is_dist_avail_and_initialized():
                paddle.distributed.all_reduce(num_boxes)
            num_boxes = paddle.clamp(num_boxes / get_world_size(), min=1).item()
        else:
            num_boxes = outputs["num_boxes"].item()

        # Compute all the requested losses
        losses = {}

        if "enc_outputs" in outputs:
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = paddle.zeros_like(bt["labels"])

            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, bin_targets)
                for loss in self.losses:
                    l_dict = self.boxer3d_enc_losses[loss + "_enc"](
                        enc_outputs, bin_targets, indices, num_boxes
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.boxer3d_losses[loss](
                        aux_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        for loss in self.losses:
            losses.update(
                self.boxer3d_losses[loss](outputs, targets, indices, num_boxes)
            )

        return losses


def _get_src_permutation_idx(indices):
    # permute predicted indices back to the original order
    batch_idx = paddle.concat([paddle.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = paddle.concat([src for (src, _) in indices])
    return batch_idx, src_idx