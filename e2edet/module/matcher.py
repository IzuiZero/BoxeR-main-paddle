import paddle
from paddle import nn
from scipy.optimize import linear_sum_assignment
from e2edet.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from e2edet.utils.det3d.box_ops import (
    box_cxcyczlwh_to_xyxyxy,
    generalized_box3d_iou,
)


class HungarianMatcher(nn.Layer):
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_label: bool = False,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_label = focal_label
        self.norm = nn.Softmax(-1) if not focal_label else None
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @paddle.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.norm is not None:
            out_prob = self.norm(
                outputs["pred_logits"].flatten(0, 1)
            )  # [batch_size * num_queries, num_classes]
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        tgt_ids = paddle.concat([v["labels"] for v in targets])
        tgt_bbox = paddle.concat([v["boxes"] for v in targets])

        # Compute the classification cost
        out_prob = out_prob.astype('float32')
        out_bbox = out_bbox.astype('float32')
        tgt_bbox = tgt_bbox.astype('float32')

        if self.norm is not None:
            cost_class = -out_prob[:, tgt_ids]
        else:
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (
                (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            )
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = paddle.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.reshape(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                paddle.to_tensor(i, dtype='int64'),
                paddle.to_tensor(j, dtype='int64'),
            )
            for i, j in indices
        ]

    def extra_repr(self):
        s = "cost_class={cost_class}, cost_bbox={cost_bbox}, cost_giou={cost_giou}, focal_label={focal_label}"
        return s.format(**self.__dict__)


class HungarianMatcher3d(nn.Layer):
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_rad: float = 1,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_rad = cost_rad

        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_rad != 0
        ), "all costs cant be 0"

    @paddle.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].sigmoid()
        out_bbox, out_rad = paddle.split(outputs["pred_boxes"], 6, axis=-1)

        tgt_ids = [v["labels"] for v in targets]
        tgt_bbox = [v["boxes"][..., :6] for v in targets]
        tgt_rad = [v["boxes"][..., 6:] for v in targets]

        alpha = 0.25
        gamma = 2.0

        C = []
        for i in range(bs):
            out_prob_i = out_prob[i].astype('float32')
            out_bbox_i = out_bbox[i].astype('float32')
            out_rad_i = out_rad[i].astype('float32')
            tgt_bbox_i = tgt_bbox[i].astype('float32')
            tgt_rad_i = tgt_rad[i].astype('float32')

            cost_giou = -generalized_box3d_iou(
                box_cxcyczlwh_to_xyxyxy(out_bbox[i]),
                box_cxcyczlwh_to_xyxyxy(tgt_bbox[i]),
            )

            neg_cost_class = (
                (1 - alpha)
                * (out_prob_i ** gamma)
                * (-(1 - out_prob_i + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - out_prob_i) ** gamma) * (-(out_prob_i + 1e-8).log())
            )
            cost_class = pos_cost_class[:, tgt_ids[i]] - neg_cost_class[:, tgt_ids[i]]

            cost_bbox = paddle.cdist(out_bbox_i, tgt_bbox_i, p=1)
            cost_rad = paddle.cdist(out_rad_i, tgt_rad_i, p=1)

            C_i = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
                + self.cost_rad * cost_rad
            )
            C_i = C_i.reshape(num_queries, -1).cpu()
            C.append(C_i)

        indices = [linear_sum_assignment(c) for c in C]

        return [
            (
                paddle.to_tensor(i, dtype='int64'),
                paddle.to_tensor(j, dtype='int64'),
            )
            for i, j in indices
        ]

    def extra_repr(self):
        s = "cost_class={cost_class}, cost_bbox={cost_bbox}, cost_giou={cost_giou}, cost_rad={cost_rad}"
        return s.format(**self.__dict__)


def build_matcher(config):
    matcher_type = config.type
    params = config.params
    if matcher_type == "hungarian3d":
        matcher = HungarianMatcher3d(
            cost_class=params["class_weight"],
            cost_bbox=params["bbox_weight"],
            cost_giou=params["giou_weight"],
            cost_rad=params["rad_weight"],
        )
    elif matcher_type == "hungarian":
        matcher = HungarianMatcher(
            cost_class=params["class_weight"],
            cost_bbox=params["bbox_weight"],
            cost_giou=params["giou_weight"],
            focal_label=params["focal_label"],
        )
    else:
        raise ValueError(f"Only hungarian3d and hungarian accepted, got {matcher_type}")

    return matcher
