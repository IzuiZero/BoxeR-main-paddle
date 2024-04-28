import paddle
import paddle.nn.functional as F

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = paddle.unbind(boxes, axis=1)
    return paddle.stack((xmin, ymin, xmax - xmin, ymax - ymin), axis=1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = paddle.unbind(x, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = paddle.unbind(x, axis=-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return paddle.stack(b, axis=-1)

def box_intersect(boxes1, boxes2):
    lt = paddle.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = paddle.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter

def box_iou(boxes1, boxes2):
    area1 = paddle.prod(boxes1[:, 2:] - boxes1[:, :2], axis=1)
    area2 = paddle.prod(boxes2[:, 2:] - boxes2[:, :2], axis=1)
    inter = box_intersect(boxes1, boxes2)
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    assert paddle.all(boxes1[:, 2:] >= boxes1[:, :2])
    assert paddle.all(boxes2[:, 2:] >= boxes2[:, :2])
    iou, union = box_iou(boxes1, boxes2)
    lt = paddle.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = paddle.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area

def masks_to_boxes(masks):
    if paddle.numel(masks) == 0:
        return paddle.zeros([0, 4], dtype=masks.dtype)

    h, w = masks.shape[-2:]
    y = paddle.arange(h, dtype=masks.dtype)
    x = paddle.arange(w, dtype=masks.dtype)
    y, x = paddle.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = paddle.max(x_mask.reshape([masks.shape[0], -1]), axis=-1)
    x_min = paddle.min(x_mask.masked_fill(~(masks.astype(bool)), 1e8).reshape([masks.shape[0], -1]), axis=-1)

    y_mask = masks * y.unsqueeze(0)
    y_max = paddle.max(y_mask.reshape([masks.shape[0], -1]), axis=-1)
    y_min = paddle.min(y_mask.masked_fill(~(masks.astype(bool)), 1e8).reshape([masks.shape[0], -1]), axis=-1)

    return paddle.stack([x_min, y_min, x_max, y_max], axis=1)

def iou_with_ign(boxes1, boxes2):
    area1 = paddle.prod(boxes1[:, 2:] - boxes1[:, :2], axis=1)
    intersect = box_intersect(boxes1, boxes2)
    iou_w_ign = intersect / area1
    return iou_w_ign