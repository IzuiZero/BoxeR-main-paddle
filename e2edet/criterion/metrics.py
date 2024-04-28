import collections
import paddle
import paddle.nn.functional as F
from e2edet.utils.distributed import reduce_dict

METRIC_REGISTRY = {}

def build_metric(metrics):
    if not isinstance(metrics, collections.abc.Sequence):
        metrics = (metrics,)

    module_metrics = {}
    for metric in metrics:
        if metric["type"] not in METRIC_REGISTRY:
            raise ValueError("Metric ({}) is not found.".format(metric["type"]))
        module_metric = METRIC_REGISTRY[metric["type"]](**metric["params"])
        module_metrics[metric["type"]] = module_metric

    return module_metrics

def register_metric(name):
    def register_metric_cls(cls):
        if name in METRIC_REGISTRY:
            raise ValueError("Cannot register duplicate loss ({})".format(name))
        METRIC_REGISTRY[name] = cls
        return cls
    return register_metric_cls

class BaseMetric:
    def __init__(self, name, params={}):
        self.name = name
        for kk, vv in params.items():
            setattr(self, kk, vv)

    def calculate(self, output, target, *args, **kwargs):
        raise NotImplementedError("'calculate' must be implemented in the child class")

    def __call__(self, *args, **kwargs):
        with paddle.no_grad():
            metric = self.calculate(*args, **kwargs) / self.iter_per_update
            output = {self.name: metric}
            output = reduce_dict(output)
        return output

@register_metric("accuracy")
class Accuracy(BaseMetric):
    def __init__(self, iter_per_update=1):
        defaults = dict(iter_per_update=iter_per_update)
        super().__init__("accuracy", defaults)

    def calculate(self, output, target, topk=(1,)):
        if target.numel() == 0:
            return paddle.zeros([], device=output.device)
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).cast('float32').sum(0)
            res.append(correct_k.mul(100.0 / batch_size))

        return res[0]

@register_metric("cardinality")
class Cardinality(BaseMetric):
    def __init__(self, iter_per_update=1):
        defaults = dict(iter_per_update=iter_per_update)
        super().__init__("cardinality", defaults)

    def calculate(self, output, target):
        pred_logits = output["pred_logits"]  # batch_size x num_queries x num_classess
        device = pred_logits.device
        tgt_lengths = paddle.to_tensor(
            [len(v["labels"]) for v in target], dtype='int32', device=device
        )  # batch_size

        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.astype('float32'), tgt_lengths.astype('float32'))

        return card_err