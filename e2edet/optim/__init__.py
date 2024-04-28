import collections.abc
import os
import copy
import paddle
import paddle.optimizer as paddle_optim

from e2edet.optim.oss import OSS
from e2edet.optim.optimizer import BaseOptimizer
from e2edet.utils.general import get_optimizer_parameters

OPTIM_REGISTRY = {"sgd": paddle_optim.SGD, "adamw": paddle_optim.AdamW}

def build_optimizer(config, model):
    optim_type = config.optimizer["type"]
    optim_config = copy.deepcopy(config.optimizer["params"])

    use_oss = optim_config.pop("use_oss", False)

    if optim_type not in OPTIM_REGISTRY:
        raise ValueError("Optimizer ({}) is not found.".format(optim_type))

    model_params = get_optimizer_parameters(model)

    if isinstance(model_params[0], collections.abc.Sequence):
        param_groups = []
        backbone_group, other_group = model_params

        lr_backbone = optim_config.pop("lr_backbone", optim_config["lr"])

        for group in backbone_group:
            group["lr"] = lr_backbone
            param_groups.append(group)

        for group in other_group:
            if "lr_multi" in group:
                group["lr"] = optim_config["lr"] * group.pop("lr_multi")
            param_groups.append(group)
    else:
        param_groups = [{"learning_rate": optim_config["lr"], "parameters": model_params}]

    optimizer_cls = OSS if use_oss else BaseOptimizer
    optimizer = optimizer_cls(parameters=param_groups, optimizer=OPTIM_REGISTRY[optim_type], **optim_config)

    return optimizer

def register_optim(name):
    def register_optim_cls(cls):
        if name in OPTIM_REGISTRY:
            raise ValueError("Cannot register duplicate optimizer ({})".format(name))
        elif not issubclass(cls, paddle.optimizer.Optimizer):
            raise ValueError(
                "Optimizer ({}: {}) must extend paddle.optimizer.Optimizer".format(
                    name, cls.__name__
                )
            )

        OPTIM_REGISTRY[name] = cls
        return cls

    return register_optim_cls

optims_dir = os.path.dirname(__file__)
for file in os.listdir(optims_dir):
    path = os.path.join(optims_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        optim_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = __import__(f"e2edet.optim.{optim_name}", fromlist=[optim_name])