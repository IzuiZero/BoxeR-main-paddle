import importlib
import os
import warnings

from e2edet.optim.scheduler.lr_scheduler import BaseScheduler


SCHEDULER_REGISTRY = {}


__all__ = ["BaseScheduler"]


def build_scheduler(config, optimizer):
    scheduler_config = config.get("scheduler", {})
    if "type" not in scheduler_config:
        raise ValueError(
            "LRScheduler attributes must have a 'type' key "
            "specifying the type of optimizer. "
            "(Custom or PyTorch)"
        )
    scheduler_type = scheduler_config["type"]

    if "params" not in scheduler_config:
        warnings.warn("schduler attributes has no params defined, defaulting to {}.")

    if scheduler_type not in SCHEDULER_REGISTRY:
        raise ValueError("Scheduler ({}) is not found.".format(scheduler_type))

    params = scheduler_config.get("params", {})
    scheduler = SCHEDULER_REGISTRY[scheduler_type](params, optimizer)

    return scheduler


def register_scheduler(name):
    def register_scheduler_cls(cls):
        if name in SCHEDULER_REGISTRY:
            raise ValueError("Cannot register duplicate lr_scheduler ({})".format(name))
        elif not issubclass(cls, BaseScheduler):
            raise ValueError(
                "LR_Scheduler ({}: {}) must extend BaseScheduler".format(
                    name, cls.__name__
                )
            )

        SCHEDULER_REGISTRY[name] = cls
        return cls

    return register_scheduler_cls


schedulers_dir = os.path.dirname(__file__)
for file in os.listdir(schedulers_dir):
    path = os.path.join(schedulers_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        scheduler_name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("e2edet.optim.scheduler." + scheduler_name)
