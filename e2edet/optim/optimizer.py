import math
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

import paddle

if TYPE_CHECKING:  # pragma: no cover
    from paddle.optimizer.optimizer import Optimizer
else:
    Optimizer = Any


class BaseOptimizer(Optimizer):
    def __init__(
        self,
        parameters: List[paddle.Tensor],
        optimizer: Type[Optimizer] = paddle.optimizer.SGD,
        **default: Any
    ):
        self.in_super_constructor = True
        super().__init__(parameters, **default)
        self.in_super_constructor = False

        self._parameters = list(chain(*parameters))

        self.optimizer = optimizer(parameters, **default)

    def state_dict(self) -> Dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict)

        BaseOptimizer._sync_param_groups(state_dict["param_groups"], self.param_groups)
        BaseOptimizer._sync_param_groups(self.param_groups, self.optimizer.parameter_groups())

    def step(
        self, closure: Optional[Callable[[], float]] = None, **kwargs: Any
    ) -> Optional[float]:
        BaseOptimizer._sync_param_groups(self.param_groups, self.optimizer.parameter_groups())

        if closure is not None:
            loss = self.optimizer.step(closure=closure, **kwargs)
        else:
            loss = self.optimizer.step(**kwargs)

        BaseOptimizer._sync_param_groups(self.optimizer.parameter_groups(), self.param_groups)

        return loss

    def clip_grad_norm(self, max_norm):
        if max_norm > 0:
            return paddle.nn.utils.clip_grad_norm_(self._parameters, max_norm)
        else:
            return paddle.sqrt(
                paddle.to_tensor(
                    [
                        sum(
                            p.grad.norm().item() ** 2
                            for p in self._parameters
                            if p.grad is not None
                        )
                    ]
                )
            )

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        if not self.in_super_constructor:
            self.optimizer.add_param_group(param_group)

    @staticmethod
    def _sync_param_groups(
        source: List[Dict[Any, Any]], destination: List[Dict[Any, Any]]
    ) -> None:
        """Sync learning rate and other optimizer attributes (needed to support schedulers)."""

        for source_group, destination_group in zip(source, destination):
            # Sync everything but the parameters
            for k in filter(lambda x: x != "parameters", source_group.keys()):
                destination_group[k] = source_group[k]
