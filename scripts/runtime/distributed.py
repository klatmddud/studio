from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist


@dataclass(slots=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device | None = None
    backend: str = "nccl"
    master_addr: str = "127.0.0.1"
    master_port: int | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def is_distributed(context: DistributedContext | None) -> bool:
    return bool(context is not None and context.enabled and context.world_size > 1)


def is_main_process(context: DistributedContext | None) -> bool:
    if not is_distributed(context):
        return True
    return bool(context and context.rank == 0)


def setup_process_group(context: DistributedContext | None) -> None:
    if not is_distributed(context):
        return
    if context is None:
        return
    if context.device is None or context.device.type != "cuda":
        raise RuntimeError("Multi-GPU training requires CUDA devices.")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    is_nccl_available = getattr(dist, "is_nccl_available", lambda: False)
    if context.backend == "nccl" and not is_nccl_available():
        raise RuntimeError(
            "Multi-GPU training uses DDP with the NCCL backend, but NCCL is not available. "
            "Run single-device training or use a PyTorch/Linux environment with NCCL support."
        )
    if dist.is_initialized():
        return
    if context.master_port is None:
        raise ValueError("DistributedContext.master_port is required for DDP setup.")
    dist.init_process_group(
        backend=context.backend,
        init_method=f"tcp://{context.master_addr}:{context.master_port}",
        rank=context.rank,
        world_size=context.world_size,
    )


def cleanup_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext | None) -> None:
    if is_distributed(context):
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def all_reduce_metrics(
    metrics: Mapping[str, float | int],
    context: DistributedContext | None,
    *,
    average: bool = True,
    device: torch.device | None = None,
) -> dict[str, float]:
    if not is_distributed(context):
        return {key: float(value) for key, value in metrics.items()}
    if not metrics:
        return {}
    if device is None:
        device = context.device if context is not None else torch.device("cpu")
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average and context is not None and context.world_size > 0:
        values /= float(context.world_size)
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


def all_gather_object(value: Any, context: DistributedContext | None) -> list[Any]:
    if not is_distributed(context):
        return [value]
    gathered: list[Any] = [None for _ in range(context.world_size)]  # type: ignore[list-item]
    dist.all_gather_object(gathered, value)
    return gathered


def broadcast_object(
    value: Any,
    context: DistributedContext | None,
    *,
    src: int = 0,
) -> Any:
    if not is_distributed(context):
        return value
    values = [value]
    dist.broadcast_object_list(values, src=src)
    return values[0]


def synchronize_extra_state(
    module: Any,
    context: DistributedContext | None,
    merge_fn: Callable[[list[Any]], Any],
) -> Any:
    get_extra_state = getattr(module, "get_extra_state", None)
    set_extra_state = getattr(module, "set_extra_state", None)
    if not callable(get_extra_state) or not callable(set_extra_state):
        return None

    local_state = get_extra_state()
    gathered = all_gather_object(local_state, context)
    merged = merge_fn(gathered) if is_main_process(context) else None
    merged = broadcast_object(merged, context, src=0)
    set_extra_state(merged)
    return merged
