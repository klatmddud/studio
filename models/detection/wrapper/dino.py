from __future__ import annotations

"""DINO wrapper: YAML config -> externally provided DINO backend."""

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
from torchvision.ops import boxes as box_ops

from modules.nn import (
    CounterfactualFeaturePerturbation,
    MDMBObservation,
    cxcywh_to_xyxy,
    select_topk_indices,
)
from modules.nn.mdmb import MissedDetectionMemoryBank
from ops import compute_cfp_loss_dict

from ._base import BaseDetectionWrapper, load_cfg


TensorDict = dict[str, torch.Tensor]
BackendBuilder = Callable[..., Any]
Postprocessor = Callable[[Mapping[str, Any], torch.Tensor], Sequence[Mapping[str, Any]]]
CFPForward = Callable[..., Any]
MDMBObserver = Callable[
    [Any, list[TensorDict], MissedDetectionMemoryBank, "DINOWrapper"],
    None,
]


@dataclass(frozen=True)
class DINOBackendComponents:
    """Container for a raw DINO model plus loss/postprocess helpers."""

    model: nn.Module
    criterion: nn.Module | Callable[[Mapping[str, Any], list[TensorDict]], Mapping[str, Any]] | None = None
    bbox_postprocessor: Postprocessor | None = None
    prepare_inputs: Callable[[list[torch.Tensor]], Any] | None = None
    mdmb_observer: MDMBObserver | None = None
    cfp_forward: CFPForward | None = None


class DINOWrapper(BaseDetectionWrapper):
    """
    Adapter that lets a DINO backend participate in this project's wrapper contract.

    The actual DINO implementation is intentionally external to this repository.
    Configure a backend builder in YAML:

    backend:
      builder: your_package.your_module:build_dino_components
      kwargs:
        some_option: value

    The builder may return:
    - a ready-to-use nn.Module that already matches this project's forward contract, or
    - a tuple / mapping / object with model + criterion + bbox postprocessor.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        mdmb: MissedDetectionMemoryBank | None = None,
        cfp: CounterfactualFeaturePerturbation | None = None,
        backend_builder: BackendBuilder | str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.num_classes = int(cfg.get("num_classes", 91))
        self.transform_cfg = cfg.get("transform", {})
        self.inference_cfg = cfg.get("inference", {})
        self.loss_cfg = cfg.get("loss", {})
        self.label_cfg = cfg.get("labels", {})
        self.cfp = cfp
        self.mdmb = mdmb

        builder = self._resolve_backend_builder(cfg, backend_builder)
        built = self._invoke_builder(
            builder=builder,
            cfg=cfg,
            pre_neck=pre_neck,
            post_neck=post_neck,
            mdmb=mdmb,
            cfp=cfp,
            extra_kwargs=kwargs,
        )

        if isinstance(built, nn.Module):
            self.model = built
            self._native_model = True
            self._backend_accepts_targets = self._call_accepts_targets(self.model)
            self.criterion = None
            self.bbox_postprocessor = None
            self.prepare_inputs = None
            self.mdmb_observer = None
            self.cfp_forward = None
            return

        components = self._coerce_components(built)
        if not isinstance(components.model, nn.Module):
            raise TypeError("DINO backend builder must provide a torch.nn.Module as 'model'.")
        self.model = components.model
        self._native_model = False
        self._backend_accepts_targets = self._call_accepts_targets(self.model)
        self.criterion = components.criterion
        self.bbox_postprocessor = components.bbox_postprocessor
        self.prepare_inputs = components.prepare_inputs
        self.mdmb_observer = components.mdmb_observer
        self.cfp_forward = components.cfp_forward or self._resolve_cfp_forward(self.model)

        if self.criterion is None:
            raise ValueError(
                "DINO backend builder must provide a criterion unless it returns a fully "
                "adapted nn.Module."
            )
        if self.bbox_postprocessor is None:
            raise ValueError(
                "DINO backend builder must provide a bbox postprocessor unless it returns "
                "a fully adapted nn.Module."
            )

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ):
        if self._native_model:
            return self.model(images, targets)

        original_sizes = self._collect_original_sizes(images)
        prepared_inputs = self._prepare_inputs(images)
        prepared_targets = self._prepare_targets(targets, original_sizes) if targets is not None else None
        outputs = self._call_backend_model(prepared_inputs, prepared_targets)

        if self.training:
            self._observe_mdmb(outputs, prepared_targets)
            loss_dict = self._compute_losses(outputs, prepared_targets)
            cfp_losses = self._compute_cfp_losses(outputs, prepared_targets)
            if cfp_losses:
                loss_dict.update(cfp_losses)
            if not loss_dict:
                raise RuntimeError("DINO backend returned no trainable losses.")
            return loss_dict

        predictions = self._postprocess(outputs, original_sizes)
        return [self._finalize_prediction(prediction) for prediction in predictions]

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        pre_neck: nn.Module | None = None,
        post_neck: nn.Module | None = None,
        mdmb: MissedDetectionMemoryBank | None = None,
        cfp: CounterfactualFeaturePerturbation | None = None,
        backend_builder: BackendBuilder | str | None = None,
        **kwargs,
    ) -> "DINOWrapper":
        """Create the wrapper from a YAML config path."""
        return cls(
            load_cfg(yaml_path),
            pre_neck=pre_neck,
            post_neck=post_neck,
            mdmb=mdmb,
            cfp=cfp,
            backend_builder=backend_builder,
            **kwargs,
        )

    def _prepare_inputs(self, images: list[torch.Tensor]) -> Any:
        if self.prepare_inputs is None:
            return images
        return self.prepare_inputs(images)

    def _call_backend_model(
        self,
        prepared_inputs: Any,
        prepared_targets: list[TensorDict] | None,
    ) -> Any:
        if prepared_targets is None or not self._backend_accepts_targets:
            return self.model(prepared_inputs)
        return self.model(prepared_inputs, prepared_targets)

    def _compute_losses(
        self,
        outputs: Any,
        prepared_targets: list[TensorDict] | None,
    ) -> dict[str, torch.Tensor]:
        if prepared_targets is None:
            raise RuntimeError("Training requires targets, but no targets were provided.")

        if self._looks_like_loss_dict(outputs):
            raw_losses = outputs
        else:
            raw_losses = self.criterion(outputs, prepared_targets)

        weighted_losses: dict[str, torch.Tensor] = {}
        for name, value in raw_losses.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim != 0:
                continue
            weight = self._infer_loss_weight(name)
            if weight is None:
                continue
            weighted_losses[name] = value * weight
        return weighted_losses

    def _postprocess(
        self,
        outputs: Any,
        original_sizes: list[tuple[int, int]],
    ) -> list[Mapping[str, Any]]:
        if isinstance(outputs, list):
            return outputs

        target_sizes = torch.tensor(
            original_sizes,
            dtype=torch.int64,
            device=self._infer_device(outputs),
        )
        predictions = self.bbox_postprocessor(outputs, target_sizes)
        return list(predictions)

    def _finalize_prediction(self, prediction: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        boxes = self._as_tensor(prediction.get("boxes"), dtype=torch.float32)
        scores = self._as_tensor(prediction.get("scores"), dtype=torch.float32)
        labels = self._restore_label_ids(
            self._as_tensor(prediction.get("labels"), dtype=torch.int64)
        )

        if boxes.shape[0] != scores.shape[0] or boxes.shape[0] != labels.shape[0]:
            raise ValueError("DINO bbox postprocessor returned inconsistent prediction lengths.")

        keep = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)

        score_thresh = float(self.inference_cfg.get("score_thresh", 0.0))
        if score_thresh > 0.0:
            keep &= scores >= score_thresh

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        topk = self.inference_cfg.get("topk")
        if topk is not None and int(topk) > 0 and scores.numel() > int(topk):
            topk_indices = torch.topk(scores, k=int(topk)).indices
            boxes = boxes[topk_indices]
            scores = scores[topk_indices]
            labels = labels[topk_indices]

        if bool(self.inference_cfg.get("use_nms", False)) and boxes.numel() > 0:
            from torchvision.ops import nms

            nms_thresh = float(self.inference_cfg.get("nms_thresh", 0.5))
            keep_indices = nms(boxes, scores, nms_thresh)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

    def _prepare_targets(
        self,
        targets: list[dict[str, torch.Tensor]],
        original_sizes: list[tuple[int, int]],
    ) -> list[TensorDict]:
        prepared: list[TensorDict] = []
        for target, (height, width) in zip(targets, original_sizes, strict=True):
            boxes_xyxy = target["boxes"]
            labels_raw = target["labels"].to(dtype=torch.int64)
            labels = self._map_label_ids(labels_raw)

            if labels.numel() > 0:
                min_label = int(labels.min().item())
                max_label = int(labels.max().item())
                if min_label < 0 or max_label >= self.num_classes:
                    raise ValueError(
                        "DINO labels must map into [0, num_classes). "
                        f"Received range [{min_label}, {max_label}] for num_classes={self.num_classes}."
                    )

            size_tensor = torch.tensor([height, width], dtype=torch.int64, device=boxes_xyxy.device)

            prepared_target: TensorDict = {
                "boxes": self._normalize_boxes(boxes_xyxy, width=width, height=height),
                "boxes_xyxy": boxes_xyxy.to(dtype=torch.float32),
                "labels": labels,
                "labels_raw": labels_raw,
                "image_id": target["image_id"].to(dtype=torch.int64),
                "size": size_tensor,
                "orig_size": size_tensor.clone(),
            }
            if "area" in target:
                prepared_target["area"] = target["area"].to(dtype=torch.float32)
            if "iscrowd" in target:
                prepared_target["iscrowd"] = target["iscrowd"].to(dtype=torch.int64)
            prepared.append(prepared_target)
        return prepared

    def _map_label_ids(self, labels: torch.Tensor) -> torch.Tensor:
        id_to_index = self.label_cfg.get("id_to_index")
        if isinstance(id_to_index, Mapping):
            if labels.numel() == 0:
                return labels.clone()
            mapped = [int(id_to_index[int(label.item())]) for label in labels]
            return torch.tensor(mapped, dtype=torch.int64, device=labels.device)

        offset = int(self.label_cfg.get("offset", 1))
        return labels.to(dtype=torch.int64) - offset

    def _restore_label_ids(self, labels: torch.Tensor) -> torch.Tensor:
        index_to_id = self.label_cfg.get("index_to_id")
        if isinstance(index_to_id, Mapping):
            if labels.numel() == 0:
                return labels.clone()
            restored = [int(index_to_id[int(label.item())]) for label in labels]
            return torch.tensor(restored, dtype=torch.int64, device=labels.device)

        offset = int(self.label_cfg.get("offset", 1))
        return labels.to(dtype=torch.int64) + offset

    def _normalize_boxes(
        self,
        boxes_xyxy: torch.Tensor,
        *,
        width: int,
        height: int,
    ) -> torch.Tensor:
        boxes = boxes_xyxy.to(dtype=torch.float32)
        if boxes.numel() == 0:
            return boxes.reshape(0, 4)

        scale = boxes.new_tensor([width, height, width, height])
        boxes = boxes / scale

        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        return torch.stack((cx, cy, w, h), dim=-1)

    def _infer_loss_weight(self, name: str) -> float | None:
        lower_name = name.lower()
        if "class_error" in lower_name or "cardinality" in lower_name:
            return None
        if "loss" not in lower_name:
            return None

        overrides = self.loss_cfg.get("weights")
        if isinstance(overrides, Mapping) and name in overrides:
            return float(overrides[name])

        if "bbox" in lower_name:
            return float(self.loss_cfg.get("bbox_weight", 5.0))
        if "giou" in lower_name:
            return float(self.loss_cfg.get("giou_weight", 2.0))
        if "ce" in lower_name or "cls" in lower_name or "class" in lower_name:
            return float(self.loss_cfg.get("class_weight", 1.0))
        return 1.0

    def _looks_like_loss_dict(self, value: Any) -> bool:
        if not isinstance(value, Mapping) or not value:
            return False
        scalar_tensors = [
            item for item in value.values() if isinstance(item, torch.Tensor) and item.ndim == 0
        ]
        if not scalar_tensors:
            return False
        return any("loss" in str(key).lower() for key in value)

    def _coerce_components(self, built: Any) -> DINOBackendComponents:
        if isinstance(built, DINOBackendComponents):
            return built

        if isinstance(built, tuple) and len(built) in {3, 4}:
            model, criterion, postprocessor, *rest = built
            prepare_inputs = rest[0] if rest else None
            return DINOBackendComponents(
                model=model,
                criterion=criterion,
                bbox_postprocessor=self._resolve_postprocessor(postprocessor),
                prepare_inputs=prepare_inputs,
            )

        if isinstance(built, tuple) and len(built) == 6:
            model, criterion, postprocessor, prepare_inputs, mdmb_observer, cfp_forward = built
            return DINOBackendComponents(
                model=model,
                criterion=criterion,
                bbox_postprocessor=self._resolve_postprocessor(postprocessor),
                prepare_inputs=prepare_inputs,
                mdmb_observer=mdmb_observer,
                cfp_forward=cfp_forward,
            )

        if isinstance(built, tuple) and len(built) == 5:
            model, criterion, postprocessor, prepare_inputs, mdmb_observer = built
            return DINOBackendComponents(
                model=model,
                criterion=criterion,
                bbox_postprocessor=self._resolve_postprocessor(postprocessor),
                prepare_inputs=prepare_inputs,
                mdmb_observer=mdmb_observer,
            )

        if isinstance(built, Mapping):
            model = built.get("model")
            criterion = built.get("criterion")
            postprocessor = built.get("bbox_postprocessor") or built.get("postprocessor")
            if postprocessor is None and "postprocessors" in built:
                postprocessor = built["postprocessors"]
            prepare_inputs = built.get("prepare_inputs")
            mdmb_observer = built.get("mdmb_observer")
            cfp_forward = built.get("cfp_forward")
            return DINOBackendComponents(
                model=model,
                criterion=criterion,
                bbox_postprocessor=self._resolve_postprocessor(postprocessor),
                prepare_inputs=prepare_inputs,
                mdmb_observer=mdmb_observer,
                cfp_forward=cfp_forward,
            )

        model = getattr(built, "model", None)
        criterion = getattr(built, "criterion", None)
        postprocessor = getattr(built, "bbox_postprocessor", None) or getattr(
            built, "postprocessor", None
        )
        if postprocessor is None:
            postprocessor = getattr(built, "postprocessors", None)
        prepare_inputs = getattr(built, "prepare_inputs", None)
        mdmb_observer = getattr(built, "mdmb_observer", None)
        cfp_forward = getattr(built, "cfp_forward", None)
        if model is None:
            raise TypeError(
                "DINO backend builder returned an unsupported object. Expected nn.Module, "
                "DINOBackendComponents, tuple, mapping, or object with a .model attribute."
            )
        return DINOBackendComponents(
            model=model,
            criterion=criterion,
            bbox_postprocessor=self._resolve_postprocessor(postprocessor),
            prepare_inputs=prepare_inputs,
            mdmb_observer=mdmb_observer,
            cfp_forward=cfp_forward,
        )

    def _resolve_postprocessor(self, value: Any) -> Postprocessor | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            bbox = value.get("bbox")
            if bbox is None:
                raise ValueError("DINO postprocessors mapping must include a 'bbox' entry.")
            return bbox
        return value

    def _invoke_builder(
        self,
        *,
        builder: BackendBuilder,
        cfg: dict[str, Any],
        pre_neck: nn.Module | None,
        post_neck: nn.Module | None,
        mdmb: MissedDetectionMemoryBank | None,
        cfp: CounterfactualFeaturePerturbation | None,
        extra_kwargs: dict[str, Any],
    ) -> Any:
        backend_cfg = cfg.get("backend", {})
        builder_kwargs = dict(backend_cfg.get("kwargs", {}))
        builder_kwargs.update(extra_kwargs)

        candidate_kwargs = {
            "cfg": cfg,
            "config": cfg,
            "num_classes": self.num_classes,
            "pre_neck": pre_neck,
            "post_neck": post_neck,
            "mdmb": mdmb,
            "cfp": cfp,
            **builder_kwargs,
        }

        signature = inspect.signature(builder)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_kwargs:
            accepted_kwargs = candidate_kwargs
        else:
            accepted_kwargs = {
                name: value
                for name, value in candidate_kwargs.items()
                if name in signature.parameters
            }
        return builder(**accepted_kwargs)

    def _resolve_backend_builder(
        self,
        cfg: dict[str, Any],
        backend_builder: BackendBuilder | str | None,
    ) -> BackendBuilder:
        builder_ref = backend_builder
        if builder_ref is None:
            backend_cfg = cfg.get("backend", {})
            if isinstance(backend_cfg, Mapping):
                builder_ref = backend_cfg.get("builder")

        if builder_ref is None:
            raise RuntimeError(
                "DINO backend is not configured. Add 'backend.builder' to the YAML or pass "
                "'backend_builder=' when constructing DINOWrapper."
            )

        if callable(builder_ref):
            return builder_ref

        if not isinstance(builder_ref, str):
            raise TypeError("backend.builder must be a callable or 'module:attribute' string.")

        module_name, attribute_name = self._split_builder_ref(builder_ref)
        module = importlib.import_module(module_name)
        builder = module
        for attribute in attribute_name.split("."):
            builder = getattr(builder, attribute)
        if not callable(builder):
            raise TypeError(f"Resolved backend builder {builder_ref!r} is not callable.")
        return builder

    def _split_builder_ref(self, ref: str) -> tuple[str, str]:
        if ":" in ref:
            module_name, attribute_name = ref.split(":", maxsplit=1)
            return module_name, attribute_name
        if "." not in ref:
            raise ValueError(
                "backend.builder must use either 'module:attribute' or 'module.attribute' syntax."
            )
        module_name, _, attribute_name = ref.rpartition(".")
        return module_name, attribute_name

    def _collect_original_sizes(self, images: list[torch.Tensor]) -> list[tuple[int, int]]:
        return [(int(image.shape[-2]), int(image.shape[-1])) for image in images]

    def _observe_mdmb(
        self,
        outputs: Any,
        prepared_targets: list[TensorDict] | None,
    ) -> None:
        if self.mdmb is None or prepared_targets is None:
            return
        if self.mdmb_observer is not None:
            self.mdmb_observer(outputs, prepared_targets, self.mdmb, self)
            return
        if not isinstance(outputs, Mapping):
            return
        candidate_batches = self._collect_mdmb_candidate_batches(outputs, prepared_targets)
        observations = [
            observation
            for batch in candidate_batches
            for observation in batch["observations"]
        ]
        if observations:
            self.mdmb.update(observations)

    def _compute_cfp_losses(
        self,
        outputs: Any,
        prepared_targets: list[TensorDict] | None,
    ) -> dict[str, torch.Tensor] | None:
        if (
            self.cfp is None
            or self.mdmb is None
            or prepared_targets is None
            or self.cfp_forward is None
            or not isinstance(outputs, Mapping)
        ):
            return None

        pred_logits = outputs.get("pred_logits")
        if not isinstance(pred_logits, torch.Tensor) or pred_logits.ndim != 3:
            return None

        query_features = self._extract_query_features(outputs, pred_logits)
        if query_features.shape[-1] != self.cfp.feature_dim:
            return None
        candidate_batches = self._collect_mdmb_candidate_batches(outputs, prepared_targets)
        selected_tasks: list[dict[str, Any]] = []
        selected_features: list[torch.Tensor] = []
        for batch in candidate_batches:
            for position, track_id in enumerate(batch["track_ids"]):
                entry = self.mdmb.get(track_id)
                if entry is None or not entry.is_chronic_miss(self.mdmb.config):
                    continue
                if bool(batch["detected_mask"][position].item()):
                    continue
                selected_tasks.append(
                    {
                        "image_index": batch["image_index"],
                        "query_index": int(batch["candidate_indices"][position].item()),
                        "gt_label_index": int(batch["gt_label_indices"][position].item()),
                        "base_score": batch["base_scores"][position],
                    }
                )
                selected_features.append(batch["query_features"][position])

        if not selected_features:
            return None

        cfp_output = self.cfp(torch.stack(selected_features, dim=0))
        cf_query_features = query_features.detach().clone()
        for task, perturbed_feature in zip(
            selected_tasks,
            cfp_output.perturbed_features,
            strict=True,
        ):
            cf_query_features[task["image_index"], task["query_index"]] = perturbed_feature

        cf_outputs = self._call_cfp_forward(outputs, cf_query_features, prepared_targets)
        if not isinstance(cf_outputs, Mapping):
            return None

        cf_loss_dict = self._compute_losses(cf_outputs, prepared_targets)
        cf_pred_logits = cf_outputs.get("pred_logits")
        if isinstance(cf_pred_logits, torch.Tensor) and cf_pred_logits.ndim == 3:
            cf_class_scores = torch.sigmoid(cf_pred_logits)
            cf_scores = torch.stack(
                [
                    cf_class_scores[
                        task["image_index"],
                        task["query_index"],
                        task["gt_label_index"],
                    ]
                    for task in selected_tasks
                ],
                dim=0,
            )
        else:
            cf_scores = None
        base_scores = torch.stack(
            [task["base_score"] for task in selected_tasks],
            dim=0,
        )
        return compute_cfp_loss_dict(
            detection_loss=cf_loss_dict,
            delta=cfp_output,
            config=self.cfp.config,
            cf_scores=cf_scores,
            base_scores=base_scores,
        )

    def _collect_mdmb_candidate_batches(
        self,
        outputs: Mapping[str, Any],
        prepared_targets: list[TensorDict],
    ) -> list[dict[str, Any]]:
        if self.mdmb is None:
            return []

        pred_logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")
        if not isinstance(pred_logits, torch.Tensor) or not isinstance(pred_boxes, torch.Tensor):
            return []
        if pred_logits.ndim != 3 or pred_boxes.ndim != 3:
            return []

        query_features = self._extract_query_features(outputs, pred_logits)
        matches = self._match_queries(outputs, prepared_targets)
        max_candidates = self.mdmb.config.max_entries_per_image * 4

        normalized_boxes = cxcywh_to_xyxy(pred_boxes.detach()).clamp_(min=0.0, max=1.0)
        class_scores = torch.sigmoid(pred_logits.detach())
        pred_scores, pred_label_indices = class_scores.max(dim=-1)
        pred_label_ids = self._restore_label_ids(pred_label_indices)
        candidate_batches: list[dict[str, Any]] = []

        for image_index, prepared_target in enumerate(prepared_targets):
            gt_boxes_xyxy = prepared_target["boxes_xyxy"]
            if gt_boxes_xyxy.numel() == 0:
                continue

            size = prepared_target["orig_size"]
            height = int(size[0].item())
            width = int(size[1].item())
            scale = pred_boxes.new_tensor([width, height, width, height])
            predicted_boxes_xyxy = normalized_boxes[image_index] * scale

            iou_matrix = box_ops.box_iou(gt_boxes_xyxy, predicted_boxes_xyxy)
            iou_max, nearest_gt = iou_matrix.max(dim=0)
            gt_label_indices = prepared_target["labels"][nearest_gt]
            gt_label_ids = prepared_target.get("labels_raw", self._restore_label_ids(gt_label_indices))
            query_indices = torch.arange(pred_boxes.shape[1], device=pred_boxes.device)
            gt_scores = class_scores[image_index][query_indices, gt_label_indices]

            detected_mask = torch.zeros_like(iou_max, dtype=torch.bool)
            matched_src, matched_tgt = matches[image_index]
            if matched_src.numel() > 0:
                matched_scores = class_scores[image_index][matched_src, prepared_target["labels"][matched_tgt]]
                matched_ious = box_ops.box_iou(
                    predicted_boxes_xyxy[matched_src],
                    gt_boxes_xyxy[matched_tgt],
                ).diagonal()
                detected = (
                    (pred_label_ids[image_index][matched_src] == gt_label_ids[matched_tgt])
                    & (matched_scores >= self.mdmb.config.detection_score_threshold)
                    & (matched_ious >= self.mdmb.config.iou_high)
                )
                detected_mask[matched_src] = detected

            candidate_mask = (iou_max >= self.mdmb.config.iou_low) | detected_mask
            candidate_indices = torch.where(candidate_mask)[0]
            if candidate_indices.numel() == 0:
                continue

            priority = iou_max[candidate_indices] + gt_scores[candidate_indices]
            keep_order = select_topk_indices(priority, k=max_candidates)
            candidate_indices = candidate_indices[keep_order]
            image_id = int(prepared_target["image_id"].item())
            observations: list[MDMBObservation] = []
            track_ids: list[str] = []
            for local_index, candidate_index in enumerate(candidate_indices.tolist()):
                detected = bool(detected_mask[candidate_index].item())
                effective_iou = float(iou_max[candidate_index].item())
                if detected:
                    effective_iou = max(effective_iou, self.mdmb.config.iou_high)
                gt_class_id = int(gt_label_ids[nearest_gt[candidate_index]].item())
                track_id = self.mdmb.make_track_id(
                    image_id=image_id,
                    region_coords=normalized_boxes[image_index, candidate_index],
                    source="query",
                    gt_class=gt_class_id,
                )
                track_ids.append(track_id)
                observations.append(
                    MDMBObservation(
                        image_id=image_id,
                        region_coords=normalized_boxes[image_index, candidate_index],
                        iou_max=effective_iou,
                        cls_score=float(gt_scores[candidate_index].item()),
                        feature_vec=query_features[image_index, candidate_index],
                        gt_class=gt_class_id,
                        detected=detected,
                        track_id=track_id,
                        source="query",
                    )
                )

            candidate_batches.append(
                {
                    "image_index": image_index,
                    "candidate_indices": candidate_indices,
                    "track_ids": track_ids,
                    "detected_mask": detected_mask[candidate_indices],
                    "gt_label_indices": gt_label_indices[candidate_indices].to(dtype=torch.int64),
                    "base_scores": gt_scores[candidate_indices],
                    "query_features": query_features[image_index, candidate_indices].detach(),
                    "observations": observations,
                }
            )

        return candidate_batches

    def _match_queries(
        self,
        outputs: Mapping[str, Any],
        prepared_targets: list[TensorDict],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        matcher = getattr(self.criterion, "matcher", None)
        if not callable(matcher):
            return [
                (
                    torch.zeros((0,), dtype=torch.int64, device=prepared_targets[0]["boxes"].device),
                    torch.zeros((0,), dtype=torch.int64, device=prepared_targets[0]["boxes"].device),
                )
                for _ in prepared_targets
            ]
        try:
            raw_matches = matcher(
                {
                    "pred_logits": outputs["pred_logits"],
                    "pred_boxes": outputs["pred_boxes"],
                },
                prepared_targets,
            )
        except Exception:
            return [
                (
                    torch.zeros((0,), dtype=torch.int64, device=prepared_targets[0]["boxes"].device),
                    torch.zeros((0,), dtype=torch.int64, device=prepared_targets[0]["boxes"].device),
                )
                for _ in prepared_targets
            ]

        matches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for src_idx, tgt_idx in raw_matches:
            matches.append(
                (
                    src_idx.to(dtype=torch.int64),
                    tgt_idx.to(dtype=torch.int64),
                )
            )
        return matches

    def _extract_query_features(
        self,
        outputs: Mapping[str, Any],
        pred_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_queries, _ = pred_logits.shape
        for key in (
            "query_features",
            "decoder_features",
            "pred_features",
            "hidden_states",
            "hs",
        ):
            candidate = outputs.get(key)
            tensor = self._coerce_query_feature_tensor(candidate, batch_size, num_queries)
            if tensor is not None:
                return tensor
        return pred_logits.detach()

    def _coerce_query_feature_tensor(
        self,
        value: Any,
        batch_size: int,
        num_queries: int,
    ) -> torch.Tensor | None:
        if not isinstance(value, torch.Tensor):
            return None
        tensor = value.detach()
        if tensor.ndim == 3:
            if tensor.shape[0] == batch_size and tensor.shape[1] == num_queries:
                return tensor
        if tensor.ndim == 4:
            if tensor.shape[0] == batch_size and tensor.shape[2] == num_queries:
                return tensor[:, -1]
            if tensor.shape[1] == batch_size and tensor.shape[2] == num_queries:
                return tensor[-1]
            if tensor.shape[0] == batch_size and tensor.shape[1] == num_queries:
                return tensor[:, :, -1, :]
        return None

    def _resolve_cfp_forward(self, model: nn.Module) -> CFPForward | None:
        for name in ("cfp_forward", "forward_cfp"):
            candidate = getattr(model, name, None)
            if callable(candidate):
                return candidate
        return None

    def _call_cfp_forward(
        self,
        outputs: Mapping[str, Any],
        cf_query_features: torch.Tensor,
        prepared_targets: list[TensorDict],
    ) -> Any:
        if self.cfp_forward is None:
            return None

        candidate_kwargs = {
            "outputs": outputs,
            "base_outputs": outputs,
            "query_features": cf_query_features,
            "cf_query_features": cf_query_features,
            "targets": prepared_targets,
            "prepared_targets": prepared_targets,
            "wrapper": self,
            "model": self.model,
        }

        signature = inspect.signature(self.cfp_forward)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_kwargs:
            accepted_kwargs = candidate_kwargs
        else:
            accepted_kwargs = {
                name: value
                for name, value in candidate_kwargs.items()
                if name in signature.parameters
            }
        return self.cfp_forward(**accepted_kwargs)

    def _call_accepts_targets(self, module: nn.Module) -> bool:
        signature = inspect.signature(module.forward)
        parameters = list(signature.parameters.values())
        return any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
        ) or len(parameters) >= 2

    def _as_tensor(self, value: Any, *, dtype: torch.dtype) -> torch.Tensor:
        if value is None:
            return torch.empty((0,), dtype=dtype)
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)

    def _infer_device(self, value: Any) -> torch.device:
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, Mapping):
            for item in value.values():
                device = self._infer_device(item)
                if device.type != "cpu" or isinstance(item, torch.Tensor):
                    return device
        if isinstance(value, (list, tuple)):
            for item in value:
                device = self._infer_device(item)
                if device.type != "cpu" or isinstance(item, torch.Tensor):
                    return device
        return torch.device("cpu")
