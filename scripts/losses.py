"""CohereFusion joint objective, manuscript Section 3.6."""

import torch
from torch.nn import functional as F

from scripts.loss_components import BaseFusionLoss


class CohereFusionLoss(BaseFusionLoss):
    def __init__(self, bins=64, entropy_margin=.12, mi_floor=2.8,
                 scd_floor=1.55, scd_weight=1.5, text_weight=.2,
                 dynamic_weight=3., dynamic_margin=.001,
                 counterfactual_weight=.3, counterfactual_margin=.025, residual_weight=.02,
                 global_weight=.35, semantic_weight=.35,
                 fixed_weight=6., fixed_margin=.001,
                 diversity_weight=.05, diversity_margin=.005,
                 dynamic_rho=.65, fixed_rho=.75,
                 fixed_targets=None, fixed_importance=None):
        super().__init__(bins, entropy_margin, mi_floor, scd_floor, scd_weight)
        self.text_weight, self.dynamic_weight = text_weight, dynamic_weight
        self.dynamic_margin, self.fixed_margin = dynamic_margin, fixed_margin
        self.counter_weight, self.counter_margin = counterfactual_weight, counterfactual_margin
        self.residual_weight = residual_weight
        self.global_weight, self.semantic_weight = global_weight, semantic_weight
        self.fixed_weight, self.diversity_weight = fixed_weight, diversity_weight
        self.diversity_margin = diversity_margin
        self.dynamic_rho, self.fixed_rho = dynamic_rho, fixed_rho
        self.register_buffer("fixed_targets", torch.tensor(
            fixed_targets if fixed_targets is not None else [5.39, .2255, .074, .0252, 2.60, .645, 1.55, .770]))
        self.register_buffer("fixed_importance", torch.tensor(
            fixed_importance if fixed_importance is not None else [2., 2., .5, 2.5, .5, .75, .75, 2.5]))
        if self.fixed_targets.shape != (8,) or self.fixed_importance.shape != (8,):
            raise ValueError("Fixed quality thresholds and importance must each contain eight values")
        if not (torch.isfinite(self.fixed_targets).all() and torch.isfinite(self.fixed_importance).all()):
            raise ValueError("Quality thresholds and importance must be finite")
        if (self.fixed_targets <= 0).any() or (self.fixed_importance < 0).any():
            raise ValueError("Quality thresholds must be positive and importance cannot be negative")
        if not (0 <= dynamic_rho <= 1 and 0 <= fixed_rho <= 1):
            raise ValueError("Quality aggregation coefficients must be between zero and one")

    @staticmethod
    def text_routing(routes, tasks):
        if tasks is None:
            raise ValueError("Training requires degradation task labels for paper text-routing supervision")
        target = routes[0].new_zeros((len(tasks), 6))
        for index, task in enumerate(tasks):
            if task not in ("low_light", "over_exposure", "ir_low_contrast", "ir_noise"):
                raise ValueError(f"Unknown degradation task: {task}")
            # Paper expert order: V-L/V-H/V-D/I-L/I-H/I-D.
            target[index, 3:6] = 1 / 3 if task in ("low_light", "over_exposure") else 0
            target[index, :3] = 0 if task in ("low_light", "over_exposure") else 1 / 3
        return torch.stack([-(target * (route + 1e-7).log()).sum(1) for route in routes]).mean()

    @staticmethod
    def shortfall(scores, reference, margin, importance=1., epsilon=1e-3):
        return importance * F.relu(reference * (1 + margin) - scores) / reference.abs().clamp_min(epsilon)

    @staticmethod
    def quality_loss(deficits, rho):
        return (rho * deficits.max(1).values + (1 - rho) * deficits.mean(1)).mean()

    def forward(self, visible, infrared, visible_gt, infrared_gt, fused, aux, tasks=None):
        # L_q retains luminance/gradient/Laplacian/contrast/chroma reconstruction,
        # source SSIM/soft-H/soft-MI/edge/correlation and expert balance terms.
        base, parts = super().forward(visible, infrared, visible_gt, infrared_gt, fused,
                                      aux["sewb_aux"], tasks)
        global_loss, _ = super().forward(visible, infrared, visible_gt, infrared_gt,
                                         aux["global_candidate"], aux["gewb_aux"], tasks)
        semantic_loss, _ = super().forward(visible, infrared, visible_gt, infrared_gt,
                                           aux["semantic_candidate"], aux["sewb_aux"], tasks)
        text_loss = self.text_routing(aux["sewb_aux"]["routes"], tasks)
        scores = self.metric_vector(visible, infrared, fused)
        with torch.no_grad():
            reference = torch.maximum(self.metric_vector(visible, infrared, aux["global_candidate"]),
                                      self.metric_vector(visible, infrared, aux["semantic_candidate"]))
        dynamic = self.shortfall(scores, reference, self.dynamic_margin)
        fixed = self.shortfall(scores, self.fixed_targets[None], self.fixed_margin,
                               self.fixed_importance[None], epsilon=1e-8)
        dynamic_loss = self.quality_loss(dynamic, self.dynamic_rho)
        fixed_loss = self.quality_loss(fixed, self.fixed_rho)
        counter = aux.get("counter_rsfr_weights")
        if counter is None and self.counter_weight:
            raise ValueError("Counterfactual text is required when counterfactual loss has nonzero weight")
        counter_loss = fused.new_zeros(()) if counter is None else F.relu(
            self.counter_margin - (aux["rsfr_weights"] - counter).abs().mean((1, 2, 3, 4))).mean()
        residual_loss = aux["band_residuals"].abs().mean()
        gap = (aux["global_candidate"] - aux["semantic_candidate"]).abs().mean()
        diversity = F.relu(self.diversity_margin - gap)
        total = (base + self.text_weight * text_loss
                 + self.global_weight * global_loss
                 + self.semantic_weight * (semantic_loss + self.text_weight * text_loss)
                 + self.dynamic_weight * dynamic_loss + self.fixed_weight * fixed_loss
                 + self.counter_weight * counter_loss + self.residual_weight * residual_loss
                 + self.diversity_weight * diversity)
        parts.update({"text_route": text_loss.detach(), "global_candidate_loss": global_loss.detach(),
                      "semantic_candidate_loss": semantic_loss.detach(), "dynamic_reference": dynamic_loss.detach(),
                      "fixed_reference": fixed_loss.detach(), "counterfactual_routing": counter_loss.detach(),
                      "band_residual": residual_loss.detach(), "candidate_gap": gap.detach(),
                      "candidate_diversity": diversity.detach(),
                      "quality_worst": fixed.max(1).values.mean().detach(),
                      "quality_mean": fixed.mean().detach()})
        return total, parts
