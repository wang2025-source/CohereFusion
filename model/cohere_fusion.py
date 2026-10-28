"""CohereFusion: GEWB / SEWB candidate generation, RSFR and SCR."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from model.components import (
    ImageEncoder, TransformerBlock, Upsample, Expert, FrozenSemanticTeacher,
    FrozenTextEncoder, SobelFeature, rgb_to_ycbcr, ycbcr_to_rgb,
)


ARCHITECTURE = "CohereFusion"
EXPERT_ORDER = ("V-low", "V-high", "V-gradient", "I-low", "I-high", "I-gradient")


def mean_filter(x, kernel):
    return F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2)


def frequency_bands(x):
    """Eq. (19): L + M + H exactly reconstructs x (zero-padded means)."""
    low, narrow = mean_filter(x, 7), mean_filter(x, 3)
    return torch.stack((low, narrow - low, x - narrow), dim=1)


def reliability(visible_y, infrared_y, nu=4., tau=6.):
    """Eq. (20): exposure reliability and infrared local activity."""
    visible_low, infrared_low = mean_filter(visible_y, 7), mean_filter(infrared_y, 7)
    return (nu * visible_low * (1 - visible_low),
            torch.tanh(tau * (infrared_y - infrared_low).abs()))


def blocks(channels, count, heads):
    return nn.Sequential(*[
        TransformerBlock(channels, heads, 2., False, "WithBias") for _ in range(count)
    ])


class SupplementaryEnhancement(nn.Module):
    """Eqs. (10)--(11): modality-specific gates and a separate refinement path."""

    def __init__(self, channels):
        super().__init__()
        self.modality_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3 * channels, channels, 1, bias=False),
                          nn.GELU(), nn.Conv2d(channels, 1, 3, padding=1))
            for _ in range(2)
        ])
        self.refine = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1, bias=False), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(math.gcd(16, channels), channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, visible, infrared, semantic, expert_features, weights):
        if weights.ndim == 2:
            weights = weights[:, :, None, None]
        weighted = expert_features * weights[:, :, None]
        global_v, global_i = weighted[:, :3].sum(1), weighted[:, 3:].sum(1)
        context = torch.cat((global_v, semantic, global_i), dim=1)
        gates = torch.cat([head(context) for head in self.modality_heads], 1).softmax(1)
        fused = gates[:, :1] * visible + self.refine(context) + gates[:, 1:] * infrared
        return fused, gates


class GEWB(nn.Module):
    """Section 3.3, Eqs. (5)--(11): six experts with global TopK routing."""

    def __init__(self, channels, semantic_channels=128, text_dim=512, top_k=4):
        super().__init__()
        if not 1 <= top_k <= 6:
            raise ValueError("GEWB top_k must be in [1, 6]")
        self.top_k = top_k
        self.semantic = nn.Conv2d(semantic_channels * 2, channels, 1, bias=False)
        self.sobel = SobelFeature(channels)
        self.experts = nn.ModuleList([Expert(channels, d) for d in (2, 1, 1, 2, 1, 1)])
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, channels), nn.GELU())
        self.router = nn.Sequential(nn.Linear(4 * channels, channels), nn.GELU(), nn.Linear(channels, 6))
        self.enhancement = SupplementaryEnhancement(channels)

    def semantic_context(self, semantic_pair, size):
        return self.semantic(F.interpolate(torch.cat(semantic_pair, 1), size,
                                           mode="bilinear", align_corners=False))

    def feature_experts(self, visible, infrared):
        vl, il = mean_filter(visible, 5), mean_filter(infrared, 5)
        inputs = (vl, visible - vl, self.sobel(visible), il, infrared - il, self.sobel(infrared))
        return torch.stack([expert(x) for expert, x in zip(self.experts, inputs)], 1)

    def forward(self, visible, infrared, semantic_pair, sentence, tokens=None, valid=None):
        semantic = self.semantic_context(semantic_pair, visible.shape[-2:])
        descriptor = torch.cat([x.mean((2, 3)) for x in (visible, infrared, semantic)]
                               + [self.text(sentence)], dim=1)
        logits = self.router(descriptor)
        values, indices = logits.topk(self.top_k, dim=1)
        weights = torch.full_like(logits, -torch.inf).scatter(1, indices, values).softmax(1)
        fused, gates = self.enhancement(visible, infrared, semantic,
                                       self.feature_experts(visible, infrared), weights)
        return fused, gates, weights


class SEWB(GEWB):
    """Eqs. (12)--(15): Phi_t = Phi_q + Z; dense global/local expert scores."""

    def __init__(self, channels, semantic_channels=128, text_dim=512):
        super().__init__(channels, semantic_channels, text_dim, top_k=6)
        # SEWB conditions its global scores on C_t, not GEWB's sentence MLP.
        del self.text
        self.query = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1, bias=False),
            nn.GroupNorm(math.gcd(16, channels), channels), nn.GELU(),
        )
        self.token_key = nn.Linear(text_dim, channels, bias=False)
        self.token_value = nn.Linear(text_dim, channels, bias=False)
        self.local_router = nn.Sequential(nn.Conv2d(4 * channels, channels, 1, bias=False),
                                          nn.GELU(), nn.Conv2d(channels, 6, 1))
        self.local_strength = nn.Parameter(torch.tensor(-3.))

    def ground_text(self, query_map, tokens, valid):
        size = query_map.shape[-2:]
        query = query_map
        if size[0] * size[1] > 1024:
            query = F.adaptive_avg_pool2d(query, (min(32, size[0]), min(32, size[1])))
        batch, channels, height, width = query.shape
        queries = query.flatten(2).transpose(1, 2)
        scores = queries @ self.token_key(tokens).transpose(1, 2) / math.sqrt(channels)
        scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
        context = (scores.softmax(-1) @ self.token_value(tokens)).transpose(1, 2)
        context = context.reshape(batch, channels, height, width)
        return F.interpolate(context, size, mode="bilinear", align_corners=False)

    def forward(self, visible, infrared, semantic_pair, sentence, tokens=None, valid=None):
        semantic = self.semantic_context(semantic_pair, visible.shape[-2:])
        query = self.query(torch.cat((visible, semantic, infrared), 1))
        text_feature = query + self.ground_text(query, tokens, valid)
        context = torch.cat((visible, semantic, infrared, text_feature), 1)
        logits = self.router(context.mean((2, 3)))[:, :, None, None]
        weights = (logits + torch.sigmoid(self.local_strength) * self.local_router(context)).softmax(1)
        fused, gates = self.enhancement(visible, infrared, semantic,
                                       self.feature_experts(visible, infrared), weights)
        return fused, gates, weights.mean((2, 3))


class CandidateBranch(nn.Module):
    """Independent visible/IR encoders, multiscale experts and decoder, Eqs. (1)--(18)."""

    def __init__(self, kind, dim=32, top_k=4, text_dim=512):
        super().__init__()
        if kind not in ("global", "semantic"):
            raise ValueError("Candidate branch must be global or semantic")
        self.encoder_v = ImageEncoder(3, dim, (2, 2, 2, 2), (1, 2, 4, 8), 2., False, "WithBias")
        self.encoder_i = ImageEncoder(3, dim, (2, 2, 2, 2), (1, 2, 4, 8), 2., False, "WithBias")
        self.semantic_reduce = nn.Sequential(nn.Conv2d(2048, 128, 1, bias=False),
                                             nn.GroupNorm(16, 128), nn.GELU())
        channels = [dim * 2 ** s for s in range(4)]
        self.expert_weighting = nn.ModuleList([
            GEWB(c, text_dim=text_dim, top_k=top_k) if kind == "global" else SEWB(c, text_dim=text_dim)
            for c in channels
        ])
        self.deep_refine = blocks(channels[3], 2, 8)
        self.up = nn.ModuleList([Upsample(channels[s]) for s in (3, 2, 1)])
        self.compress = nn.ModuleList([nn.Conv2d(2 * channels[s], channels[s], 1) for s in (2, 1, 0)])
        self.decode = nn.ModuleList([blocks(channels[s], 3 if s == 0 else 2, 2 ** s) for s in (2, 1, 0)])
        self.output_y = nn.Conv2d(dim, 1, 3, padding=1)
        self.output_scale = nn.Parameter(torch.tensor(-2.))
        nn.init.zeros_(self.output_y.weight)
        nn.init.zeros_(self.output_y.bias)

    def forward(self, visible, infrared, semantic_pair, sentence, tokens, valid):
        semantic = tuple(self.semantic_reduce(x.to(visible.dtype)) for x in semantic_pair)
        v_scales = self.encoder_v(visible)[::-1]
        i_scales = self.encoder_i(infrared)[::-1]
        fused, gates, routes = [], [], []
        for module, v, i in zip(self.expert_weighting, v_scales, i_scales):
            feature, gate, route = module(v, i, semantic, sentence, tokens, valid)
            fused.append(feature); gates.append(gate); routes.append(route)
        decoded = self.deep_refine(fused[-1])
        for up, compress, decode, skip in zip(self.up, self.compress, self.decode, fused[-2::-1]):
            decoded = decode(compress(torch.cat((up(decoded), skip), dim=1)))
        vy, cb, cr = rgb_to_ycbcr(visible)
        iy, _, _ = rgb_to_ycbcr(infrared)
        gate = F.interpolate(gates[0], visible.shape[-2:], mode="bilinear", align_corners=False)
        base = gate[:, :1] * vy + gate[:, 1:] * iy
        luminance = base + .15 * torch.sigmoid(self.output_scale) * torch.tanh(self.output_y(decoded))
        return luminance, ycbcr_to_rgb(luminance, cb, cr), {"routes": routes, "gates": gates}


class TextConditionedVisualEncoder(nn.Module):
    """Eqs. (21)--(22), independently instantiated by RSFR and SCR."""

    def __init__(self, text_dim=512, hidden=48, xi=.25):
        super().__init__()
        self.xi = xi
        self.visual = nn.Sequential(
            nn.Conv2d(12, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 1, bias=False), nn.GELU(),
        )
        self.text_film = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden * 2))

    def condition(self, visual, sentence):
        gamma, beta = self.text_film(sentence).chunk(2, dim=1)
        return visual * (1 + self.xi * torch.tanh(gamma)[:, :, None, None]) + self.xi * beta[:, :, None, None]


class RSFR(TextConditionedVisualEncoder):
    """Section 3.4: candidate weights and independent residual heads on F_L/F_M/F_H."""

    def __init__(self, text_dim=512, hidden=48):
        super().__init__(text_dim, hidden)
        self.route_heads = nn.ModuleList([nn.Conv2d(hidden, 2, 1) for _ in range(3)])
        self.band_residual_heads = nn.ModuleList([nn.Conv2d(1, 1, 3, padding=1) for _ in range(3)])
        self.residual_scale = nn.Parameter(torch.full((3,), -4.))
        for head in self.band_residual_heads:
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def routing_weights(self, visual, sentence):
        feature = self.condition(visual, sentence)
        return torch.stack([head(feature).softmax(1) for head in self.route_heads], 1)

    def forward(self, global_y, semantic_y, visible_y, infrared_y, sentence, counter_sentence=None):
        gb, sb = frequency_bands(global_y), frequency_bands(semantic_y)
        rv, ri = reliability(visible_y, infrared_y)
        inputs = torch.cat((global_y, semantic_y, visible_y, infrared_y,
                            *gb.unbind(1), *sb.unbind(1), rv, ri), dim=1)
        visual = self.visual(inputs)
        weights = self.routing_weights(visual, sentence)
        candidates = torch.stack((gb, sb), dim=2)
        bands = (weights.unsqueeze(3) * candidates).sum(2)
        residuals = torch.stack([torch.tanh(head(band)) for head, band in
                                 zip(self.band_residual_heads, bands.unbind(1))], 1)
        amplitudes = .08 * self.residual_scale.sigmoid()[None, :, None, None, None]
        luminance = (bands + amplitudes * residuals).sum(1)
        counter = None if counter_sentence is None else self.routing_weights(visual, counter_sentence)
        return luminance, {"rsfr_weights": weights, "counter_rsfr_weights": counter,
                           "band_residuals": residuals, "selected_bands": bands,
                           "visible_reliability": rv, "infrared_activity": ri}


class SCR(TextConditionedVisualEncoder):
    """Section 3.5, Eqs. (26)--(30): retain routed low frequency, refine high frequency."""

    def __init__(self, text_dim=512, hidden=48):
        super().__init__(text_dim, hidden)
        self.source_detail_head = nn.Conv2d(hidden, 2, 1)
        self.contrast_head = nn.Conv2d(hidden, 1, 1)
        self.detail_head = nn.Conv2d(hidden, 1, 1)
        self.contrast_strength = nn.Parameter(torch.tensor(-3.))
        self.detail_strength = nn.Parameter(torch.tensor(-3.))
        for head in (self.contrast_head, self.detail_head):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, routed_y, visible_y, infrared_y, sentence):
        low, vl, il = (mean_filter(x, 7) for x in (routed_y, visible_y, infrared_y))
        high, vh, ih = routed_y - low, visible_y - vl, infrared_y - il
        rv, ri = reliability(visible_y, infrared_y)
        inputs = torch.cat((routed_y, visible_y, infrared_y, low, vl, il, high, vh, ih,
                            rv, ri, (visible_y - infrared_y).abs()), dim=1)
        feature = self.condition(self.visual(inputs), sentence)
        weights = self.source_detail_head(feature).softmax(1)
        source_high = weights[:, :1] * vh + weights[:, 1:] * ih
        gain = 1 + .35 * self.contrast_strength.sigmoid() * self.contrast_head(feature).sigmoid()
        gate = self.detail_head(feature).sigmoid()
        detail = .15 * self.detail_strength.sigmoid() * gate * (source_high - high)
        refined = (low + gain * high + detail).clamp(0, 1)
        return refined, {"source_detail_weights": weights, "source_high": source_high,
                         "contrast_gain": gain, "detail_gate": gate, "routed_low": low}


class CohereFusion(nn.Module):
    """GEWB + SEWB -> RSFR -> SCR -> color reconstruction using visible chrominance."""

    architecture = ARCHITECTURE

    def __init__(self, clip_model, dim=32, global_top_k=4,
                 semantic_teacher=None, text_encoder=None):
        super().__init__()
        if dim < 8 or dim % 8:
            raise ValueError("dim must be a positive multiple of 8 (at least 8)")
        self.text_encoder = text_encoder if text_encoder is not None else FrozenTextEncoder(clip_model)
        self.semantic_teacher = semantic_teacher if semantic_teacher is not None else FrozenSemanticTeacher()
        self.text_encoder.requires_grad_(False).eval()
        self.semantic_teacher.requires_grad_(False).eval()
        self.gewb = CandidateBranch("global", dim, global_top_k)
        self.sewb = CandidateBranch("semantic", dim)
        self.rsfr = RSFR()
        self.scr = SCR()

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval(); self.semantic_teacher.eval()
        return self

    def forward(self, visible, infrared, text, counter_text=None, return_aux=False):
        if visible.shape != infrared.shape or visible.ndim != 4 or visible.shape[1] != 3:
            raise ValueError("Expected registered RGB inputs with matching [B, 3, H, W] shapes")
        if any(size < 8 or size % 8 for size in visible.shape[-2:]):
            raise ValueError("Input H and W must be multiples of 8")
        if text.ndim != 2 or text.shape[0] != visible.shape[0] or not text.ne(0).any(1).all():
            raise ValueError("Each image requires a non-empty tokenized text sequence")
        if counter_text is not None and (counter_text.shape != text.shape or not counter_text.ne(0).any(1).all()):
            raise ValueError("Counterfactual text must match the input text shape and contain valid tokens")
        with torch.no_grad():
            tokens, sentence, valid = self.text_encoder(text)
            semantics = self.semantic_teacher(visible, infrared)
            counter_sentence = None if counter_text is None else self.text_encoder(counter_text)[1]
        tokens, sentence = tokens.to(visible.dtype), sentence.to(visible.dtype)
        if counter_sentence is not None:
            counter_sentence = counter_sentence.to(visible.dtype)
        g, global_rgb, global_aux = self.gewb(visible, infrared, semantics, sentence, tokens, valid)
        s, semantic_rgb, semantic_aux = self.sewb(visible, infrared, semantics, sentence, tokens, valid)
        vy, cb, cr = rgb_to_ycbcr(visible)
        iy, _, _ = rgb_to_ycbcr(infrared)
        routed_y, routing_aux = self.rsfr(g, s, vy, iy, sentence, counter_sentence)
        refined_y, refinement_aux = self.scr(routed_y, vy, iy, sentence)
        output = ycbcr_to_rgb(refined_y, cb, cr)
        aux = {"global_candidate": global_rgb, "semantic_candidate": semantic_rgb,
               "global_luminance": g, "semantic_luminance": s,
               "gewb_aux": global_aux, "sewb_aux": semantic_aux,
               "routes": semantic_aux["routes"], "routed_y": routed_y, "fused_y": refined_y,
               **routing_aux, **refinement_aux}
        return (output, aux) if return_aux else output
