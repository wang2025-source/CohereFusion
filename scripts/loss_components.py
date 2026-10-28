"""Reconstruction, information-preservation and differentiable quality terms."""

import torch
from torch import nn
from torch.nn import functional as F
from model.components import rgb_to_ycbcr


def correlation(a, b):
    a = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    b = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    return (a * b).sum(1) / ((a.square().sum(1) * b.square().sum(1)).clamp_min(1e-12).sqrt() + 1e-6)


class BaseFusionLoss(nn.Module):
    """Optimize EN/MI/Qabf directly while retaining structure and clean detail."""

    def __init__(self, bins=64, entropy_margin=.12, mi_floor=2.8, scd_floor=1.55, scd_weight=1.5):
        super().__init__()
        if bins < 2:
            raise ValueError("Soft histogram requires at least two bins")
        kx = torch.tensor(((-1., 0., 1.), (-2., 0., 2.), (-1., 0., 1.)))[None, None]
        self.register_buffer("kx", kx)
        self.register_buffer("ky", kx.transpose(-1, -2))
        self.register_buffer("lap", torch.tensor(((0., 1., 0.), (1., -4., 1.), (0., 1., 0.)))[None, None])
        self.register_buffer("centers", torch.linspace(0, 1, bins))
        self.entropy_margin, self.mi_floor = entropy_margin, mi_floor
        self.scd_floor, self.scd_weight = scd_floor, scd_weight

    @staticmethod
    def y(x):
        return rgb_to_ycbcr(x)[0].clamp(0, 1)

    def membership(self, x):
        x = F.adaptive_avg_pool2d(x, (32, 32)).flatten(1)
        distance = x[:, :, None] - self.centers[None, None, :]
        weights = torch.exp(-.5 * (distance / .018).square())
        return weights / (weights.sum(-1, keepdim=True) + 1e-8)

    @staticmethod
    def entropy_from_membership(weights):
        probability = weights.mean(1)
        return -(probability * torch.log2(probability + 1e-8)).sum(1)

    def soft_mi(self, a, b):
        wa, wb = self.membership(a), self.membership(b)
        joint = torch.einsum("bnk,bnl->bkl", wa, wb) / wa.shape[1]
        pa, pb = joint.sum(2, keepdim=True), joint.sum(1, keepdim=True)
        return (joint * torch.log2((joint + 1e-8) / (pa * pb + 1e-8))).sum((1, 2))

    def gradients(self, x):
        padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
        gx, gy = F.conv2d(padded, self.kx), F.conv2d(padded, self.ky)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
        laplace = F.conv2d(padded, self.lap).abs()
        return gx, gy, magnitude, laplace

    @staticmethod
    def ssim_loss(a, b):
        ma, mb = F.avg_pool2d(a, 7, 1, 3), F.avg_pool2d(b, 7, 1, 3)
        va = F.avg_pool2d(a.square(), 7, 1, 3) - ma.square()
        vb = F.avg_pool2d(b.square(), 7, 1, 3) - mb.square()
        vab = F.avg_pool2d(a * b, 7, 1, 3) - ma * mb
        score = ((2 * ma * mb + .01 ** 2) * (2 * vab + .03 ** 2) /
                 ((ma.square() + mb.square() + .01 ** 2) * (va + vb + .03 ** 2) + 1e-8))
        return 1 - score.mean()

    @staticmethod
    def local_std(x):
        mean = F.avg_pool2d(x, 7, 1, 3)
        return torch.sqrt((F.avg_pool2d(x.square(), 7, 1, 3) - mean.square()).clamp_min(0) + 1e-6)

    @staticmethod
    def edge_quality(sx, sy, sm, fx, fy, fm):
        ratio = torch.minimum(sm / (fm + 1e-4), fm / (sm + 1e-4)).clamp(0, 1)
        direction = ((sx * fx + sy * fy).abs() / (sm * fm + 1e-4)).clamp(0, 1)
        return (.9994 * torch.sigmoid(15 * (ratio - .5)) *
                .9879 * torch.sigmoid(22 * (direction - .8)))

    def forward(self, visible, infrared, visible_gt, infrared_gt, fused, aux, tasks=None):
        # Metric terms deliberately use the degraded inputs because the test protocol does.
        v, i, f = self.y(visible), self.y(infrared), self.y(fused)
        vg, ig = self.y(visible_gt), self.y(infrared_gt)
        vx, vy, vm, _ = self.gradients(v); ix, iy, im, _ = self.gradients(i)
        fx, fy, fm, fl = self.gradients(f)
        _, _, vgm, vgl = self.gradients(vg); _, _, igm, igl = self.gradients(ig)

        # Direct differentiable counterparts of entropy and base-2 MI.
        wf, wv, wi = self.membership(f), self.membership(v), self.membership(i)
        ef = self.entropy_from_membership(wf)
        ev, ei = self.entropy_from_membership(wv), self.entropy_from_membership(wi)
        entropy = F.relu(torch.maximum(ev, ei) + self.entropy_margin - ef).mean()
        mi_value = self.soft_mi(f, v) + self.soft_mi(f, i)
        mi = F.relu(self.mi_floor - mi_value).mean()

        # Differentiable Xydeas-Petrovic edge preservation (Qabf-like).
        qv = self.edge_quality(vx, vy, vm, fx, fy, fm)
        qi = self.edge_quality(ix, iy, im, fx, fy, fm)
        qabf_value = ((qv * vm + qi * im).sum((1, 2, 3)) /
                      ((vm + im).sum((1, 2, 3)) + 1e-6)).mean()
        qabf = 1 - qabf_value

        # Clean targets control noise while source terms preserve evaluated information.
        saliency = torch.softmax(torch.cat((vgm, igm), 1) * 3, 1)
        target = saliency[:, :1] * vg + saliency[:, 1:] * ig
        intensity = F.smooth_l1_loss(f, target)
        edge = F.l1_loss(fm, torch.maximum(vgm, igm))
        laplacian = F.l1_loss(fl, torch.maximum(vgl, igl))
        ssim = .5 * (self.ssim_loss(f, v) + self.ssim_loss(f, i))
        contrast = F.l1_loss(self.local_std(f), torch.maximum(self.local_std(vg), self.local_std(ig)))

        _, fcb, fcr = rgb_to_ycbcr(fused); _, vcb, vcr = rgb_to_ycbcr(visible_gt)
        color = F.l1_loss(fcb, vcb) + F.l1_loss(fcr, vcr)
        importance = torch.stack([route.mean(0) for route in aux["routes"]]).mean(0)
        load = importance.var(unbiased=False) / (importance.mean().square() + 1e-6)

        total = (2 * intensity + 3 * edge + laplacian + 1.5 * ssim + contrast +
                 2.5 * entropy + 2.0 * mi + 1.5 * qabf + color + .03 * load)
        values = {"intensity": intensity, "edge": edge, "laplacian": laplacian,
                  "ssim": ssim, "contrast": contrast, "entropy": entropy,
                  "entropy_fused": ef.mean(), "mi": mi, "mi_value": mi_value.mean(),
                  "qabf": qabf, "qabf_value": qabf_value, "color": color, "load": load}
        scd_value = (correlation(i, f - v) + correlation(v, f - i)).mean()
        scd = F.relu(self.scd_floor - scd_value)
        total = total + self.scd_weight * scd
        values.update(scd=scd, scd_value=scd_value)
        return total, {key: value.detach() for key, value in values.items()}

    metric_names = ("EN", "SD", "SF", "AG", "MI", "SSIM", "SCD", "Qabf")

    @staticmethod
    def _ssim_per_image(a, b):
        ma, mb = F.avg_pool2d(a, 7, 1, 3), F.avg_pool2d(b, 7, 1, 3)
        va = F.avg_pool2d(a.square(), 7, 1, 3) - ma.square()
        vb = F.avg_pool2d(b.square(), 7, 1, 3) - mb.square()
        vab = F.avg_pool2d(a * b, 7, 1, 3) - ma * mb
        score = ((2 * ma * mb + .01 ** 2) * (2 * vab + .03 ** 2) /
                 ((ma.square() + mb.square() + .01 ** 2) *
                  (va + vb + .03 ** 2) + 1e-8))
        return score.flatten(1).mean(1)

    def metric_vector(self, visible, infrared, fused):
        v, i, f = (rgb_to_ycbcr(x)[0].clamp(0, 1)
                   for x in (visible, infrared, fused))
        vx, vy, vm, _ = self.gradients(v)
        ix, iy, im, _ = self.gradients(i)
        fx, fy, fm, _ = self.gradients(f)

        entropy = self.entropy_from_membership(self.membership(f))
        sd = f.flatten(1).std(1)
        dx = f[..., :, 1:] - f[..., :, :-1]
        dy = f[..., 1:, :] - f[..., :-1, :]
        sf = torch.sqrt(dx.square().mean((1, 2, 3)) +
                        dy.square().mean((1, 2, 3)) + 1e-8)
        ag = .5 * (dx.abs().mean((1, 2, 3)) + dy.abs().mean((1, 2, 3)))
        mi = self.soft_mi(f, v) + self.soft_mi(f, i)
        ssim = .5 * (self._ssim_per_image(f, v) + self._ssim_per_image(f, i))
        scd = correlation(i, f - v) + correlation(v, f - i)
        qv = self.edge_quality(vx, vy, vm, fx, fy, fm)
        qi = self.edge_quality(ix, iy, im, fx, fy, fm)
        qabf = ((qv * vm + qi * im).sum((1, 2, 3)) /
                ((vm + im).sum((1, 2, 3)) + 1e-6))
        return torch.stack((entropy, sd, sf, ag, mi, ssim, scd, qabf), dim=1)
