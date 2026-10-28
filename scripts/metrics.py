"""Classical image-fusion metrics used by the official CDDFuse evaluator.

The formulas and constants follow:
https://github.com/Zhaozixiang1228/MMIF-CDDFuse/blob/main/utils/Evaluator.py

All inputs must be aligned 2-D grayscale arrays in the 0..255 range.
"""

import math

import numpy as np
import sklearn.metrics as skm
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim


class FusionMetrics:
    @staticmethod
    def _check(fused, infrared=None, visible=None):
        if infrared is None:
            assert isinstance(fused, np.ndarray) and fused.ndim == 2
            return
        assert type(fused) is type(infrared) is type(visible) is np.ndarray
        assert fused.shape == infrared.shape == visible.shape
        assert fused.ndim == 2

    @classmethod
    def en(cls, image):
        cls._check(image)
        values = np.uint8(np.round(image)).ravel()
        histogram = np.bincount(values) / values.size
        return -sum(histogram * np.log2(histogram + (histogram == 0)))

    @classmethod
    def sd(cls, image):
        cls._check(image)
        return np.std(image)

    @classmethod
    def sf(cls, image):
        cls._check(image)
        row = np.mean((image[:, 1:] - image[:, :-1]) ** 2)
        column = np.mean((image[1:, :] - image[:-1, :]) ** 2)
        return np.sqrt(row + column)

    @classmethod
    def mi(cls, fused, infrared, visible):
        cls._check(fused, infrared, visible)
        return (
            skm.mutual_info_score(fused.ravel(), infrared.ravel())
            + skm.mutual_info_score(fused.ravel(), visible.ravel())
        )

    @staticmethod
    def _corr(first, second):
        first_centered = first - np.mean(first)
        second_centered = second - np.mean(second)
        denominator = np.sqrt(
            np.sum(first_centered**2) * np.sum(second_centered**2)
        )
        if denominator == 0:
            return 0.0
        return np.sum(first_centered * second_centered) / denominator

    @classmethod
    def scd(cls, fused, infrared, visible):
        cls._check(fused, infrared, visible)
        return cls._corr(infrared, fused - visible) + cls._corr(
            visible, fused - infrared
        )

    @classmethod
    def vif(cls, fused, infrared, visible):
        cls._check(fused, infrared, visible)
        return cls._vif_pair(infrared, fused) + cls._vif_pair(visible, fused)

    @staticmethod
    def _vif_pair(reference, distorted):
        sigma_nsq = 2
        eps = 1e-10
        numerator = 0.0
        denominator = 0.0

        for scale in range(1, 5):
            size = 2 ** (4 - scale + 1) + 1
            sigma = size / 5.0
            midpoint = (size - 1.0) / 2.0
            y, x = np.ogrid[-midpoint : midpoint + 1, -midpoint : midpoint + 1]
            window = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
            window[window < np.finfo(window.dtype).eps * window.max()] = 0
            window /= window.sum()

            if scale > 1:
                reference = convolve2d(
                    reference, np.rot90(window, 2), mode="valid"
                )[::2, ::2]
                distorted = convolve2d(
                    distorted, np.rot90(window, 2), mode="valid"
                )[::2, ::2]

            mean_reference = convolve2d(
                reference, np.rot90(window, 2), mode="valid"
            )
            mean_distorted = convolve2d(
                distorted, np.rot90(window, 2), mode="valid"
            )
            variance_reference = (
                convolve2d(reference * reference, np.rot90(window, 2), mode="valid")
                - mean_reference * mean_reference
            )
            variance_distorted = (
                convolve2d(distorted * distorted, np.rot90(window, 2), mode="valid")
                - mean_distorted * mean_distorted
            )
            covariance = (
                convolve2d(reference * distorted, np.rot90(window, 2), mode="valid")
                - mean_reference * mean_distorted
            )

            variance_reference[variance_reference < 0] = 0
            variance_distorted[variance_distorted < 0] = 0
            gain = covariance / (variance_reference + eps)
            residual_variance = variance_distorted - gain * covariance

            low_reference = variance_reference < eps
            gain[low_reference] = 0
            residual_variance[low_reference] = variance_distorted[low_reference]
            variance_reference[low_reference] = 0

            low_distorted = variance_distorted < eps
            gain[low_distorted] = 0
            residual_variance[low_distorted] = 0

            negative_gain = gain < 0
            residual_variance[negative_gain] = variance_distorted[negative_gain]
            gain[negative_gain] = 0
            residual_variance[residual_variance <= eps] = eps

            numerator += np.sum(
                np.log10(
                    1
                    + gain * gain * variance_reference
                    / (residual_variance + sigma_nsq)
                )
            )
            denominator += np.sum(
                np.log10(1 + variance_reference / sigma_nsq)
            )

        value = numerator / denominator
        return 1.0 if np.isnan(value) else value

    @classmethod
    def qabf(cls, fused, infrared, visible):
        cls._check(fused, infrared, visible)
        gradient_ir, angle_ir = cls._edge_arrays(infrared)
        gradient_vis, angle_vis = cls._edge_arrays(visible)
        gradient_fused, angle_fused = cls._edge_arrays(fused)
        quality_ir = cls._edge_quality(
            angle_ir, gradient_ir, angle_fused, gradient_fused
        )
        quality_vis = cls._edge_quality(
            angle_vis, gradient_vis, angle_fused, gradient_fused
        )
        denominator = np.sum(gradient_ir + gradient_vis)
        if denominator == 0:
            return 0.0
        numerator = np.sum(
            quality_ir * gradient_ir + quality_vis * gradient_vis
        )
        return numerator / denominator

    @staticmethod
    def _edge_arrays(image):
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        x_gradient = convolve2d(image, sobel_x, mode="same")
        y_gradient = convolve2d(image, sobel_y, mode="same")
        magnitude = np.sqrt(x_gradient * x_gradient + y_gradient * y_gradient)
        angle = np.zeros_like(image)
        angle[x_gradient == 0] = math.pi / 2
        nonzero = x_gradient != 0
        angle[nonzero] = np.arctan(y_gradient[nonzero] / x_gradient[nonzero])
        return magnitude, angle

    @staticmethod
    def _edge_quality(source_angle, source_gradient, fused_angle, fused_gradient):
        gradient_ratio = np.zeros_like(source_angle)
        source_larger = source_gradient > fused_gradient
        equal = source_gradient == fused_gradient
        fused_larger = source_gradient < fused_gradient
        with np.errstate(divide="ignore", invalid="ignore"):
            gradient_ratio[source_larger] = (
                fused_gradient[source_larger] / source_gradient[source_larger]
            )
            gradient_ratio[equal] = fused_gradient[equal]
            gradient_ratio[fused_larger] = (
                source_gradient[fused_larger] / fused_gradient[fused_larger]
            )
        angle_ratio = 1 - np.abs(source_angle - fused_angle) / (math.pi / 2)
        gradient_quality = 0.9994 / (1 + np.exp(-15 * (gradient_ratio - 0.5)))
        angle_quality = 0.9879 / (1 + np.exp(-22 * (angle_ratio - 0.8)))
        return gradient_quality * angle_quality

    @classmethod
    def ssim_sum(cls, fused, infrared, visible):
        cls._check(fused, infrared, visible)
        return ssim(fused, infrared, data_range=255) + ssim(
            fused, visible, data_range=255
        )

    @classmethod
    def all(cls, fused, infrared, visible):
        return {
            "EN": cls.en(fused),
            "SD": cls.sd(fused),
            "SF": cls.sf(fused),
            "MI": cls.mi(fused, infrared, visible),
            "SCD": cls.scd(fused, infrared, visible),
            "VIF": cls.vif(fused, infrared, visible),
            "Qabf": cls.qabf(fused, infrared, visible),
            "SSIM_sum": cls.ssim_sum(fused, infrared, visible),
        }
