"""
Brightfield Bacteria Detector v5.0 — AnalysisPlugin for GUI
============================================================
Converted from the standalone v5.0 script following Plugin Integration Prompt rules.

Every detection step, constant, weight, kernel, and threshold is preserved exactly.
No file I/O, no visualization, no batch processing, no feature extraction.
Returns a boolean mask (dtype=bool).
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.filters import threshold_local
from skimage.feature import blob_log

from core.plugin_interface import AnalysisPlugin

# ═══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY PRESETS — exact values from v5.0 standalone script
# ═══════════════════════════════════════════════════════════════════════════════

PRESETS = {
    'conservative': {
        'background_sigma': 30,
        'clahe_clip_limit': 2.0,
        'clahe_tile_size': 8,
        'darkness_percentile': 20,
        'absolute_darkness': 95,
        'local_threshold_block_size': 15,
        'local_threshold_offset': 5,
        'local_k_factor': 2.0,
        'log_threshold': 0.02,
        'log_num_sigma': 8,
        'min_bacteria_size': 3,
        'max_bacteria_size': 100,
        'morph_open_size': 1,
        'morph_close_size': 2,
        'vote_threshold': 0.45,
    },
    'balanced': {
        'background_sigma': 35,
        'clahe_clip_limit': 2.5,
        'clahe_tile_size': 8,
        'darkness_percentile': 25,
        'absolute_darkness': 100,
        'local_threshold_block_size': 11,
        'local_threshold_offset': 4,
        'local_k_factor': 1.8,
        'log_threshold': 0.015,
        'log_num_sigma': 10,
        'min_bacteria_size': 2,
        'max_bacteria_size': 150,
        'morph_open_size': 1,
        'morph_close_size': 2,
        'vote_threshold': 0.40,
    },
    'sensitive': {
        'background_sigma': 40,
        'clahe_clip_limit': 3.0,
        'clahe_tile_size': 8,
        'darkness_percentile': 30,
        'absolute_darkness': 110,
        'local_threshold_block_size': 9,
        'local_threshold_offset': 3,
        'local_k_factor': 1.5,
        'log_threshold': 0.01,
        'log_num_sigma': 12,
        'min_bacteria_size': 1,
        'max_bacteria_size': 200,
        'morph_open_size': 1,
        'morph_close_size': 2,
        'vote_threshold': 0.35,
    },
    'ultra_sensitive': {
        'background_sigma': 50,
        'clahe_clip_limit': 4.0,
        'clahe_tile_size': 6,
        'darkness_percentile': 35,
        'absolute_darkness': 120,
        'local_threshold_block_size': 7,
        'local_threshold_offset': 2,
        'local_k_factor': 1.2,
        'log_threshold': 0.008,
        'log_num_sigma': 15,
        'min_bacteria_size': 1,
        'max_bacteria_size': 300,
        'morph_open_size': 1,
        'morph_close_size': 3,
        'vote_threshold': 0.30,
    },
}


class BrightfieldBacteriaDetectorPlugin(AnalysisPlugin):
    """
    Brightfield Bacteria Detector v5.0 — GUI Plugin.

    6-method weighted voting detection:
      1. Direct darkness threshold
      2. Multi-scale Black Top-Hat (5 kernels: 2×2, 3×3, 4×4, 5×5, 7×7)
      3. Laplacian of Gaussian blob detection
      4. Adaptive local thresholding
      5. Statistical thresholding (mean − k·std)
      6. CLAHE-enhanced darkness detection

    Weights: darkness=1.0, tophat=1.2, log=1.5, adaptive=1.0, statistical=1.3, clahe=1.0
    """

    # ───────────────────────────────────────────────────────────────────
    # Interface properties
    # ───────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "Brightfield Bacteria Detector v5.0"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "sensitivity": "balanced",
            "roi_top": 0,
            "roi_bottom": 0,
            "min_blob_sigma": 0.5,
            "max_blob_sigma": 3.0,
            "min_size": 2,
            "max_size": 150,
            "vote_threshold": 0.40,
            "skip_morphological_cleanup": True,
        }

    @property
    def parameters_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {
            "sensitivity": {
                "type": "list",
                "options": ["conservative", "balanced", "sensitive", "ultra_sensitive"],
                "default": "balanced",
                "tooltip": "Detection sensitivity preset. Higher = more detections, more false positives.",
            },
            "roi_top": {
                "type": int, "min": 0, "max": 5000, "default": 0,
                "tooltip": "0 = auto-detect. Positive value = manual ROI top boundary (pixels).",
            },
            "roi_bottom": {
                "type": int, "min": 0, "max": 5000, "default": 0,
                "tooltip": "0 = auto-detect. Positive value = manual ROI bottom boundary (pixels).",
            },
            "min_blob_sigma": {
                "type": float, "min": 0.1, "max": 5.0, "default": 0.5,
                "tooltip": "Min sigma for LoG blob detection. Smaller = smaller bacteria.",
            },
            "max_blob_sigma": {
                "type": float, "min": 0.5, "max": 10.0, "default": 3.0,
                "tooltip": "Max sigma for LoG blob detection.",
            },
            "min_size": {
                "type": int, "min": 1, "max": 50, "default": 2,
                "tooltip": "Minimum bacteria component size in pixels.",
            },
            "max_size": {
                "type": int, "min": 10, "max": 1000, "default": 150,
                "tooltip": "Maximum bacteria component size in pixels.",
            },
            "vote_threshold": {
                "type": float, "min": 0.1, "max": 1.0, "default": 0.40,
                "tooltip": "Weighted vote threshold. Lower = more detections.",
            },
            "skip_morphological_cleanup": {
                "type": bool, "default": True,
                "tooltip": "Skip morphological open/close to preserve tiny bacteria.",
            },
        }

    # ───────────────────────────────────────────────────────────────────
    # Main entry point
    # ───────────────────────────────────────────────────────────────────

    def run(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        try:
            # 1. Resolve parameters (preset + user overrides)
            cfg = self._resolve_params(params)

            # 2. Normalize input to uint8 grayscale
            gray = self._to_gray_uint8(image)
            if gray is None:
                return None

            # 3. ROI detection
            roi_top, roi_bottom = self._detect_roi(gray, cfg)
            roi = gray[roi_top:roi_bottom, :]

            print(f"  [{self.name}] ROI: {roi_top}-{roi_bottom} ({roi_bottom - roi_top}px)")

            # 4. Preprocessing
            clahe_roi = self._apply_clahe(roi, cfg)
            corrected = self._correct_background(roi, cfg)
            enhanced = self._enhance_dark_spots_multiscale(corrected)

            # 5. Core detection — 6-method weighted voting
            binary = self._detect_multi_method(roi, corrected, enhanced, clahe_roi, cfg)

            # 6. Post-processing
            if not cfg['skip_morphological_cleanup']:
                binary = self._morphological_cleanup(binary, cfg)

            filtered = self._filter_components(binary, cfg)

            # 7. Place ROI result into full-size boolean mask
            full_mask = np.zeros(gray.shape[:2], dtype=bool)
            full_mask[roi_top:roi_bottom, :] = filtered > 0
            return full_mask

        except Exception as exc:
            print(f"ERROR [{self.name}] run() failed: {exc}")
            return None

    # ───────────────────────────────────────────────────────────────────
    # Parameter resolution
    # ───────────────────────────────────────────────────────────────────

    def _resolve_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load sensitivity preset, then apply user overrides."""
        sensitivity = params.get("sensitivity", "balanced")
        cfg = PRESETS.get(sensitivity, PRESETS["balanced"]).copy()

        # User overrides
        cfg['roi_top'] = int(params.get("roi_top", 0))
        cfg['roi_bottom'] = int(params.get("roi_bottom", 0))
        cfg['min_blob_sigma'] = float(params.get("min_blob_sigma", 0.5))
        cfg['max_blob_sigma'] = float(params.get("max_blob_sigma", 3.0))
        cfg['min_bacteria_size'] = int(params.get("min_size", cfg['min_bacteria_size']))
        cfg['max_bacteria_size'] = int(params.get("max_size", cfg['max_bacteria_size']))
        cfg['vote_threshold'] = float(params.get("vote_threshold", cfg['vote_threshold']))
        cfg['skip_morphological_cleanup'] = bool(params.get("skip_morphological_cleanup", True))

        return cfg

    # ───────────────────────────────────────────────────────────────────
    # Input normalization
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray_uint8(image: np.ndarray) -> Optional[np.ndarray]:
        """Normalize any input (uint8/uint16/float, gray/RGB/RGBA) to uint8 grayscale."""
        if image is None or image.size == 0:
            return None

        img = image.copy()

        # Handle channel dimension
        if img.ndim == 3:
            if img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:  # RGB/BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = img[:, :, 0]

        # Handle dtype
        if img.dtype == np.uint8:
            return img
        elif img.dtype == np.uint16:
            return (img.astype(np.float64) / 65535.0 * 255.0).astype(np.uint8)
        elif np.issubdtype(img.dtype, np.floating):
            mn, mx = img.min(), img.max()
            if mx - mn < 1e-10:
                return np.zeros(img.shape[:2], dtype=np.uint8)
            return ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
        else:
            return img.astype(np.uint8)

    # ───────────────────────────────────────────────────────────────────
    # ROI detection
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_roi(img: np.ndarray, cfg: Dict) -> Tuple[int, int]:
        """Detect ROI — manual override or auto-detect gray region."""
        # Manual ROI
        if cfg['roi_top'] > 0 and cfg['roi_bottom'] > cfg['roi_top']:
            return cfg['roi_top'], min(cfg['roi_bottom'], img.shape[0] - 1)

        height, width = img.shape
        vertical_profile = np.mean(img, axis=1)
        smoothed = gaussian_filter1d(vertical_profile, sigma=10)
        gradient = np.abs(np.gradient(smoothed))
        gradient_smooth = gaussian_filter1d(gradient, sigma=5)

        # Top boundary
        top_search = gradient_smooth[:int(height * 0.4)]
        top_peaks = np.where(top_search > np.percentile(top_search, 90))[0]
        roi_top = top_peaks[-1] + 20 if len(top_peaks) > 0 else int(height * 0.35)

        # Bottom boundary
        bottom_region_start = int(height * 0.6)
        bottom_search = gradient_smooth[bottom_region_start:]
        bottom_peaks = np.where(bottom_search > np.percentile(bottom_search, 85))[0]
        roi_bottom = bottom_region_start + bottom_peaks[0] - 20 if len(bottom_peaks) > 0 else int(height * 0.80)

        # Gray region refinement
        overall_mean = np.mean(smoothed)
        gray_tolerance = np.std(smoothed) * 0.8
        gray_rows = np.where(
            (smoothed > overall_mean - gray_tolerance) &
            (smoothed < overall_mean + gray_tolerance * 0.5)
        )[0]
        if len(gray_rows) > 50:
            roi_top = max(roi_top, gray_rows[0] - 30)
            roi_bottom = min(roi_bottom, gray_rows[-1] + 30)

        # Sanity checks
        roi_top = max(int(height * 0.30), min(roi_top, int(height * 0.45)))
        roi_bottom = max(int(height * 0.65), min(roi_bottom, int(height * 0.85)))

        min_roi_height = int(height * 0.30)
        if roi_bottom - roi_top < min_roi_height:
            center = (roi_top + roi_bottom) // 2
            roi_top = center - min_roi_height // 2
            roi_bottom = center + min_roi_height // 2

        roi_top = max(10, roi_top)
        roi_bottom = min(height - 10, roi_bottom)

        return roi_top, roi_bottom

    # ───────────────────────────────────────────────────────────────────
    # Preprocessing helpers
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_clahe(img: np.ndarray, cfg: Dict) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=cfg['clahe_clip_limit'],
            tileGridSize=(cfg['clahe_tile_size'], cfg['clahe_tile_size'])
        )
        return clahe.apply(img)

    @staticmethod
    def _correct_background(img: np.ndarray, cfg: Dict) -> np.ndarray:
        background = gaussian_filter(img.astype(float), sigma=cfg['background_sigma'])
        corrected = img.astype(float) - background + 128
        return np.clip(corrected, 0, 255).astype(np.uint8)

    @staticmethod
    def _enhance_dark_spots_multiscale(img: np.ndarray) -> np.ndarray:
        """5 kernel sizes: (2,2), (3,3), (4,4), (5,5), (7,7)."""
        kernels = [(2, 2), (3, 3), (4, 4), (5, 5), (7, 7)]
        blackhat_results = []
        for ks in kernels:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ks)
            blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
            blackhat_results.append(blackhat)
        return np.maximum.reduce(blackhat_results)

    # ───────────────────────────────────────────────────────────────────
    # Detection methods
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_blobs_log(img: np.ndarray, cfg: Dict) -> np.ndarray:
        inverted = 255 - img
        img_norm = inverted.astype(float) / 255.0
        try:
            blobs = blob_log(
                img_norm,
                min_sigma=cfg['min_blob_sigma'],
                max_sigma=cfg['max_blob_sigma'],
                num_sigma=cfg['log_num_sigma'],
                threshold=cfg['log_threshold']
            )
            mask = np.zeros_like(img, dtype=np.uint8)
            for blob in blobs:
                y, x, sigma = blob
                radius = max(1, int(sigma * np.sqrt(2)))
                cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
            return mask
        except Exception:
            return np.zeros_like(img, dtype=np.uint8)

    @staticmethod
    def _adaptive_local_threshold(img: np.ndarray, cfg: Dict) -> np.ndarray:
        block_size = cfg['local_threshold_block_size']
        if block_size % 2 == 0:
            block_size += 1
        local_thresh = threshold_local(img, block_size=block_size,
                                        offset=cfg['local_threshold_offset'])
        return (img < local_thresh).astype(np.uint8) * 255

    @staticmethod
    def _statistical_local_threshold(img: np.ndarray, cfg: Dict) -> np.ndarray:
        kernel_size = cfg['local_threshold_block_size']
        if kernel_size % 2 == 0:
            kernel_size += 1
        local_mean = cv2.blur(img.astype(float), (kernel_size, kernel_size))
        local_sq_mean = cv2.blur((img.astype(float)) ** 2, (kernel_size, kernel_size))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
        k = cfg['local_k_factor']
        threshold = local_mean - k * local_std
        return ((img.astype(float) < threshold) & (local_std > 2)).astype(np.uint8) * 255

    def _detect_multi_method(self, roi, corrected, enhanced, clahe_img, cfg) -> np.ndarray:
        """6-method weighted voting — exact weights from v5.0."""

        # Method 1: Direct darkness
        dark_threshold = np.percentile(corrected, cfg['darkness_percentile'])
        dark_threshold = min(dark_threshold, cfg['absolute_darkness'])
        _, binary_dark = cv2.threshold(corrected, dark_threshold, 255, cv2.THRESH_BINARY_INV)

        # Method 2: Top-hat enhanced
        enhanced_mean = np.mean(enhanced)
        enhanced_std = np.std(enhanced)
        enhanced_threshold = enhanced_mean + (2.0 * enhanced_std)
        _, binary_enhanced = cv2.threshold(enhanced, enhanced_threshold, 255, cv2.THRESH_BINARY)

        # Method 3: LoG blob detection
        binary_log = self._detect_blobs_log(clahe_img, cfg)

        # Method 4: Adaptive local threshold
        binary_adaptive = self._adaptive_local_threshold(corrected, cfg)

        # Method 5: Statistical local threshold
        binary_statistical = self._statistical_local_threshold(corrected, cfg)

        # Method 6: CLAHE + darkness
        clahe_threshold = np.percentile(clahe_img, cfg['darkness_percentile'] - 5)
        _, binary_clahe = cv2.threshold(clahe_img, clahe_threshold, 255, cv2.THRESH_BINARY_INV)

        # Weighted voting — exact weights
        weights = {
            'darkness': 1.0, 'tophat': 1.2, 'log': 1.5,
            'adaptive': 1.0, 'statistical': 1.3, 'clahe': 1.0
        }
        combined = np.zeros_like(roi, dtype=np.float32)
        combined += (binary_dark / 255.0) * weights['darkness']
        combined += (binary_enhanced / 255.0) * weights['tophat']
        combined += (binary_log / 255.0) * weights['log']
        combined += (binary_adaptive / 255.0) * weights['adaptive']
        combined += (binary_statistical / 255.0) * weights['statistical']
        combined += (binary_clahe / 255.0) * weights['clahe']

        total_weight = sum(weights.values())
        return ((combined / total_weight) >= cfg['vote_threshold']).astype(np.uint8) * 255

    # ───────────────────────────────────────────────────────────────────
    # Post-processing
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _morphological_cleanup(binary: np.ndarray, cfg: Dict) -> np.ndarray:
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg['morph_open_size'], cfg['morph_open_size'])
        )
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg['morph_close_size'], cfg['morph_close_size'])
        )
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        return closed

    @staticmethod
    def _filter_components(binary: np.ndarray, cfg: Dict) -> np.ndarray:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        final_mask = np.zeros_like(binary)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (cfg['min_bacteria_size'] <= area <= cfg['max_bacteria_size']):
                continue
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]
            aspect_ratio = max(w, h) / (min(w, h) + 1)
            max_aspect = 20 if area < 10 else 15
            if aspect_ratio > max_aspect:
                continue
            final_mask[labels == i] = 255

        return final_mask
