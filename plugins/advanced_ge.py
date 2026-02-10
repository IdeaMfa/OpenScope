import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.filters import threshold_local
from skimage.feature import blob_log
from core.plugin_interface import AnalysisPlugin
from typing import Dict, Any, Optional, Tuple

"""
Advanced Brightfield Bacteria Detector v4.5
============================================
Faithful port of Okan's BrightfieldBacteriaDetectorV45.

Original script'ten KORUNAN her şey:
  - ROI auto-detection (middle gray band) + manual ROI desteği
  - CLAHE ön işleme
  - Gaussian background correction
  - Multi-scale black top-hat (2x2, 3x3, 4x4, 5x5, 7x7)
  - 6-method weighted vote:
      1) Global darkness threshold          (w=1.0)
      2) Multi-scale top-hat threshold      (w=1.2)
      3) LoG blob detection                 (w=1.5)
      4) Adaptive local threshold           (w=1.0)
      5) Statistical local threshold        (w=1.3)
      6) CLAHE + darkness threshold         (w=1.0)
  - Morphological open → close
  - Connected component filtering (size + aspect ratio)

Sensitivity presets: conservative / balanced / sensitive / ultra_sensitive
Her preset orijinal script'teki değerlerin AYNISINI kullanır.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Sensitivity presets — orijinal script'ten birebir
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
    },
}


class AdvancedBacteriaDetector(AnalysisPlugin):
    """
    Advanced Brightfield Bacteria Detector v4.5 — GUI Plugin.
    Orijinal script'in birebir portu.
    """

    # ══════════════════════════════════════════════════════════════════════
    # AnalysisPlugin interface
    # ══════════════════════════════════════════════════════════════════════

    @property
    def name(self) -> str:
        return "Advanced Bacteria Detector (v4.5)"

    @property
    def parameters_metadata(self) -> Dict[str, Dict[str, Any]]:
        """GUI kontrolleri için parametre tanımları."""
        return {
            "sensitivity": {
                "type": "list",
                "options": ["conservative", "balanced", "sensitive", "ultra_sensitive"],
                "default": "balanced",
            },
            "roi_top": {
                "type": int, "min": 0, "max": 5000, "default": 0,
                "tooltip": "0 = auto-detect. Pozitif değer = manuel ROI üst sınırı (piksel).",
            },
            "roi_bottom": {
                "type": int, "min": 0, "max": 5000, "default": 0,
                "tooltip": "0 = auto-detect. Pozitif değer = manuel ROI alt sınırı (piksel).",
            },
            "min_blob_sigma": {
                "type": float, "min": 0.1, "max": 5.0, "default": 0.5,
                "tooltip": "LoG minimum sigma — küçültmek daha küçük bakterileri yakalar.",
            },
            "max_blob_sigma": {
                "type": float, "min": 1.0, "max": 20.0, "default": 3.0,
                "tooltip": "LoG maximum sigma.",
            },
            "min_size": {
                "type": int, "min": 1, "max": 500, "default": 2,
                "tooltip": "Minimum bileşen alanı (piksel).",
            },
            "max_size": {
                "type": int, "min": 10, "max": 5000, "default": 150,
                "tooltip": "Maksimum bileşen alanı (piksel).",
            },
            "vote_threshold": {
                "type": float, "min": 0.1, "max": 0.9, "default": 0.40,
                "tooltip": "Ağırlıklı oylama eşiği (0.0–1.0). Düşük = daha çok tespit.",
            },
        }

    @property
    def parameters(self) -> Dict[str, Any]:
        """Varsayılan parametreler — GUI bu dict'i slider/spinbox'lara çevirir."""
        return {
            "sensitivity": "balanced",
            "roi_top": 0,
            "roi_bottom": 0,
            "min_blob_sigma": 0.5,
            "max_blob_sigma": 3.0,
            "min_size": 2,
            "max_size": 150,
            "vote_threshold": 0.40,
        }

    def run(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Ana giriş noktası — GUI buradan çağırır.

        Args:
            image: Giriş görüntüsü (herhangi bir renk/bit derinliği).
            params: Kullanıcının ayarladığı parametreler.

        Returns:
            Boolean maske (True = bakteri) veya None.
        """
        try:
            # Sensitivity preset + kullanıcı override'larını birleştir
            sensitivity = params.get("sensitivity", "balanced")
            cfg = PRESETS.get(sensitivity, PRESETS["balanced"]).copy()

            # Kullanıcı parametreleri preset'in üstüne yazılır
            cfg["roi_top"] = int(params.get("roi_top", 0))
            cfg["roi_bottom"] = int(params.get("roi_bottom", 0))
            cfg["min_blob_sigma"] = float(params.get("min_blob_sigma", 0.5))
            cfg["max_blob_sigma"] = float(params.get("max_blob_sigma", 3.0))
            cfg["min_bacteria_size"] = int(params.get("min_size", cfg["min_bacteria_size"]))
            cfg["max_bacteria_size"] = int(params.get("max_size", cfg["max_bacteria_size"]))
            cfg["vote_threshold"] = float(params.get("vote_threshold", 0.40))

            # ── Görüntüyü uint8 grayscale'e çevir ──
            gray = self._to_gray_uint8(image)
            if gray is None:
                return None

            # ── ROI tespiti ──
            roi_top, roi_bottom = self._detect_roi(gray, cfg)
            roi = gray[roi_top:roi_bottom, :]
            print(f"DEBUG [{self.name}] ROI: {roi_top}–{roi_bottom} ({roi_bottom - roi_top}px)")

            # ── Ön işleme ──
            clahe_roi = self._apply_clahe(roi, cfg)
            corrected = self._correct_background(roi, cfg)
            enhanced = self._enhance_dark_spots_multiscale(corrected)

            # ── 6-method weighted vote ──
            binary_roi = self._multi_method_detect(roi, corrected, enhanced, clahe_roi, cfg)

            # ── Morfolojik temizlik ──
            cleaned = self._morphological_cleanup(binary_roi, cfg)

            # ── Bileşen filtreleme (boyut + aspect ratio) ──
            filtered = self._filter_components(cleaned, cfg)

            # ── Tam boyut maskeye yerleştir ──
            full_mask = np.zeros(gray.shape[:2], dtype=bool)
            full_mask[roi_top:roi_bottom, :] = filtered > 0

            count = np.count_nonzero(full_mask)
            print(f"DEBUG [{self.name}] {count} bacteria pixels detected.")
            return full_mask

        except Exception as exc:
            print(f"ERROR [{self.name}] run() failed: {exc}")
            import traceback
            traceback.print_exc()
            return None

    # ══════════════════════════════════════════════════════════════════════
    # Görüntü hazırlık
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _to_gray_uint8(image: np.ndarray) -> Optional[np.ndarray]:
        """Herhangi bir giriş formatını uint8 grayscale'e çevirir."""
        if image is None or image.size == 0:
            return None

        img = image.copy()

        # Renk → gri
        if img.ndim == 3:
            channels = img.shape[2]
            if channels == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img = img[:, :, 0]

        # Bit derinliği normalizasyonu
        if img.dtype == np.uint16:
            img = (img.astype(np.float64) / 65535.0 * 255.0).astype(np.uint8)
        elif img.dtype == np.float32 or img.dtype == np.float64:
            mn, mx = img.min(), img.max()
            if mx - mn > 0:
                img = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        elif img.dtype != np.uint8:
            mn, mx = img.min(), img.max()
            if mx - mn > 0:
                img = ((img.astype(np.float64) - mn) / (mx - mn) * 255.0).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        return img

    # ══════════════════════════════════════════════════════════════════════
    # ROI Detection — orijinal script'ten birebir
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_roi(img: np.ndarray, cfg: Dict) -> Tuple[int, int]:
        """
        Orta gri bant ROI'sini tespit eder.
        roi_top/roi_bottom > 0 ise manuel ROI kullanılır.
        """
        manual_top = cfg.get("roi_top", 0)
        manual_bot = cfg.get("roi_bottom", 0)
        if manual_top > 0 and manual_bot > manual_top:
            return manual_top, min(manual_bot, img.shape[0] - 1)

        height, width = img.shape[:2]

        # Dikey profil + yumuşatma
        vertical_profile = np.mean(img, axis=1)
        smoothed = gaussian_filter1d(vertical_profile, sigma=10)
        gradient = np.abs(np.gradient(smoothed))
        gradient_smooth = gaussian_filter1d(gradient, sigma=5)

        # ── Üst sınır ──
        top_search = gradient_smooth[:int(height * 0.4)]
        top_peaks = np.where(top_search > np.percentile(top_search, 90))[0]
        roi_top = (top_peaks[-1] + 20) if len(top_peaks) > 0 else int(height * 0.35)

        # ── Alt sınır ──
        bot_start = int(height * 0.6)
        bot_search = gradient_smooth[bot_start:]
        bot_peaks = np.where(bot_search > np.percentile(bot_search, 85))[0]
        roi_bottom = (bot_start + bot_peaks[0] - 20) if len(bot_peaks) > 0 else int(height * 0.80)

        # ── Gri bölge tespiti ile ince ayar ──
        overall_mean = np.mean(smoothed)
        gray_tol = np.std(smoothed) * 0.8
        gray_rows = np.where(
            (smoothed > overall_mean - gray_tol) &
            (smoothed < overall_mean + gray_tol * 0.5)
        )[0]
        if len(gray_rows) > 50:
            roi_top = max(roi_top, gray_rows[0] - 30)
            roi_bottom = min(roi_bottom, gray_rows[-1] + 30)

        # ── Sınır kontrolleri ──
        roi_top = max(int(height * 0.30), min(roi_top, int(height * 0.45)))
        roi_bottom = max(int(height * 0.65), min(roi_bottom, int(height * 0.85)))

        min_h = int(height * 0.30)
        if roi_bottom - roi_top < min_h:
            centre = (roi_top + roi_bottom) // 2
            roi_top = centre - min_h // 2
            roi_bottom = centre + min_h // 2

        roi_top = max(10, roi_top)
        roi_bottom = min(height - 10, roi_bottom)

        return roi_top, roi_bottom

    # ══════════════════════════════════════════════════════════════════════
    # Ön İşleme (Preprocessing)
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _apply_clahe(img: np.ndarray, cfg: Dict) -> np.ndarray:
        """CLAHE — yerel kontrast artırma."""
        tile = int(cfg.get("clahe_tile_size", 8))
        clahe = cv2.createCLAHE(
            clipLimit=float(cfg.get("clahe_clip_limit", 2.5)),
            tileGridSize=(tile, tile),
        )
        return clahe.apply(img)

    @staticmethod
    def _correct_background(img: np.ndarray, cfg: Dict) -> np.ndarray:
        """Gaussian background correction."""
        sigma = float(cfg.get("background_sigma", 35))
        bg = gaussian_filter(img.astype(float), sigma=sigma)
        corrected = img.astype(float) - bg + 128.0
        return np.clip(corrected, 0, 255).astype(np.uint8)

    @staticmethod
    def _enhance_dark_spots_multiscale(img: np.ndarray) -> np.ndarray:
        """
        Multi-scale black top-hat.
        Orijinal: (2,2), (3,3), (4,4), (5,5), (7,7) — 5 kernel.
        """
        results = []
        for k in [(2, 2), (3, 3), (4, 4), (5, 5), (7, 7)]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k)
            results.append(cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel))
        return np.maximum.reduce(results)

    # ══════════════════════════════════════════════════════════════════════
    # 6-Method Weighted Vote — orijinal ağırlıklar ve logic
    # ══════════════════════════════════════════════════════════════════════

    def _multi_method_detect(
        self,
        roi: np.ndarray,
        corrected: np.ndarray,
        enhanced: np.ndarray,
        clahe_img: np.ndarray,
        cfg: Dict,
    ) -> np.ndarray:
        """
        6 farklı yöntemle tespit → ağırlıklı oylama → binary maske.

        Ağırlıklar (orijinal):
            darkness=1.0, tophat=1.2, log=1.5,
            adaptive=1.0, statistical=1.3, clahe=1.0
        Toplam = 7.0
        """
        weights = [1.0, 1.2, 1.5, 1.0, 1.3, 1.0]
        total_weight = sum(weights)  # 7.0

        votes = np.zeros_like(roi, dtype=np.float32)

        # ── Method 1: Global darkness threshold (w=1.0) ──
        dark_thresh = min(
            np.percentile(corrected, float(cfg.get("darkness_percentile", 25))),
            float(cfg.get("absolute_darkness", 100)),
        )
        _, m1 = cv2.threshold(corrected, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        votes += (m1.astype(np.float32) / 255.0) * weights[0]

        # ── Method 2: Multi-scale top-hat threshold (w=1.2) ──
        enh_thresh = np.mean(enhanced) + 2.0 * np.std(enhanced)
        _, m2 = cv2.threshold(enhanced, enh_thresh, 255, cv2.THRESH_BINARY)
        votes += (m2.astype(np.float32) / 255.0) * weights[1]

        # ── Method 3: LoG blob detection (w=1.5) ──
        try:
            inverted = (255 - clahe_img).astype(float) / 255.0
            blobs = blob_log(
                inverted,
                min_sigma=float(cfg.get("min_blob_sigma", 0.5)),
                max_sigma=float(cfg.get("max_blob_sigma", 3.0)),
                num_sigma=int(cfg.get("log_num_sigma", 10)),
                threshold=float(cfg.get("log_threshold", 0.015)),
            )
            m3 = np.zeros_like(roi, dtype=np.float32)
            for y, x, sigma in blobs:
                r = max(1, int(sigma * np.sqrt(2)))
                cv2.circle(m3, (int(x), int(y)), r, 1.0, -1)
            votes += m3 * weights[2]
        except Exception as e:
            print(f"  LoG detection warning: {e}")

        # ── Method 4: Adaptive local threshold (w=1.0) ──
        block_size = int(cfg.get("local_threshold_block_size", 11))
        if block_size % 2 == 0:
            block_size += 1
        offset = float(cfg.get("local_threshold_offset", 4))
        local_thresh = threshold_local(corrected, block_size=block_size, offset=offset)
        m4 = (corrected < local_thresh).astype(np.float32)
        votes += m4 * weights[3]

        # ── Method 5: Statistical threshold — mean − k·σ (w=1.3) ──
        ks = block_size  # Aynı kernel boyutu
        k_factor = float(cfg.get("local_k_factor", 1.8))
        fimg = corrected.astype(float)
        local_mean = cv2.blur(fimg, (ks, ks))
        local_sq_mean = cv2.blur(fimg ** 2, (ks, ks))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
        stat_thresh = local_mean - k_factor * local_std
        m5 = ((fimg < stat_thresh) & (local_std > 2)).astype(np.float32)
        votes += m5 * weights[4]

        # ── Method 6: CLAHE + darkness threshold (w=1.0) ──
        clahe_thresh = np.percentile(
            clahe_img,
            max(1, float(cfg.get("darkness_percentile", 25)) - 5),
        )
        _, m6 = cv2.threshold(clahe_img, clahe_thresh, 255, cv2.THRESH_BINARY_INV)
        votes += (m6.astype(np.float32) / 255.0) * weights[5]

        # ── Karar: vote_threshold'un üzerindekiler bakteri ──
        consensus = votes / total_weight
        vote_thr = float(cfg.get("vote_threshold", 0.40))
        return (consensus >= vote_thr).astype(np.uint8) * 255

    # ══════════════════════════════════════════════════════════════════════
    # Post-processing
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _morphological_cleanup(binary: np.ndarray, cfg: Dict) -> np.ndarray:
        """
        Morfolojik temizlik: Open → Close.
        Orijinal script iki adım da yapıyor; eski entegrasyonda close eksikti.
        """
        osz = int(cfg.get("morph_open_size", 1))
        csz = int(cfg.get("morph_close_size", 2))

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (osz, osz))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (csz, csz))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close, iterations=1)

        return closed

    @staticmethod
    def _filter_components(binary: np.ndarray, cfg: Dict) -> np.ndarray:
        """
        Connected component filtreleme — boyut + aspect ratio.
        Orijinal script'te aspect ratio kontrolü var, eski entegrasyonda yoktu.
        """
        min_sz = int(cfg.get("min_bacteria_size", 2))
        max_sz = int(cfg.get("max_bacteria_size", 150))

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        output = np.zeros_like(binary, dtype=np.uint8)

        count = 0
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Boyut filtresi
            if not (min_sz <= area <= max_sz):
                continue

            # Aspect ratio filtresi (orijinal script'ten)
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(w, h) / (min(w, h) + 1)
            limit = 20 if area < 10 else 15
            if aspect > limit:
                continue

            output[labels == i] = 255
            count += 1

        print(f"DEBUG: Component filter kept {count} bacteria (of {n_labels - 1} candidates).")
        return output