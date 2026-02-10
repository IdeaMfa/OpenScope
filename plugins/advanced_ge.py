import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import morphology, filters
from skimage.filters import threshold_local
from skimage.feature import blob_log
from core.plugin_interface import AnalysisPlugin
from typing import Dict, Any, Optional, Tuple, List

class AdvancedBacteriaDetector(AnalysisPlugin):
    """
    Advanced Brightfield Bacteria Detector v4.5 (Ported from Okan's Script).
    Uses Multi-Method Voting (Consensus) for robust detection.
    """

    @property
    def name(self) -> str:
        return "Advanced Bacteria Detector (v4.5)"

    @property
    def parameters_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        GUI için parametre tanımları.
        """
        return {
            "sensitivity": {"type": "list", "options": ["conservative", "balanced", "sensitive", "ultra_sensitive"], "default": "balanced"},
            "min_size": {"type": int, "min": 1, "max": 500, "default": 2},
            "max_size": {"type": int, "min": 10, "max": 5000, "default": 150}
        }
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {"sensitivity": "balanced"}

    def _get_internal_params(self, sensitivity: str) -> Dict:
        """Sensitivite moduna göre algoritma ayarlarını döndürür."""
        presets = {
            'conservative': {
                'background_sigma': 30, 'clahe_clip': 2.0, 'darkness_p': 20,
                'local_k': 2.0, 'log_thresh': 0.02, 'vote_ratio': 0.45
            },
            'balanced': {
                'background_sigma': 35, 'clahe_clip': 2.5, 'darkness_p': 25,
                'local_k': 1.8, 'log_thresh': 0.015, 'vote_ratio': 0.40
            },
            'sensitive': {
                'background_sigma': 40, 'clahe_clip': 3.0, 'darkness_p': 30,
                'local_k': 1.5, 'log_thresh': 0.01, 'vote_ratio': 0.35
            },
            'ultra_sensitive': {
                'background_sigma': 50, 'clahe_clip': 4.0, 'darkness_p': 35,
                'local_k': 1.2, 'log_thresh': 0.008, 'vote_ratio': 0.30
            }
        }
        return presets.get(sensitivity, presets['balanced'])

    def run(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        print(f"DEBUG: Running {self.name}...")
        
        # 1. Görüntü Hazırlığı (Grayscale & Float)
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Parametreleri al
        sens_mode = params.get("sensitivity", "balanced")
        cfg = self._get_internal_params(sens_mode)

        # --- AŞAMA 1: Ön İşleme ---
        # CLAHE (Kontrast Artırma)
        clahe = cv2.createCLAHE(clipLimit=cfg['clahe_clip'], tileGridSize=(8, 8))
        img_clahe = clahe.apply(gray)

        # Background Correction (Arka planı düzeltme)
        bg = gaussian_filter(gray.astype(float), sigma=cfg['background_sigma'])
        img_corrected = gray.astype(float) - bg + 128
        img_corrected = np.clip(img_corrected, 0, 255).astype(np.uint8)

        # Multi-scale TopHat (Karanlık noktaları belirginleştirme)
        kernels = [(3,3), (5,5), (7,7)]
        tophats = []
        for k in kernels:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k)
            tophats.append(cv2.morphologyEx(img_corrected, cv2.MORPH_BLACKHAT, kernel))
        img_enhanced = np.maximum.reduce(tophats)

        # --- AŞAMA 2: Oylama Yöntemleri (6 Algoritma) ---
        votes = np.zeros_like(gray, dtype=np.float32)
        
        # 1. Darkness (Basit Koyuluk)
        thresh_dark = np.percentile(img_corrected, cfg['darkness_p'])
        votes += (img_corrected < thresh_dark).astype(float) * 1.0

        # 2. TopHat Threshold
        thresh_th = np.mean(img_enhanced) + 2.0 * np.std(img_enhanced)
        votes += (img_enhanced > thresh_th).astype(float) * 1.2

        # 3. Adaptive Threshold (Yerel)
        local_thresh = threshold_local(img_corrected, block_size=15, offset=5)
        votes += (img_corrected < local_thresh).astype(float) * 1.0

        # 4. Statistical Threshold (Mean - k*Std)
        local_mean = cv2.blur(img_corrected.astype(float), (11, 11))
        local_sq_mean = cv2.blur(img_corrected.astype(float)**2, (11, 11))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        stat_thresh = local_mean - (cfg['local_k'] * local_std)
        votes += (img_corrected < stat_thresh).astype(float) * 1.3

        # 5. CLAHE Darkness
        thresh_clahe = np.percentile(img_clahe, cfg['darkness_p'] - 5)
        votes += (img_clahe < thresh_clahe).astype(float) * 1.0

        # 6. LoG (Laplacian of Gaussian) - En yavaşı ama en iyisi
        # Küçük resimlerde veya ROI'lerde çalıştırıyoruz
        try:
            img_norm = (255 - img_clahe) / 255.0 # Invert & Normalize
            blobs = blob_log(img_norm, min_sigma=0.5, max_sigma=3.0, num_sigma=5, threshold=cfg['log_thresh'])
            mask_log = np.zeros_like(gray, dtype=float)
            for blob in blobs:
                y, x, sigma = blob
                r = int(sigma * 1.414)
                if r < 1: r = 1
                cv2.circle(mask_log, (int(x), int(y)), r, 1.0, -1)
            votes += mask_log * 1.5
        except Exception as e:
            print(f"LoG Error: {e}")

        # --- AŞAMA 3: Karar Verme ---
        total_weight = 1.0 + 1.2 + 1.0 + 1.3 + 1.0 + 1.5
        consensus_map = votes / total_weight
        
        # Oyların %40'ından fazlasını alanlar seçilir
        final_binary = (consensus_map >= cfg['vote_ratio']).astype(np.uint8) * 255

        # --- AŞAMA 4: Temizlik ve Filtreleme ---
        # Küçük delikleri kapat
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        cleaned = cv2.morphologyEx(final_binary, cv2.MORPH_OPEN, kernel_clean)
        
        # Boyut Filtresi
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        
        output_mask = np.zeros_like(gray, dtype=bool)
        
        min_sz = params.get("min_size", 2)
        max_sz = params.get("max_size", 150)

        count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_sz <= area <= max_sz:
                output_mask[labels == i] = True
                count += 1
        
        print(f"DEBUG: Advanced detection found {count} bacteria.")
        return output_mask