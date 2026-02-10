import cv2
import numpy as np
from typing import Dict, Any, Optional
from core.plugin_interface import AnalysisPlugin

class BasicBlobDetector(AnalysisPlugin):
    """
    High Sensitivity Blob Detector.
    Adjusted to detect fainter and smaller objects.
    """

    @property
    def name(self) -> str:
        return "Basic Blob Detection (Sensitive)"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "blur_kernel": 3,     # Daha az bulanıklık (Eski: 5)
            "block_size": 25,     # Daha geniş alana bak (Eski: 11)
            "c_value": 0,         # HASSASİYET: 0 = En ufak koyuluğu bile al (Eski: 2)
            "min_size": 5,        # Çok küçük noktaları da al (Eski: 10)
            "max_size": 2000      # Büyük kümeleri de al
        }

    def run(self, image: np.ndarray, params: Dict[str, Any]) -> Optional[np.ndarray]:
        print(f"DEBUG: Running {self.name} with params: {params}")

        # 1. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 2. Hafif Blur (Gürültüyü çok silmeden)
        k = params.get("blur_kernel", 3)
        if k % 2 == 0: k += 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # 3. Adaptive Thresholding
        block_size = params.get("block_size", 25)
        if block_size % 2 == 0: block_size += 1
        
        c_val = params.get("c_value", 0)

        binary_mask = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            block_size, 
            c_val
        )

        # 4. Morfolojik İşlem (Opsiyonel - Çok gürültü çıkarsa burayı açın)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # 5. Boyut Filtreleme
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        min_size = params.get("min_size", 5)
        max_size = params.get("max_size", 2000)
        
        final_mask = np.zeros_like(gray, dtype=bool)
        count = 0

        # Arka plan (0) hariç dön
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_size <= area <= max_size:
                final_mask[labels == i] = True
                count += 1

        print(f"DEBUG: Found {count} objects.")
        return final_mask