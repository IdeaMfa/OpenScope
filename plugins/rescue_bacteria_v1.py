"""
Rescue Bacteria Detector v1.0 - GUI Plugin
==========================================
Plugin interface for the Brightfield Bacteria Detection GUI.
Provides complementary detection to catch bacteria missed by v5.0.

Uses 6 different methods from v5.0:
  H-dome, Matched Filter, Morph Gradient Interior,
  Percentile Anomaly, DoG, H-minima

Install: Copy to plugins/ folder in GUI project.
"""

import numpy as np
from scipy.ndimage import (
    uniform_filter, median_filter, gaussian_filter,
    minimum_filter, maximum_filter, label as ndlabel,
    binary_dilation, percentile_filter,
    grey_dilation, grey_erosion, find_objects
)
from scipy.signal import fftconvolve
from skimage.morphology import disk, reconstruction
from skimage.morphology import h_minima as sk_h_minima

try:
    from core.plugin_interface import AnalysisPlugin
except ImportError:
    class AnalysisPlugin:
        pass


class RescueBacteriaV1Plugin(AnalysisPlugin):
    """Rescue detection plugin - complementary to Brightfield v5.0."""
    
    @property
    def name(self):
        return "Rescue Bacteria Detector v1.0"
    
    @property
    def parameters(self):
        return {
            'sensitivity': 'balanced',
            'roi_top': 0,
            'roi_bottom': 0,
            'hdome_h': 8,
            'matched_percentile': 3,
            'percentile_z_thresh': 0.8,
            'dog_percentile': 3,
            'hminima_h': 5,
            'vote_threshold': 0.40,
            'min_size': 3,
            'max_size': 150,
            'bg_sigma': 15,
        }
    
    @property
    def parameters_metadata(self):
        return {
            'sensitivity': {
                'type': 'list',
                'options': ['conservative', 'balanced', 'sensitive', 'ultra_sensitive'],
                'tooltip': 'Detection sensitivity preset'
            },
            'roi_top': {'type': 'int', 'min': 0, 'max': 2000, 'tooltip': 'ROI top (0=auto)'},
            'roi_bottom': {'type': 'int', 'min': 0, 'max': 2000, 'tooltip': 'ROI bottom (0=auto)'},
            'hdome_h': {'type': 'int', 'min': 2, 'max': 20, 'tooltip': 'H-dome depth'},
            'matched_percentile': {'type': 'float', 'min': 1, 'max': 10, 'tooltip': 'Matched filter percentile'},
            'percentile_z_thresh': {'type': 'float', 'min': 0.3, 'max': 2.0, 'tooltip': 'Percentile Z threshold'},
            'dog_percentile': {'type': 'float', 'min': 1, 'max': 10, 'tooltip': 'DoG percentile'},
            'hminima_h': {'type': 'int', 'min': 2, 'max': 15, 'tooltip': 'H-minima depth'},
            'vote_threshold': {'type': 'float', 'min': 0.1, 'max': 1.0, 'tooltip': 'Vote threshold'},
            'min_size': {'type': 'int', 'min': 1, 'max': 20, 'tooltip': 'Min component size'},
            'max_size': {'type': 'int', 'min': 50, 'max': 500, 'tooltip': 'Max component size'},
            'bg_sigma': {'type': 'float', 'min': 5, 'max': 30, 'tooltip': 'Background estimation sigma'},
        }
    
    def _load_preset(self, sensitivity, params):
        """Apply preset values if sensitivity changed."""
        presets = {
            'conservative': {'hdome_h':12,'matched_percentile':2,'percentile_z_thresh':1.0,
                           'dog_percentile':2,'hminima_h':7,'vote_threshold':0.50,'min_size':4,'max_size':150},
            'balanced':     {'hdome_h':8,'matched_percentile':3,'percentile_z_thresh':0.8,
                           'dog_percentile':3,'hminima_h':5,'vote_threshold':0.40,'min_size':3,'max_size':150},
            'sensitive':    {'hdome_h':6,'matched_percentile':4,'percentile_z_thresh':0.6,
                           'dog_percentile':4,'hminima_h':4,'vote_threshold':0.35,'min_size':2,'max_size':200},
            'ultra_sensitive':{'hdome_h':4,'matched_percentile':5,'percentile_z_thresh':0.5,
                             'dog_percentile':5,'hminima_h':3,'vote_threshold':0.30,'min_size':1,'max_size':250},
        }
        if sensitivity in presets:
            for k, v in presets[sensitivity].items():
                if k not in params or params[k] == self.parameters.get(k):
                    params[k] = v
        return params
    
    def run(self, image, params=None):
        """
        Run rescue detection.
        
        Args:
            image: numpy array (grayscale or RGB)
            params: dict of parameters
        
        Returns:
            Boolean mask (same size as input image)
        """
        if params is None:
            params = self.parameters.copy()
        
        p = self._load_preset(params.get('sensitivity', 'balanced'), params)
        
        # Grayscale
        if len(image.shape) == 3:
            gray = np.mean(image[:,:,:3], axis=2).astype(np.float64)
        else:
            gray = image.astype(np.float64)
        
        h, w = gray.shape
        
        # ROI
        roi_top = p.get('roi_top', 0)
        roi_bottom = p.get('roi_bottom', 0)
        if roi_top == 0 or roi_bottom == 0:
            roi_top, roi_bottom = int(h * 0.05), int(h * 0.85)
        
        roi = gray[roi_top:roi_bottom, :]
        roi_u8 = np.clip(roi, 0, 255).astype(np.uint8)
        bg_sigma = p.get('bg_sigma', 15)
        bg = gaussian_filter(roi, sigma=bg_sigma)
        bg_corr = bg - roi
        min_bg = 2
        
        # Weights
        weights = {'hdome':1.5, 'matched':1.3, 'morph':0.8,
                   'percentile':1.2, 'dog':1.3, 'hminima':1.0}
        tw = sum(weights.values())
        
        # M1: H-dome
        inv = 255.0 - roi
        seed = np.clip(inv - p['hdome_h'], 0, 255)
        recon = reconstruction(seed, inv, method='dilation')
        hdome = inv - recon
        m1 = (hdome > 2) & (bg_corr > min_bg)
        
        # M2: Matched filter
        mf = np.zeros_like(roi)
        for r in [1.0, 1.5, 2.0, 2.5, 3.0]:
            sz = int(4*r)+1
            y, x = np.mgrid[-sz:sz+1, -sz:sz+1]
            t = -np.exp(-(x**2+y**2)/(2*r**2))
            t -= t.mean()
            n = np.sqrt(np.sum(t**2))
            if n > 0: t /= n
            mf = np.minimum(mf, fftconvolve(roi, t, mode='same'))
        neg = mf[mf < 0]
        th2 = np.percentile(mf, p['matched_percentile']) if len(neg)>0 else -1
        m2 = (mf < th2) & (bg_corr > min_bg)
        
        # M3: Morph gradient interior
        mg = grey_dilation(roi_u8,size=3).astype(float) - grey_erosion(roi_u8,size=3).astype(float)
        interior = (mg < 5) & (bg_corr > 5)
        near_edge = binary_dilation(mg > np.percentile(mg, 80), iterations=2)
        m3 = interior & near_edge
        
        # M4: Percentile anomaly
        win = 15
        lp25 = percentile_filter(roi, percentile=25, size=win)
        lp75 = percentile_filter(roi, percentile=75, size=win)
        iqr = lp75 - lp25 + 1e-6
        z = (lp25 - roi) / iqr
        m4 = z > p['percentile_z_thresh']
        
        # M5: DoG
        dog = np.zeros_like(roi)
        for s1, s2 in [(0.5,1.0),(0.7,1.4),(1.0,2.0),(1.5,3.0),(2.0,4.0)]:
            dog = np.minimum(dog, gaussian_filter(roi,s1) - gaussian_filter(roi,s2))
        neg5 = dog[dog < 0]
        th5 = np.percentile(dog, p['dog_percentile']) if len(neg5)>0 else -1
        m5 = (dog < th5) & (bg_corr > min_bg)
        
        # M6: H-minima
        hmin = sk_h_minima(roi_u8, h=p['hminima_h'])
        m6 = (hmin < roi_u8) & (bg_corr > 3)
        
        # Vote
        vote = (m1.astype(float)*weights['hdome'] +
                m2.astype(float)*weights['matched'] +
                m3.astype(float)*weights['morph'] +
                m4.astype(float)*weights['percentile'] +
                m5.astype(float)*weights['dog'] +
                m6.astype(float)*weights['hminima']) / tw
        
        mask = vote >= p['vote_threshold']
        
        # Size filter
        labeled, n = ndlabel(mask)
        if n > 0:
            sizes = np.bincount(labeled.ravel())
            keep = (sizes >= p['min_size']) & (sizes <= p['max_size'])
            keep[0] = False
            mask = keep[labeled]
        
        # Aspect ratio filter
        labeled2, n2 = ndlabel(mask)
        if n2 > 0:
            slices = find_objects(labeled2)
            for i, s in enumerate(slices):
                if s is None: continue
                comp = labeled2[s] == (i+1)
                hc = s[0].stop - s[0].start
                wc = s[1].stop - s[1].start
                if comp.sum() > 5:
                    aspect = max(hc, wc) / (min(hc, wc) + 1e-6)
                    if aspect > 15:
                        mask[s][comp] = False
        
        # Build full mask
        full = np.zeros((h, w), dtype=bool)
        full[roi_top:roi_bottom, :] = mask
        
        return full
