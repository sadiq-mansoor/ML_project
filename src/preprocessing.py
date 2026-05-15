from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pywt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_STATE = 42


# MUST be module-level — locally-defined classes cannot be pickled for n_jobs=-1
class SafeSelectKBest(SelectKBest):
    """Caps k to n_features at fit time. Defined at module level for pickling."""
    def fit(self, X, y=None):
        self.k = min(self.k, X.shape[1])
        return super().fit(X, y)


class WaveletDenoiser(BaseEstimator, TransformerMixin):
    """Row-wise wavelet denoising using db4 wavelet, level 4."""
    def __init__(self, wavelet='db4', level=4):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = np.empty_like(X)
        for i, row in enumerate(X):
            out[i] = self._denoise_row(row)
        return out

    def _denoise_row(self, signal):
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        N = len(coeffs[-1])  # use finest detail length, NOT len(signal)
        threshold = sigma * np.sqrt(2 * np.log(N))
        return pywt.waverec(
            [pywt.threshold(c, threshold, mode='soft') for c in coeffs],
            self.wavelet
        )[:len(signal)]


def build_pipeline_a(k=20):
    """Pipeline A: StandardScaler → MI Feature Selection (top-k).
    Uses module-level SafeSelectKBest so it can be pickled by n_jobs=-1.
    """
    score_fn = partial(mutual_info_classif, random_state=RANDOM_STATE)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SafeSelectKBest(score_fn, k=k)),
    ])


def build_pipeline_b(pca_variance=0.95):
    """Pipeline B: StandardScaler → PCA (95% variance)."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_variance)),
    ])


def build_pipeline_c(pca_variance=0.95):
    """Pipeline C: Wavelet Denoising → StandardScaler → PCA. Bonn EEG raw signals only."""
    return Pipeline([
        ('wavelet', WaveletDenoiser()),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_variance)),
    ])


def get_pipelines(dataset_name: str) -> dict:
    """Returns applicable pipelines for the given dataset."""
    p = {'A': build_pipeline_a(), 'B': build_pipeline_b()}
    if dataset_name == 'bonn_eeg':
        p['C'] = build_pipeline_c()
    return p
