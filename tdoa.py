import numpy as np
from scipy.signal import correlate, correlation_lags
from numpy.fft import fft, ifft, fftshift
import time

def return_nan_with_time(start_time):
    """Devuelve (NaN, tiempo transcurrido) para casos con error."""
    return np.nan, time.perf_counter() - start_time

def estimate_tdoa_cc(sig1: np.ndarray, sig2: np.ndarray, fs: float) -> tuple[float, float]:
    """
    Estima el TDOA usando correlación cruzada clásica.

    Parámetros:
    - sig1, sig2: Señales de entrada (1D)
    - fs: Frecuencia de muestreo en Hz

    Retorna:
    - tdoa: Estimación del retardo en segundos
    - duration: Tiempo que tomó la estimación (en segundos)
    """
    start_time = time.perf_counter()
    sig1 = np.asarray(sig1).flatten()
    sig2 = np.asarray(sig2).flatten()

    if len(sig1) == 0 or len(sig2) == 0:
        return return_nan_with_time(start_time)

    try:
        corr = correlate(sig1, sig2, mode='full')
        lags_samples = correlation_lags(len(sig1), len(sig2), mode='full')
    except Exception:
        return return_nan_with_time(start_time)

    if len(lags_samples) == 0 or len(corr) != len(lags_samples):
        return return_nan_with_time(start_time)

    lags_seconds = lags_samples / fs
    tdoa_idx = np.argmax(corr)
    tdoa = lags_seconds[tdoa_idx]
    return tdoa, time.perf_counter() - start_time

def estimate_tdoa_gcc(sig1: np.ndarray, sig2: np.ndarray, fs: float, method: str = 'phat') -> tuple[float, float]:
    """
    Estima el TDOA usando GCC (Generalized Cross-Correlation).

    Parámetros:
    - sig1, sig2: Señales de entrada (1D)
    - fs: Frecuencia de muestreo
    - method: Método de ponderación ('phat', 'scot', 'ml', 'roth', 'classic')

    Retorna:
    - tdoa: Retardo estimado en segundos
    - duration: Tiempo que tomó la estimación
    """
    start_time = time.perf_counter()
    sig1 = np.asarray(sig1).flatten()
    sig2 = np.asarray(sig2).flatten()
    len_sig1, len_sig2 = len(sig1), len(sig2)

    if len_sig1 == 0 or len_sig2 == 0:
        return return_nan_with_time(start_time)

    n = len_sig1 + len_sig2 - 1
    if n <= 0:
        return return_nan_with_time(start_time)

    try:
        SIG1 = fft(sig1, n=n)
        SIG2 = fft(sig2, n=n)
    except Exception:
        return return_nan_with_time(start_time)

    R = SIG1 * np.conj(SIG2)
    method = method.lower()

    if method == 'phat':
        R_abs = np.abs(R)
        R_weighted = R / (R_abs + 1e-10) if np.any(R_abs >= 1e-12) else R

    elif method == 'scot':
        G11 = np.abs(SIG1)**2
        G22 = np.abs(SIG2)**2
        den = np.sqrt(G11 * G22 + 1e-10)
        R_weighted = R / (den + 1e-10) if np.any(den >= 1e-12) else R

    elif method == 'ml':
        G11 = np.abs(SIG1)**2
        G22 = np.abs(SIG2)**2
        abs_R_sq = np.abs(R)**2
        denominator_coherence = G11 * G22
        coherence_sq = np.zeros_like(abs_R_sq)
        valid_idx = denominator_coherence > 1e-12
        with np.errstate(divide='ignore', invalid='ignore'):
            coherence_sq[valid_idx] = abs_R_sq[valid_idx] / denominator_coherence[valid_idx]
        coherence_sq = np.clip(coherence_sq, 0.0, 1.0 - 1e-7)
        Psi = coherence_sq / (1.0 - coherence_sq + 1e-10)
        R_weighted = R * Psi

    elif method == 'roth':
        R_abs_sq = np.abs(SIG2)**2
        R_weighted = R / (R_abs_sq + 1e-10) if np.any(R_abs_sq >= 1e-12) else R

    elif method == 'classic':
        return estimate_tdoa_cc(sig1, sig2, fs)

    else:
        raise ValueError("Método GCC no reconocido. Use 'phat', 'scot', 'ml', 'classic' o 'roth'.")

    try:
        cc = np.real(ifft(R_weighted))
    except Exception:
        return return_nan_with_time(start_time)

    if len(cc) == 0:
        return return_nan_with_time(start_time)

    lags_vector = correlation_lags(len_sig1, len_sig2, mode='full') / fs
    # El máximo de cc corresponde al lag de lags_vector
    tdoa_index = np.argmax(cc)
    tdoa = lags_vector[tdoa_index]
    return tdoa, time.perf_counter() - start_time
