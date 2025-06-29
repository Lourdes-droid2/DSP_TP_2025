import numpy as np
from scipy.signal import correlation_lags

C_SOUND = 343.0  # Velocidad del sonido en m/s

def estimate_doa(signal_input, mic_d, fs, method="classic"):
    """
    Estima el ángulo de arribo (DOA) y el TDOA promedio desde un array lineal de micrófonos omni.

    Parámetros:
    - signal_input: lista de rutas a archivos `.wav` o array (N, L)
    - mic_d: separación entre micrófonos (m)
    - fs: frecuencia de muestreo (Hz)
    - method: método de correlación ('classic', 'gcc_roth', 'gcc_phat')

    Retorna:
    - avg_angle_deg: ángulo de arribo promedio (°)
    - avg_tdoa: TDOA promedio (s)
    - hemi_avgs: promedio de ángulos por hemisferio
    - tdoas: lista de TDOAs individuales (s)
    - angles: lista de ángulos individuales (°)
    """
    # Cargar señales
    signals, fs = load_signals(signal_input, fs)  # Esta función debe estar definida en tu entorno
    n_mics = len(signals)
    ref_idx = n_mics // 2
    ref_signal = signals[ref_idx]

    # Selección de método de correlación
    methods = {
        'classic': cross_correlation_fft,
        'gcc_phat': gcc_phat,
        'gcc_roth': gcc_roth
    }
    if method not in methods:
        raise ValueError("Método de correlación no válido. Use 'classic', 'gcc_phat' o 'gcc_roth'.")
    corr_func = methods[method]

    tdoas = []
    angles = []

    for i, sig in enumerate(signals):
        if i == ref_idx:
            continue  # No estimar TDOA/DOA consigo mismo

        # Correlación cruzada
        corr = corr_func(sig, ref_signal)
        lags = correlation_lags(len(sig), len(ref_signal))
        lag = lags[np.argmax(corr)]
        tdoa = lag / fs
        tdoas.append(tdoa)

        # Distancia al micrófono de referencia
        d_mic_n_to_ref = mic_d * abs(i - ref_idx)
        if d_mic_n_to_ref == 0:
            continue  # Evitar división por cero

        # Estimación del ángulo
        cos_val = np.clip(tdoa * C_SOUND / d_mic_n_to_ref, -1.0, 1.0)
        angle_rad = np.arccos(cos_val)
        angle_deg = np.degrees(angle_rad)

        # Determinar hemisferio por signo del TDOA
        if tdoa < 0:
            angle_deg = (360 - angle_deg) % 360

        angles.append(angle_deg)

    # Clasificar en hemisferios
    hemispheres = {
        "H1": [a for a in angles if (0 <= a <= 90) or (270 <= a < 360)],
        "H2": [a for a in angles if 90 < a < 270],
    }
    hemi_avgs = {h: np.mean(a) for h, a in hemispheres.items() if a}

    # Elegir hemisferio dominante
    dominant_hemi, dominant_angles = max(hemispheres.items(), key=lambda x: len(x[1]))
    if not dominant_angles:
        raise RuntimeError("No se pudo determinar un hemisferio dominante.")

    avg_angle_deg = np.mean(dominant_angles)
    avg_tdoa = np.mean(tdoas)

    return avg_angle_deg, avg_tdoa, hemi_avgs, tdoas, angles

import numpy as np

C_SOUND = 343.0  # Velocidad del sonido en m/s

def estimate_doa_from_tdoa(tdoa, d, c=C_SOUND):
    """
    Estima el ángulo de llegada (DOA) en grados a partir del TDOA y la distancia entre micrófonos.
    El ángulo es respecto al eje del par de micrófonos (0° = endfire, 90° = broadside).
    """
    if np.isnan(tdoa) or d <= 0:
        return np.nan
    val = (c * tdoa) / d
    val_clipped = np.clip(val, -1.0, 1.0)
    theta_rad = np.arccos(val_clipped)
    return np.degrees(theta_rad)