import os
import numpy as np
import soundfile as sf
import pandas as pd
import time

from load_signal import load_signal_from_wav
from tdoa import estimate_tdoa_cc, estimate_tdoa_gcc
from doa import estimate_doa_from_tdoa

# --- Configuración ---
RIR_DATASET_DIR = "rir_dataset_user_defined"
METADATA_FILENAME = os.path.join(RIR_DATASET_DIR, "simulation_metadata.csv")
ANECHOIC_SIGNAL_PATH = "p336_007.wav"
SNRS_TO_TEST_DB = [90]
C_SOUND = 343.0

def calculate_real_tdoa(source_pos, mic_a_pos, mic_b_pos, c=C_SOUND):
    """Calcula TDOA real basado en geometría."""
    dist_a = np.linalg.norm(np.array(source_pos) - np.array(mic_a_pos))
    dist_b = np.linalg.norm(np.array(source_pos) - np.array(mic_b_pos))
    return (dist_a - dist_b) / c

def add_noise_for_snr(signal, target_snr_db):
    """Añade ruido AWGN a una señal para un SNR objetivo."""
    signal_power = np.mean(signal**2)
    if signal_power == 0:
        return signal
    snr_linear = 10**(target_snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, 1, len(signal))
    noise = noise * np.sqrt(noise_power / (np.mean(noise**2) + 1e-12))
    return signal + noise

def process_simulation_data():
    print("--- main.py: Iniciando procesamiento de datos de simulación ---")
    if not os.path.exists(METADATA_FILENAME):
        print(f"Error: Archivo de metadatos no encontrado: {METADATA_FILENAME}")
        return

    metadata_df = pd.read_csv(METADATA_FILENAME, engine='python')
    print(f"Metadatos cargados: {len(metadata_df)} configuraciones encontradas en CSV.")

    anechoic_signal, fs_anechoic = load_signal_from_wav(ANECHOIC_SIGNAL_PATH, target_fs=48000)
    if anechoic_signal is None:
        print(f"Error: No se pudo cargar la señal anecoica de {ANECHOIC_SIGNAL_PATH}")
        return
    print(f"Señal anecoica cargada: {ANECHOIC_SIGNAL_PATH} (Fs: {fs_anechoic} Hz)")

    all_experiment_results = []
    tdoa_methods = ['cc', 'phat', 'scot', 'ml', 'roth']  # Asegúrate de que estén implementados en tdoa.py
    print(f"Métodos TDOA a probar: {tdoa_methods}")

    for idx, sim_params in metadata_df.iterrows():
        print(f"\nProcesando Config ID: {sim_params['config_id']} ({idx+1}/{len(metadata_df)})")
        fs_sim = sim_params['fs_hz']
        if fs_sim != fs_anechoic:
            print(f"  Advertencia: Fs de simulación ({fs_sim}) no coincide con Fs anecoica ({fs_anechoic}). Saltando config.")
            continue

        num_mics = int(sim_params['num_mics_processed'])
        mic_rirs, mic_positions = [], []
        valid = True

        for i in range(num_mics):
            rir_path = os.path.join(RIR_DATASET_DIR, f"{sim_params['rir_file_basename']}_micidx_{i}.wav")
            if not os.path.exists(rir_path):
                print(f"  Error: RIR no encontrada: {rir_path}. Saltando config.")
                valid = False
                break
            try:
                rir_data, _ = sf.read(rir_path)
                mic_rirs.append(rir_data)
                pos = [sim_params.get(f'mic{i}_pos_x', np.nan),
                       sim_params.get(f'mic{i}_pos_y', np.nan),
                       sim_params.get(f'mic{i}_pos_z', np.nan)]
                if any(pd.isna(pos)):
                    print(f"  Advertencia: Posición incompleta para micrófono {i}. Saltando config.")
                    valid = False
                    break
                mic_positions.append(pos)
            except Exception as e:
                print(f"  Error cargando RIR {rir_path}: {e}. Saltando config.")
                valid = False
                break

        if not valid or len(mic_rirs) != num_mics:
            continue

        reverberant_signals = [np.convolve(anechoic_signal, rir, mode='full') for rir in mic_rirs]
        source_pos = [sim_params['source_pos_x'], sim_params['source_pos_y'], sim_params['source_pos_z']]
        real_doa_deg = sim_params.get('actual_azimuth_src_to_array_center_deg', np.nan)
        mic_sep = sim_params['mic_separation_m']

        for snr_db in SNRS_TO_TEST_DB:
            noisy_signals = [add_noise_for_snr(sig, snr_db) for sig in reverberant_signals]
            mic_pairs = []
            for i in range(num_mics):
                for j in range(i + 1, num_mics):
                    d_pair = abs(j - i) * mic_sep
                    real_tdoa = calculate_real_tdoa(source_pos, mic_positions[i], mic_positions[j])
                    mic_pairs.append({'mic1': i, 'mic2': j, 'd': d_pair, 'real_tdoa': real_tdoa})

            for pair in mic_pairs:
                idx1, idx2, d_pair, real_tdoa = pair['mic1'], pair['mic2'], pair['d'], pair['real_tdoa']
                sig_a, sig_b = noisy_signals[idx1], noisy_signals[idx2]
                result_base = sim_params.to_dict()
                result_base.update({
                    'snr_db': snr_db,
                    'mic_pair': f"{idx1}-{idx2}",
                    'mic_pair_distance_m': d_pair,
                    'tdoa_real_s': real_tdoa,
                    'doa_real_deg': real_doa_deg
                })
                for method in tdoa_methods:
                    try:
                        if method == 'cc':
                            tdoa_val, comp_time = estimate_tdoa_cc(sig_a, sig_b, fs_sim)
                        else:
                            tdoa_val, comp_time = estimate_tdoa_gcc(sig_a, sig_b, fs_sim, method=method)
                        tdoa_error = tdoa_val - real_tdoa if not np.isnan(tdoa_val) else np.nan
                        doa_est = estimate_doa_from_tdoa(tdoa_val, d_pair)

                    except Exception as e:
                        print(f"  Error en método {method} para par {idx1}-{idx2}: {e}")
                        tdoa_val, comp_time, tdoa_error, doa_est = np.nan, np.nan, np.nan, np.nan
                    result = result_base.copy()
                    result.update({
                        'tdoa_method': method,
                        'tdoa_estimated_s': tdoa_val,
                        'tdoa_error_s': tdoa_error,
                        'tdoa_computation_time_s': comp_time,
                        'doa_estimated_from_pair_deg': doa_est
                    })
                    all_experiment_results.append(result)

    if all_experiment_results:
        results_df = pd.DataFrame(all_experiment_results)
        # --- RESUMEN POR POSICIÓN Y MÉTODO ---
        pair_rows = results_df[results_df['mic_pair'].str.contains('-') & results_df['tdoa_method'].notna()]
        summary = pair_rows.groupby([
            'azimuth_deg', 'elevation_deg', 'snr_db', 'tdoa_method'
        ]).agg(
            doa_estimated_mean=('doa_estimated_from_pair_deg', 'mean'),
            doa_estimated_std=('doa_estimated_from_pair_deg', 'std'),
            doa_real_deg=('doa_real_deg', 'first'),
        ).reset_index()
        summary['doa_error_mean'] = summary['doa_estimated_mean'] - summary['doa_real_deg']
        summary_csv_path = "doa_summary_by_position.csv"
        summary.to_csv(summary_csv_path, index=False)
        print(f"Resumen por posición guardado en: {summary_csv_path}")
    else:
        print("No se generaron resultados.")
    print("--- main.py: Procesamiento finalizado ---")

if __name__ == "__main__":
    if not os.path.exists(ANECHOIC_SIGNAL_PATH):
        raise ValueError(f"Archivo anecoico {ANECHOIC_SIGNAL_PATH} no encontrado. Por favor, asegúrate de que el archivo existe en el directorio actual.")
    if not os.path.exists(METADATA_FILENAME):
        raise ValueError(f"Archivo de metadatos {METADATA_FILENAME} no encontrado. Por favor, asegúrate de que el archivo existe en el directorio {RIR_DATASET_DIR}.")
    process_simulation_data()