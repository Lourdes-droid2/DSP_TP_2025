import os
import numpy as np
import soundfile as sf
import pandas as pd
import time
from scipy.signal import fftconvolve

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

        reverberant_signals = [fftconvolve(anechoic_signal, rir, mode='full') for rir in mic_rirs]
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

            # Inicializar el diccionario para guardar resultados por método
            pair_results_by_method = {}

            for pair in mic_pairs:
                idx1, idx2, d_pair, real_tdoa = pair['mic1'], pair['mic2'], pair['d'], pair['real_tdoa']
                sig_a, sig_b = noisy_signals[idx1], noisy_signals[idx2]
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
                    # Guardar resultados en lista temporal por método
                    if method not in pair_results_by_method:
                        pair_results_by_method[method] = []
                    pair_results_by_method[method].append({
                        'doa_estimated_from_pair_deg': doa_est,
                        'mic_pair': f"{idx1}-{idx2}",
                        'tdoa_val': tdoa_val,
                        'd_pair': d_pair,
                        'real_tdoa': real_tdoa,
                        'doa_real_deg': real_doa_deg,
                    })

            # --- DEBUG: Mostrar qué pares adyacentes hay para cada método ---
            for method in tdoa_methods:
                adj_pairs = [r for r in pair_results_by_method.get(method, []) if abs(int(r['mic_pair'].split('-')[0]) - int(r['mic_pair'].split('-')[1])) == 1]
                print(f"\nMétodo: {method} | Pares adyacentes encontrados: {len(adj_pairs)}")
                for r in adj_pairs:
                    print(f"  Par: {r['mic_pair']} | DOA: {r['doa_estimated_from_pair_deg']} | TDOA: {r['tdoa_val']} | d: {r['d_pair']} | real_tdoa: {r['real_tdoa']}")

                if not adj_pairs:
                    print(f"  [WARN] No hay pares adyacentes para método {method} en config {sim_params['config_id']} SNR {snr_db}")
                    continue

                # --- DEBUG: Mostrar los DOA antes de promediar ---
                doa_vals = [r['doa_estimated_from_pair_deg'] for r in adj_pairs]
                print(f"  DOAs a promediar para método {method}: {doa_vals}")

                # Si todos los valores son nan, el promedio será nan
                if all(np.isnan(doa_vals)):
                    print(f"  [WARN] Todos los DOA son NaN para método {method} en config {sim_params['config_id']} SNR {snr_db}")
                    doa_array_est = np.nan
                else:
                    doa_array_est = np.nanmean(doa_vals)
                doa_array_real = real_doa_deg
                doa_array_error = doa_array_est - doa_array_real if not np.isnan(doa_array_est) and not np.isnan(doa_array_real) else np.nan

                print(f"  DOA promedio array para método {method}: {doa_array_est} (real: {doa_array_real}, error: {doa_array_error})")

                result = sim_params.to_dict()
                result.update({
                    'snr_db': snr_db,
                    'mic_pair': 'array_avg_adj_pairs',
                    'tdoa_method_for_avg_doa': method,
                    'doa_array_estimated_deg': doa_array_est,
                    'doa_array_real_deg': doa_array_real,
                    'doa_array_error_deg': doa_array_error
                })
                all_experiment_results.append(result)

            for method in tdoa_methods:
                for pair_result in pair_results_by_method.get(method, []):
                    pair_result_full = sim_params.to_dict()
                    pair_result_full.update(pair_result)
                    pair_result_full['snr_db'] = snr_db
                    pair_result_full['tdoa_method'] = method
                    all_experiment_results.append(pair_result_full)

    if all_experiment_results:
        results_df = pd.DataFrame(all_experiment_results)
        # Solo filas de promedio de array
        avg_results = results_df[results_df['mic_pair'] == 'array_avg_adj_pairs']
        # Guardar todos los campos relevantes
        avg_results.to_csv("doa_array_avg_results.csv", index=False)
        print("Resultados promediados por array guardados en: doa_array_avg_results.csv")
    else:
        print("No se generaron resultados.")
    print("--- main.py: Procesamiento finalizado ---")

if __name__ == "__main__":
    if not os.path.exists(ANECHOIC_SIGNAL_PATH):
        raise ValueError(f"Archivo anecoico {ANECHOIC_SIGNAL_PATH} no encontrado. Por favor, asegúrate de que el archivo existe en el directorio actual.")
    if not os.path.exists(METADATA_FILENAME):
        raise ValueError(f"Archivo de metadatos {METADATA_FILENAME} no encontrado. Por favor, asegúrate de que el archivo existe en el directorio {RIR_DATASET_DIR}.")
    process_simulation_data()