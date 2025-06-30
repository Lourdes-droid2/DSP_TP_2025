import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import os
import random
import csv

def calculate_angle_and_distance(source_pos, array_center):
    """Calculates distance, azimuth, and elevation from source to array center."""
    delta = np.array(source_pos) - np.array(array_center)
    distance = np.linalg.norm(delta)
    # Azimuth in XY plane (angle from X-axis towards Y-axis)
    azimuth = np.degrees(np.arctan2(delta[1], delta[0]))
    # Elevation is the angle with respect to the XY plane
    # Positive elevation towards +Z
    xy_distance = np.linalg.norm(delta[:2]) # Magnitude in XY plane
    elevation = np.degrees(np.arctan2(delta[2], xy_distance))
    return distance, azimuth, elevation

def create_rir_example(base_output_filename_prefix, rt60_tgt, room_dim, source_pos, mic_positions, fs, is_anechoic=False):
    """
    Creates Room Impulse Responses (RIRs) and saves them to WAV files.
    Returns the count of successfully created RIR files and a list of actual microphone positions used.
    """
    files_created_count = 0
    actual_mic_positions_in_room = []
    try:
        margin = 0.01
        if not all(d > 2 * margin for d in room_dim):
            return files_created_count, actual_mic_positions_in_room
        if not all(margin <= source_pos[i] < room_dim[i] - margin for i in range(3)):
            return files_created_count, actual_mic_positions_in_room

        original_mic_indices_to_process = []
        mic_positions_for_pra = []
        for idx, mic_coord in enumerate(mic_positions):
            if not all(margin <= mic_coord[i] < room_dim[i] - margin for i in range(3)):
                continue
            if np.linalg.norm(np.array(source_pos) - np.array(mic_coord)) < 0.1: # Min 10cm
                continue
            original_mic_indices_to_process.append(idx)
            mic_positions_for_pra.append(mic_coord)

        if len(mic_positions_for_pra) < 1:
            return files_created_count, actual_mic_positions_in_room

        current_max_order = 3 # Default for non-anechoic, can be adjusted by pyroomacoustics
        e_absorption = 0.1 # Default
        if is_anechoic:
            # For anechoic, rt60_tgt is small, max_order is 0.
            # inverse_sabine might fail if rt60_tgt is too small for the room size.
            try:
                e_absorption, _ = pra.inverse_sabine(rt60_tgt if rt60_tgt > 0 else 0.01, room_dim)
            except ValueError: # Likely absorption > 1.0
                 print(f"WARN: inverse_sabine failed for anechoic {base_output_filename_prefix} (RT60={rt60_tgt}, Room={room_dim}). Using default absorption for material.")
                 # e_absorption remains default, but max_order=0 is key.
            current_max_order = 0
        else:
            e_absorption, current_max_order = pra.inverse_sabine(rt60_tgt, room_dim)

        room = pra.ShoeBox(
            room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=current_max_order
        )
        room.add_source(source_pos)
        mic_array_for_pra_np = np.array(mic_positions_for_pra).T
        room.add_microphone_array(pra.MicrophoneArray(mic_array_for_pra_np, fs=room.fs))
        actual_mic_positions_in_room = mic_positions_for_pra
        room.compute_rir()

        for pyroom_mic_idx in range(len(mic_positions_for_pra)):
            original_mic_idx_in_config = original_mic_indices_to_process[pyroom_mic_idx]
            rir_signal = room.rir[pyroom_mic_idx][0]
            output_filename_mic = f"{base_output_filename_prefix}_micidx_{original_mic_idx_in_config}.wav"
            sf.write(output_filename_mic, rir_signal, fs)
            files_created_count += 1
    except ValueError as ve:
        print(f"ERROR: PRA ValueError for {base_output_filename_prefix}: {ve}. Params: RT60={rt60_tgt}, Room={room_dim}")
    except Exception as e:
        print(f"ERROR: Generic error in create_rir_example for {base_output_filename_prefix}: {e}")
    return files_created_count, actual_mic_positions_in_room

# ...existing code...

if __name__ == "__main__":
    print("--- simulation.py: Iniciando generación de RIRs y metadatos ---")
    FS = 48000
    OUTPUT_DIR = "rir_dataset_user_defined"
    METADATA_FILENAME = "simulation_metadata.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    min_wall_dist = 0.5
    min_src_mic_dist = 0.5

    user_defined_simulations = []

    # Definiciones de escenarios según solicitud del usuario
    room_dim = [10.0, 10.0, 10.0]
    array_center = [2.5, 2.5, 2.5]
    mic_counts = [4]
    mic_separation = 0.10
    rt60_values = [0.01, 0.05, 0.3, 0.7]
    angles = range(0, 181, 45)  # <-- Angulos de 0 a 180 grados en pasos de 10
    elevations = [0]  # Ángulos de elevación en grados

    for rt60 in rt60_values:
        for n_mics in mic_counts:
            for angle in angles:
                for elevation_deg in elevations:
                    rad = np.deg2rad(angle)
                    elev_rad = np.deg2rad(elevation_deg)
                    dist = 2.0  # Distancia radial al centro del array

                    # Coordenadas polares a cartesianas con elevación
                    xy_dist = dist * np.cos(elev_rad)
                    src_x = array_center[0] + xy_dist * np.cos(rad)
                    src_y = array_center[1] + xy_dist * np.sin(rad)
                    src_z = array_center[2] + dist * np.sin(elev_rad)
                    source_pos = [src_x, src_y, src_z]

                    # Calcula posiciones de micrófonos (array lineal sobre eje x)
                    half_array_len = ((n_mics - 1) / 2.0) * mic_separation
                    mic_positions = []
                    for i in range(n_mics):
                        offset = i * mic_separation - half_array_len
                        mic_pos = [array_center[0] + offset, array_center[1], array_center[2]]
                        mic_positions.append(mic_pos)

                    config_id_suffix_str = f"room10x10x10_rt60_{rt60}_mics_{n_mics}_az_{angle}_el_{elevation_deg}"
                    sim_config_entry = {
                    "config_id_suffix": config_id_suffix_str,
                    "room_dim": list(room_dim),
                    "rt60_tgt": rt60,
                    "is_anechoic": rt60 < 0.1,
                    "array_center_target": list(array_center),
                    "array_orientation_axis": 'x',
                    "source_pos": list(source_pos),
                    "num_mics": n_mics,
                    "azimuth_deg": angle,
                    "elevation_deg": elevation_deg
                    }
                    user_defined_simulations.append(sim_config_entry)
        num_random_configs = 0 # Asegurar que no se generen configuraciones aleatorias

    all_metadata_entries = []
    total_rirs_generated_overall = 0
    successful_configurations_count = 0
    max_configured_mics_overall = 0
    if user_defined_simulations:
        for sim_conf in user_defined_simulations:
            max_configured_mics_overall = max(max_configured_mics_overall, sim_conf.get("num_mics", 4))
    if max_configured_mics_overall == 0 and user_defined_simulations: # Debería haber simulaciones
        max_configured_mics_overall = 8 # Maximo esperado de las configs definidas

    print(f"\n--- Iniciando procesamiento de {len(user_defined_simulations)} configuraciones totales ---")
    for idx, config in enumerate(user_defined_simulations):
        config_name_for_print = config.get('config_id_suffix', f'unnamed_config_{idx}')
        print(f"\nProcesando configuración {idx+1}/{len(user_defined_simulations)}: {config_name_for_print}")

        NUM_MICS = config.get("num_mics", 4)
        MIC_SEPARATION = 0.10
        ARRAY_LENGTH = (NUM_MICS - 1) * MIC_SEPARATION
        room_dim_current = config["room_dim"]
        rt60_current = config["rt60_tgt"]
        is_anechoic_current = config["is_anechoic"]
        array_center_config_current = config["array_center_target"]
        orientation_current = config["array_orientation_axis"].lower()
        source_pos_current = config["source_pos"]
        axis_map = {'x':0, 'y':1, 'z':2}; axis_idx_current = axis_map.get(orientation_current)

        if axis_idx_current is None:
            print(f"INFO: Config {config_name_for_print} saltada. Orientación inválida: {orientation_current}"); continue
        if room_dim_current[axis_idx_current] < ARRAY_LENGTH + 2 * min_wall_dist:
            print(f"INFO: Config {config_name_for_print} saltada. Array ({ARRAY_LENGTH}m) muy largo para la sala ({room_dim_current[axis_idx_current]}m en eje {orientation_current} con márgenes)."); continue

        mic_positions_calculated = []
        half_array_len = ((NUM_MICS - 1) / 2.0) * MIC_SEPARATION
        for i in range(NUM_MICS):
            offset = i * MIC_SEPARATION - half_array_len
            pos = list(array_center_config_current); pos[axis_idx_current] = round(array_center_config_current[axis_idx_current] + offset, 2)
            mic_positions_calculated.append(pos)

        current_config_valid = True
        combined_points_for_validation = mic_positions_calculated + [source_pos_current]
        for p_idx, p_coord in enumerate(combined_points_for_validation):
            point_label = "Fuente" if (p_idx == len(mic_positions_calculated)) else f"Micrófono {p_idx}"
            for dim_idx_val, dim_name in enumerate(['x', 'y', 'z']):
                if not (min_wall_dist <= p_coord[dim_idx_val] < room_dim_current[dim_idx_val] - min_wall_dist):
                    print(f"INFO: Config {config_name_for_print} saltada. {point_label} {p_coord} viola min_wall_dist ({min_wall_dist}m) en eje '{dim_name}' de sala {room_dim_current}."); current_config_valid = False; break
            if not current_config_valid: break
        if not current_config_valid: continue
        for mic_idx_val, mp_coord_val in enumerate(mic_positions_calculated):
            if np.linalg.norm(np.array(mp_coord_val) - np.array(source_pos_current)) < min_src_mic_dist:
                print(f"INFO: Config {config_name_for_print} saltada. Mic {mic_idx_val} ({mp_coord_val}) muy cerca de fuente {source_pos_current} (min_src_mic_dist: {min_src_mic_dist}m)."); current_config_valid = False; break
        if not current_config_valid: continue

        config_id_str_current = config_name_for_print # Usar el suffix directamente
        rir_base_filename_prefix = os.path.join(OUTPUT_DIR, f"rir_{config_id_str_current}")

        rirs_created_for_config, mic_positions_used_by_pra = create_rir_example(
            rir_base_filename_prefix, rt60_current, room_dim_current, source_pos_current,
            mic_positions_calculated, FS, is_anechoic_current)

        if rirs_created_for_config == 0:
             print(f"INFO: Config {config_id_str_current} 0 RIRs. Omitiendo."); continue
        if rirs_created_for_config < NUM_MICS:
            print(f"INFO: Config {config_id_str_current} {rirs_created_for_config}/{NUM_MICS} RIRs (incompleto). Omitiendo y limpiando.")
            for i_mic_cleanup in range(NUM_MICS): # Limpiar RIRs parciales si se crearon
                 temp_fn_cleanup = f"{rir_base_filename_prefix}_micidx_{i_mic_cleanup}.wav"
                 if os.path.exists(temp_fn_cleanup):
                     try: os.remove(temp_fn_cleanup)
                     except OSError as e: print(f"WARN: No se pudo limpiar {temp_fn_cleanup}: {e}")
            continue

        total_rirs_generated_overall += rirs_created_for_config; successful_configurations_count +=1
        actual_array_center_coords = []
        dist_src_actual_val, azimuth_src_actual_val, elevation_src_actual_val = "N/A", "N/A", "N/A"
        if mic_positions_used_by_pra: # Debería ser true si rirs_created_for_config > 0
            actual_array_center_coords = list(np.mean(mic_positions_used_by_pra, axis=0))
            dist_src_actual_val, azimuth_src_actual_val, elevation_src_actual_val = calculate_angle_and_distance(source_pos_current, actual_array_center_coords)

        entry = {"config_id": config_id_str_current, "fs_hz": FS,
            "room_dim_x": room_dim_current[0],
            "room_dim_y": room_dim_current[1],
            "room_dim_z": room_dim_current[2],
            "source_pos_x": source_pos_current[0],
            "source_pos_y": source_pos_current[1],
            "source_pos_z": source_pos_current[2],
            "rt60_target_s": rt60_current,
            "angle_deg": angle,
            "azimuth_deg": angle,
            "array_center_x_config": array_center_config_current[0], "array_center_y_config": array_center_config_current[1], "array_center_z_config": array_center_config_current[2],
            "array_orientation_axis": orientation_current, "num_mics_configured": NUM_MICS, "num_mics_processed": rirs_created_for_config,
            "mic_separation_m": MIC_SEPARATION, "rir_file_basename": os.path.basename(rir_base_filename_prefix)}
        if actual_array_center_coords: # Solo añadir si se calcularon
            entry.update({
                "array_center_x_actual": round(actual_array_center_coords[0],2),
                "array_center_y_actual": round(actual_array_center_coords[1],2),
                "array_center_z_actual": round(actual_array_center_coords[2],2),
                "actual_dist_src_to_array_center_m": round(dist_src_actual_val,3),
                "actual_azimuth_src_to_array_center_deg": round(azimuth_src_actual_val,2),
                "actual_elevation_src_to_array_center_deg": round(elevation_src_actual_val, 2) # Added elevation
            })
        else:
            entry.update({
                "array_center_x_actual": "N/A", "array_center_y_actual": "N/A", "array_center_z_actual": "N/A",
                "actual_dist_src_to_array_center_m": "N/A",
                "actual_azimuth_src_to_array_center_deg": "N/A",
                "actual_elevation_src_to_array_center_deg": "N/A" # Added elevation
            })
        for i_mic_meta, mic_c_meta in enumerate(mic_positions_used_by_pra): # Usar mic_positions_used_by_pra que son las reales
            entry.update({f"mic{i_mic_meta}_pos_x": round(mic_c_meta[0],2), f"mic{i_mic_meta}_pos_y": round(mic_c_meta[1],2), f"mic{i_mic_meta}_pos_z": round(mic_c_meta[2],2)})
        all_metadata_entries.append(entry)
        print(f"INFO: Configuración {config_id_str_current} procesada con {rirs_created_for_config} RIRs.")

    if all_metadata_entries:
        metadata_filepath = os.path.join(OUTPUT_DIR, METADATA_FILENAME)
        # Definir todos los campos posibles para el CSV, incluyendo todas las posiciones de micrófonos hasta max_configured_mics_overall
        ordered_base_fieldnames = ["config_id", "fs_hz", "room_dim_x", "room_dim_y", "room_dim_z", "rt60_target_s", "is_anechoic",
            "source_pos_x", "source_pos_y", "source_pos_z", "array_center_x_config", "array_center_y_config", "array_center_z_config",
            "array_center_x_actual", "array_center_y_actual", "array_center_z_actual", "array_orientation_axis",
            "actual_dist_src_to_array_center_m", "actual_azimuth_src_to_array_center_deg", "angle_deg" ,
            "actual_elevation_src_to_array_center_deg", "num_mics_configured", # Added elevation
            "num_mics_processed", "mic_separation_m", "rir_file_basename", "azimuth_deg", 'elevation_deg', 'snr_db', 'tdoa_method']

        mic_coord_fieldnames = []
        if max_configured_mics_overall > 0: # Asegurar que hay mics para añadir columnas
            for i in range(max_configured_mics_overall):
                mic_coord_fieldnames.extend([f"mic{i}_pos_x", f"mic{i}_pos_y", f"mic{i}_pos_z"])

        # Unir todos los nombres de campo, asegurando que los de micrófono estén al final
        final_fieldnames = ordered_base_fieldnames + mic_coord_fieldnames

        # Eliminar duplicados si alguna clave de mic_coord ya estaba en ordered_base_fieldnames (no debería)
        final_fieldnames = sorted(list(set(final_fieldnames)), key=lambda x: (x not in mic_coord_fieldnames, final_fieldnames.index(x) if x in final_fieldnames else float('inf')))


        try:
            with open(metadata_filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=final_fieldnames, quoting=csv.QUOTE_NONNUMERIC, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_metadata_entries)
            print(f"Metadatos guardados en: {metadata_filepath}")
        except Exception as e: print(f"ERROR escribiendo CSV: {e}")
    else: print("No se generaron metadatos.")
    print(f"\nTotal RIRs generadas: {total_rirs_generated_overall}")
    print(f"Configuraciones exitosas: {successful_configurations_count}")
    print("--- simulation.py: Finalizado ---")