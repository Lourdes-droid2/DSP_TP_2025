import pandas as pd
import numpy as np

def analyze_results():
    csv_path = "full_experiment_results.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: El archivo {csv_path} no fue encontrado.")
        return
    except Exception as e:
        print(f"Error cargando el CSV {csv_path}: {e}")
        return

    print("Archivo CSV cargado exitosamente.")
    print("\nColumnas disponibles:")
    print(df.columns.tolist())

    # Filtrar solo las filas que corresponden a resultados de DOA de array promedio
    # ya que estas son las que tienen 'doa_array_error_deg'
    df_array_doa = df[df['mic_pair'] == 'array_avg_adj_pairs'].copy() # Usar .copy() para evitar SettingWithCopyWarning

    if df_array_doa.empty:
        print("\nNo se encontraron resultados de DOA de array ('array_avg_adj_pairs').")
        # Podríamos analizar 'doa_estimated_from_pair_deg' si es necesario,
        # pero el plan se enfoca en el error del array.
        # Por ahora, intentaremos analizar el error de DOA por par si el error de array no está.
        if 'doa_estimated_from_pair_deg' in df.columns and 'actual_azimuth_src_to_array_center_deg' in df.columns:
            print("Intentando analizar error de DOA por par...")
            # Calcular error de DOA por par. Primero asegurar que real_doa_deg está disponible para todos los pares.
            # Esto requiere un merge o una forma de asignar el real_doa_deg general a cada par.
            # La columna 'actual_azimuth_src_to_array_center_deg' ya está en cada fila por sim_params.to_dict()
            df['doa_pair_error_deg'] = df['doa_estimated_from_pair_deg'] - df['actual_azimuth_src_to_array_center_deg']

            # Agrupar por config_id, tdoa_method (para pares), y snr_db
            # Usar 'tdoa_method' en lugar de 'tdoa_method_for_avg_doa'
            if not df[['config_id', 'tdoa_method', 'snr_db', 'doa_pair_error_deg']].isnull().all().all():
                 # Calcular la media del valor absoluto del error para tener una métrica de "qué tan lejos" estamos, sin importar la dirección.
                df['abs_doa_pair_error_deg'] = df['doa_pair_error_deg'].abs()
                summary_stats_pair = df.groupby(['config_id', 'tdoa_method', 'snr_db'])['abs_doa_pair_error_deg'].agg(['mean', 'std', 'count'])
                print("\nEstadísticas de error absoluto de DOA por par (media, std, count):")
                print(summary_stats_pair)
            else:
                print("No hay suficientes datos para el análisis de error de DOA por par.")

        else:
            print("Columnas necesarias para el análisis de error de DOA por par no encontradas.")
        return

    print(f"\nEncontradas {len(df_array_doa)} filas para análisis de DOA de array.")

    # Convertir la columna de error a numérico, errores a NaN
    df_array_doa.loc[:, 'doa_array_error_deg'] = pd.to_numeric(df_array_doa['doa_array_error_deg'], errors='coerce')
    df_array_doa.loc[:, 'abs_doa_array_error_deg'] = df_array_doa['doa_array_error_deg'].abs()


    # Agrupar y calcular estadísticas
    # Los campos relevantes para agrupar son config_id, tdoa_method_for_avg_doa, snr_db
    grouping_fields = ['config_id', 'tdoa_method_for_avg_doa', 'snr_db']

    # Verificar si todas las columnas de agrupación existen
    missing_cols = [col for col in grouping_fields if col not in df_array_doa.columns]
    if missing_cols:
        print(f"Error: Faltan las siguientes columnas necesarias para agrupar: {missing_cols}")
        return

    if df_array_doa['abs_doa_array_error_deg'].isnull().all():
        print("\nLa columna 'abs_doa_array_error_deg' solo contiene NaNs. No se pueden calcular estadísticas.")
    else:
        summary_stats = df_array_doa.groupby(grouping_fields)['abs_doa_array_error_deg'].agg(['mean', 'std', 'count'])
        print("\nEstadísticas de error absoluto de DOA de array (media, std, count):")
        print(summary_stats)

        print("\n--- Config IDs presentes en el análisis de array DOA ---")
        print(df_array_doa['config_id'].unique())

        print("\n--- Ejemplo de datos para una configuración y SNR específicos (Array DOA) ---")
        # Intentar mostrar un ejemplo más específico
        # Tomar la primera config_id y el primer snr_db de los datos filtrados
        if not df_array_doa.empty:
            sample_config_id = df_array_doa['config_id'].unique()[0]
            sample_snr = df_array_doa['snr_db'].unique()[0]
            sample_data = df_array_doa[(df_array_doa['config_id'] == sample_config_id) & (df_array_doa['snr_db'] == sample_snr)]
            print(f"Datos para {sample_config_id} con SNR {sample_snr} dB:")
            print(sample_data[['tdoa_method_for_avg_doa', 'doa_array_estimated_deg', 'doa_array_real_deg', 'doa_array_error_deg', 'abs_doa_array_error_deg']])
        else:
            print("No hay datos de array DOA para mostrar un ejemplo.")

if __name__ == "__main__":
    analyze_results()
