import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Cargo datos
print("Attempting to load full_experiment_results.csv...")
try:
    df = pd.read_csv("full_experiment_results.csv")
    print("CSV loaded successfully.")
    print("Shape of df:", df.shape)
    print("Head of df:\n", df.head())
    print("Columns in df:", df.columns.tolist())
    print("Data types of df columns:\n", df.dtypes)
except FileNotFoundError:
    print("ERROR: full_experiment_results.csv not found! Please run main.py to generate it.")
    exit()
except Exception as e:
    print(f"ERROR loading full_experiment_results.csv: {e}")
    exit()

# Filtrar sólo datos válidos de DOA promedio para el arreglo
print("\nFiltering df_array...")
df_array = df[(df['mic_pair'] == 'array_avg_adj_pairs') & df['doa_array_error_deg'].notna()].copy() # Added .copy() to avoid SettingWithCopyWarning
print("Shape of df_array:", df_array.shape)
print("Head of df_array:\n", df_array.head())

# Verificar columnas necesarias antes de la conversión
# Note: The plotting script uses 'elevation_angle_deg', but the CSV from simulation.py will have 'actual_elevation_src_to_array_center_deg'
# We will rename it after loading or adjust the plotting script to use the new name.
# For now, let's adjust the script to expect the new name.
required_cols_for_conversion = ['actual_elevation_src_to_array_center_deg', 'actual_dist_src_to_array_center_m',
                                'mic_separation_m', 'num_mics_processed', 'rt60_target_s', 'fs_hz']
missing_cols = [col for col in required_cols_for_conversion if col not in df_array.columns]

# Attempt to rename if old name 'elevation_angle_deg' is present and new one is missing
if 'actual_elevation_src_to_array_center_deg' not in df_array.columns and 'elevation_angle_deg' in df_array.columns:
    print("Info: 'actual_elevation_src_to_array_center_deg' not found, but 'elevation_angle_deg' found. Renaming for compatibility.")
    df_array.rename(columns={'elevation_angle_deg': 'actual_elevation_src_to_array_center_deg'}, inplace=True)
    # Re-check missing_cols after potential rename
    missing_cols = [col for col in required_cols_for_conversion if col not in df_array.columns]

if missing_cols:
    print(f"ERROR: The following required columns are missing from df_array: {missing_cols}")
    # Potentially print df_array.columns here to show what IS available
    print("Available columns in df_array:", df_array.columns.tolist())
    # Decide whether to exit or proceed with caution
    # For now, let's print a warning and try to continue, as some plots might still work
    # if their specific columns are present.
    # exit() # Or handle more gracefully

# Asegurarse que columnas numéricas estén en el tipo adecuado
print("\nConverting columns to numeric...")
for col in required_cols_for_conversion:
    if col in df_array.columns:
        df_array.loc[:, col] = pd.to_numeric(df_array[col], errors='coerce')
        print(f"Converted {col} to numeric. NaN count: {df_array[col].isna().sum()}")
    else:
        print(f"Warning: Column {col} not found for numeric conversion.")


print("\nData types of df_array columns after conversion:\n", df_array.dtypes)
print("Head of df_array after conversion:\n", df_array.head())


# Para todos los gráficos, filtramos SNR fijo para claridad (ejemplo 10 dB)
snr_target = 10
print(f"\nFiltering df_snr for snr_db == {snr_target}...")
if 'snr_db' in df_array.columns:
    df_snr = df_array[df_array['snr_db'] == snr_target].copy() # Added .copy()
    print("Shape of df_snr:", df_snr.shape)
    print("Head of df_snr:\n", df_snr.head())
    if df_snr.empty:
        print(f"WARNING: df_snr is empty after filtering for snr_db == {snr_target}. Plots using df_snr will likely be blank.")
else:
    print("ERROR: 'snr_db' column not found in df_array. Cannot create df_snr.")
    df_snr = pd.DataFrame() # Create empty DataFrame to avoid further errors if script continues

# Columnas clave para los plots
# Adjusted 'elevation_angle_deg' to 'actual_elevation_src_to_array_center_deg'
plot_x_cols = ['actual_elevation_src_to_array_center_deg', 'actual_dist_src_to_array_center_m', 'mic_separation_m', 'num_mics_processed', 'rt60_target_s', 'fs_hz', 'snr_db']
plot_y_col = 'doa_array_error_deg'
plot_hue_col = 'tdoa_method_for_avg_doa'
plot_hue_col2 = 'fs_hz' # for the last plot

print("\nChecking unique values for key plotting columns in df_snr (if not empty):")
if not df_snr.empty:
    for col in plot_x_cols:
        if col in df_snr.columns and col != 'snr_db': # snr_db is fixed for df_snr
            print(f"Unique values in df_snr['{col}']: {df_snr[col].unique()[:20]} (NaNs: {df_snr[col].isna().sum()})")
    if plot_y_col in df_snr.columns:
        print(f"Unique values in df_snr['{plot_y_col}']: {df_snr[plot_y_col].unique()[:20]} (NaNs: {df_snr[plot_y_col].isna().sum()})")
    if plot_hue_col in df_snr.columns:
        print(f"Unique values in df_snr['{plot_hue_col}']: {df_snr[plot_hue_col].unique()[:20]} (NaNs: {df_snr[plot_hue_col].isna().sum()})")
else:
    print("df_snr is empty, skipping unique value checks for it.")

print("\nChecking unique values for key plotting columns in df_array (for the last plot):")
if not df_array.empty:
    if 'snr_db' in df_array.columns:
         print(f"Unique values in df_array['snr_db']: {df_array['snr_db'].unique()[:20]} (NaNs: {df_array['snr_db'].isna().sum()})")
    if plot_y_col in df_array.columns:
        print(f"Unique values in df_array['{plot_y_col}']: {df_array[plot_y_col].unique()[:20]} (NaNs: {df_array[plot_y_col].isna().sum()})")
    if plot_hue_col2 in df_array.columns:
        print(f"Unique values in df_array['{plot_hue_col2}']: {df_array[plot_hue_col2].unique()[:20]} (NaNs: {df_array[plot_hue_col2].isna().sum()})")
else:
    print("df_array is empty, skipping unique value checks for it (relevant for the last plot).")


# Función general para línea con promedio y error estándar
def plot_metric_vs_param(df_plot, param_col, title, xlabel):
    print(f"\nAttempting to generate plot: {title}")
    if df_plot.empty:
        print(f"Skipping plot '{title}' because input DataFrame is empty.")
        return
    if param_col not in df_plot.columns:
        print(f"Skipping plot '{title}' because column '{param_col}' is missing from the DataFrame.")
        return
    if df_plot[param_col].isna().all():
        print(f"Skipping plot '{title}' because column '{param_col}' contains all NaN values.")
        return
    if 'doa_array_error_deg' not in df_plot.columns or df_plot['doa_array_error_deg'].isna().all():
        print(f"Skipping plot '{title}' because 'doa_array_error_deg' is missing or all NaN.")
        return
    if 'tdoa_method_for_avg_doa' not in df_plot.columns or df_plot['tdoa_method_for_avg_doa'].isna().all():
        print(f"Skipping plot '{title}' because 'tdoa_method_for_avg_doa' is missing or all NaN (used for hue).")
        # Alternatively, plot without hue if this is the case
        # return

    plt.figure(figsize=(8,5))
    sns.lineplot(
        data=df_plot, # Corrected: was df, should be df_plot
        x=param_col,
        y='doa_array_error_deg',
        hue='tdoa_method_for_avg_doa',
        errorbar='sd', # intervalo de confianza ±1 std dev (ci was deprecated)
        marker='o'
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Error promedio DOA (grados)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- 1. Error vs Ángulo de elevación de la fuente ---
plot_metric_vs_param(
    df_snr,
    'actual_elevation_src_to_array_center_deg', # Corrected column name
    "Error promedio DOA vs Ángulo de elevación de la fuente (SNR 10 dB)",
    "Ángulo de elevación (grados)"
)

# --- 2. Error vs Distancia fuente - arreglo ---
plot_metric_vs_param(
    df_snr,
    'actual_dist_src_to_array_center_m',
    "Error promedio DOA vs Distancia fuente-arreglo (SNR 10 dB)",
    "Distancia (m)"
)

# --- 3. Error vs Separación entre micrófonos ---
plot_metric_vs_param(
    df_snr,
    'mic_separation_m',
    "Error promedio DOA vs Separación entre micrófonos (SNR 10 dB)",
    "Separación entre micrófonos (m)"
)

# --- 4. Error vs Cantidad de micrófonos ---
plot_metric_vs_param(
    df_snr,
    'num_mics_processed',
    "Error promedio DOA vs Cantidad de micrófonos (SNR 10 dB)",
    "Cantidad de micrófonos"
)

# --- 5. Error vs Tiempo de reverberación RT60 ---
plot_metric_vs_param(
    df_snr,
    'rt60_target_s',
    "Error promedio DOA vs Tiempo de reverberación RT60 (SNR 10 dB)",
    "Tiempo RT60 (s)"
)

# --- 6. Error vs Frecuencia de muestreo ---
plot_metric_vs_param(
    df_snr,
    'fs_hz',
    "Error promedio DOA vs Frecuencia de muestreo (SNR 10 dB)",
    "Frecuencia de muestreo (Hz)"
)

# --- 7. Gráfico adicional: Error promedio DOA vs SNR para todas las frecuencias de muestreo ---
title_last_plot = "Error promedio DOA vs SNR para distintas frecuencias de muestreo"
print(f"\nAttempting to generate plot: {title_last_plot}")
if df_array.empty:
    print(f"Skipping plot '{title_last_plot}' because df_array is empty.")
elif 'snr_db' not in df_array.columns or df_array['snr_db'].isna().all():
    print(f"Skipping plot '{title_last_plot}' because 'snr_db' is missing or all NaN in df_array.")
elif 'doa_array_error_deg' not in df_array.columns or df_array['doa_array_error_deg'].isna().all():
    print(f"Skipping plot '{title_last_plot}' because 'doa_array_error_deg' is missing or all NaN in df_array.")
elif 'fs_hz' not in df_array.columns or df_array['fs_hz'].isna().all():
    print(f"Skipping plot '{title_last_plot}' because 'fs_hz' (hue) is missing or all NaN in df_array.")
else:
    plt.figure(figsize=(8,5))
    sns.lineplot(
        data=df_array,
        x='snr_db',
        y='doa_array_error_deg',
        hue='fs_hz',
        errorbar='sd', # ci was deprecated
        marker='o',
        palette='viridis'
    )
    plt.title(title_last_plot)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Error promedio DOA (grados)")
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("plot_7_snr_vs_doa_error.png") # Optional: savefig
    # print("Saved plot_7_snr_vs_doa_error.png")
    plt.show()

print("\n--- Script generate_plots.py finished ---")
