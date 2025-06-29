import soundfile as sf
import os

def load_signal_from_wav(filename, target_fs=48000):
    """
    Carga un archivo de audio .wav, verifica su frecuencia de muestreo y la normaliza.

    Parameters:
    filename (str): Ruta al archivo .wav.
    target_fs (int): La frecuencia de muestreo deseada (por defecto 48000 Hz).

    Returns:
    tuple: (signal, fs) donde signal es el array de numpy y fs es la frecuencia de muestreo.
           Devuelve (None, None) si hay un error (archivo no encontrado, fs incorrecta).

    Raises:
    ValueError: Si la frecuencia de muestreo del archivo no coincide con target_fs.
    FileNotFoundError: Si el archivo no se encuentra.
    """
    try:
        signal, original_fs = sf.read(filename)
        print(f"Archivo '{filename}' cargado. Frecuencia de muestreo original: {original_fs} Hz.")

        if original_fs != target_fs:
            error_message = (
                f"Error: La frecuencia de muestreo del archivo '{filename}' es {original_fs} Hz, "
                f"pero se esperaba {target_fs} Hz. No se realizará remuestreo."
            )
            print(error_message)
            raise ValueError(error_message)

        # Si querés normalizar la señal entre -1 y 1, descomenta estas líneas:
        # max_val = np.max(np.abs(signal))
        # if max_val > 0:
        #     signal = signal / max_val

        return signal, original_fs

    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en la ruta especificada: {filename}")
        return None, None
    except Exception as e:
        print(f"Error al cargar o procesar el archivo '{filename}': {e}")
        return None, None


# Uso del código con el path que diste:
filepath = "p336_007.wav"
signal, fs = load_signal_from_wav(filepath, target_fs=48000)

if signal is not None:
    print(f"Se cargó la señal con éxito. Duración: {len(signal)/fs:.2f} segundos.")
else:
    print("No se pudo cargar la señal.")
