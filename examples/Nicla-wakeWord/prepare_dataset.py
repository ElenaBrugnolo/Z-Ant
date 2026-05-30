import os
import json
import numpy as np

# --- 1. FUNZIONI MATEMATICHE Z-ANT (Dal Paper) ---
# GASF captures static angular similarity between time instants, while GADF captures directional change

def min_max_normalize(series):
    s_min, s_max = np.min(series), np.max(series)
    if s_max == s_min:
        return np.zeros_like(series)
    return ((series - s_max) + (series - s_min)) / (s_max - s_min)

def compute_gasf(series_norm):
    c = series_norm
    s = np.sqrt(np.clip(1.0 - c**2, 0, 1)) 
    return np.outer(c, c) - np.outer(s, s)

def compute_gadf(series_norm):
    c = series_norm
    s = np.sqrt(np.clip(1.0 - c**2, 0, 1))
    return np.outer(s, c) - np.outer(c, s)

def compute_mtf(series, q=4):
    n = len(series)
    sorted_series = np.sort(series)
    boundaries = [sorted_series[int(k * n / q)] for k in range(1, q)]
    
    bins = np.zeros(n, dtype=int)
    for i, x in enumerate(series):
        bins[i] = sum(1 for b in boundaries if x >= b)
        
    w = np.zeros((q, q))
    for t in range(n - 1):
        w[bins[t], bins[t+1]] += 1
        
    w_hat = np.zeros_like(w)
    for i in range(q):
        row_sum = np.sum(w[i, :])
        if row_sum > 0:
            w_hat[i, :] = w[i, :] / row_sum
            
    mtf = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mtf[i, j] = w_hat[bins[i], bins[j]]
            
    return mtf

def generate_zant_image(series_64):
    series_norm = min_max_normalize(series_64)
    gasf = compute_gasf(series_norm)
    gadf = compute_gadf(series_norm)
    mtf = compute_mtf(series_64, q=4)
    
    gasf_scaled = (gasf + 1.0) / 2.0
    gadf_scaled = (gadf + 1.0) / 2.0
    
    chw_tensor = np.stack([gasf_scaled, gadf_scaled, mtf], axis=0)
    return np.clip(chw_tensor, 0.0, 1.0)

# --- 2. ELABORAZIONE AUDIO (Da JSON a 64 Feature RMS) ---

def process_json_to_image(json_path):
    # Leggi il file JSON di Edge Impulse
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Estrai l'array dei valori (16000 campioni)
    raw_audio = np.array(data['payload']['values'])
    
    # Assicuriamoci che siano esattamente 16000 campioni
    if len(raw_audio) > 16000:
        raw_audio = raw_audio[:16000]
    elif len(raw_audio) < 16000:
        raw_audio = np.pad(raw_audio, (0, 16000 - len(raw_audio)))
        
    # Dividi in 64 cesti (16000 / 64 = 250 campioni l'uno)
    windows = np.split(raw_audio, 64)
    
    # Calcola l'energia (RMS) per ogni cesto
    rms_features = np.zeros(64)
    for i, window in enumerate(windows):
        rms_features[i] = np.sqrt(np.mean(window**2))
        
    # Trasforma i 64 valori RMS nell'immagine 3x64x64 finale
    final_image = generate_zant_image(rms_features)
    
    return final_image

# --- 3. CONVERSIONE DI MASSA DEL DATASET ---

def convert_dataset_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Lista tutti i file JSON nella cartella
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    print(f"Trovati {len(json_files)} file in {input_folder}. Inizio conversione...")
    
    for filename in json_files:
        in_path = os.path.join(input_folder, filename)
        
        # Sostituisci l'estensione .json con .npy
        out_filename = filename.replace('.json', '.npy')
        out_path = os.path.join(output_folder, out_filename)
        
        # Genera l'immagine e salvala
        image_tensor = process_json_to_image(in_path)
        np.save(out_path, image_tensor)
        
    print(f"Conversione completata! {len(json_files)} immagini salvate in {output_folder}\n")

if __name__ == "__main__":
    # Percorsi base
    base_dir = r"C:\Users\Lenovo\Desktop\Embedded systems\Z-Ant\examples\Nicla-wakeWord"
    
    # Input
    train_in = os.path.join(base_dir, "dataset", "training")
    test_in = os.path.join(base_dir, "dataset", "testing")
    
    # Output (Cartelle dove andranno i file NumPy pronti per la rete neurale)
    train_out = os.path.join(base_dir, "dataset_numpy", "training")
    test_out = os.path.join(base_dir, "dataset_numpy", "testing")
    
    convert_dataset_folder(train_in, train_out)
    convert_dataset_folder(test_in, test_out)