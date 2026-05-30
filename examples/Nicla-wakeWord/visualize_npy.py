import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_zant_tensor(npy_path):
    # 1. Carica il tensore Numpy
    tensor = np.load(npy_path)
    
    print(f"File caricato: {os.path.basename(npy_path)}")
    print(f"Shape del tensore: {tensor.shape} (Dovrebbe essere 3, 64, 64)")
    print(f"Range valori: Min = {np.min(tensor):.4f}, Max = {np.max(tensor):.4f} (Dovrebbe essere tra 0 e 1)")
    
    # Assicuriamoci che sia un 3x64x64
    if tensor.shape != (3, 64, 64):
        print("ERRORE: La shape del tensore non è quella prevista!")
        return

    # Estrai i singoli canali
    gasf = tensor[0]
    gadf = tensor[1]
    mtf = tensor[2]

    # --- SETUP DEL PLOT (Stile Paper) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Usa una colormap che metta in risalto i contrasti (es. 'viridis' o 'plasma')
    cmap_choice = 'viridis' 
    
    # 1. GASF
    im1 = axes[0].imshow(gasf, cmap=cmap_choice, origin='lower', vmin=0, vmax=1)
    axes[0].set_title('Channel 0: GASF')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. GADF
    im2 = axes[1].imshow(gadf, cmap=cmap_choice, origin='lower', vmin=0, vmax=1)
    axes[1].set_title('Channel 1: GADF')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. MTF
    im3 = axes[2].imshow(mtf, cmap=cmap_choice, origin='lower', vmin=0, vmax=1)
    axes[2].set_title('Channel 2: MTF')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Visualizzazione Tensor Z-ANT\n{os.path.basename(npy_path)}", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Inserisci qui il percorso di uno dei file .npy che hai appena caricato
    # (Adatta il percorso alla cartella in cui li hai salvati)
    file_da_testare = r"dataset_numpy\training\heyfranco.6olmsknn.ingestion-8684c9459f-n9j72.s1.npy"
    
    if os.path.exists(file_da_testare):
        visualize_zant_tensor(file_da_testare)
    else:
        print(f"File non trovato: {file_da_testare}")