import os
import cv2
import random

# Nustatome projekto šaknį, kad keliai veiktų nepriklausomai nuo paleidimo direktorijos
this_dir = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(this_dir, '..'))
VIDEO_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUT_DIR   = os.path.join(BASE_DIR, 'data', 'frames')
# Nustatykite norimą kadrų per sekundę skaičių (FPS) arba None, jei naudosite atsitiktinį imties dydį
FPS_EXTRACT = None    # jei norime naudoti tik atsitiktinę imtį, nustatyk None
TOTAL_SAMPLES = 600      # bendras kadrų skaičius, kurį norime sugeneruoti iš vieno video
# Jei nori papildomo apribojimo fps, nustatyk FPS_EXTRACT; jei ne, naudok tik TOTAL_SAMPLES  # maksimalus kadrų skaičius iš kiekvieno video

# Sukuriame OUT_DIR, jei neegzistuoja
os.makedirs(OUT_DIR, exist_ok=True)

for video in os.listdir(VIDEO_DIR):
    path = os.path.join(VIDEO_DIR, video)
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Generuojame visus galimus rėžius
    all_indices = list(range(total_frames))
    # Išrenkame atsitiktinius 600 kadrų
    if TOTAL_SAMPLES and total_frames > TOTAL_SAMPLES:
        frame_indices = sorted(random.sample(all_indices, TOTAL_SAMPLES))
    else:
        frame_indices = all_indices

    out_idx = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"frame_{os.path.splitext(video)[0]}_{out_idx:04d}.png"
        cv2.imwrite(os.path.join(OUT_DIR, fname), frame)
        out_idx += 1
    cap.release()()

print("Kadrų išskyrimas baigtas.")