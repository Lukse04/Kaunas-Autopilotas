import os
import sys
import glob
import cv2
import torch
import numpy as np
import time
import threading
import queue
import logging

import torchvision.transforms as T
from torch.cuda.amp import autocast
from segmentation_models_pytorch import DeepLabV3Plus

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
BATCH_SIZE        = 2          # Pradinis batch size; pakoreguok pagal savo VRAM
DOWNSCALE_FACTOR  = 1          # 1 = originali rezoliucija; 2 = pusė, 4 = ketvirtadalis
QUEUE_SIZE        = 32         # kiek kadrų iš anksto pakrauti
LOG_FILE          = 'process_video.log'

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Paths & Device
# -----------------------------------------------------------------------------
tmp_dir     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if tmp_dir not in sys.path:
    sys.path.insert(0, tmp_dir)
BASE_DIR    = tmp_dir
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_path= os.path.join(BASE_DIR, 'best_deeplab.pth')
input_dir   = os.path.join(BASE_DIR, 'data', 'raw')
output_dir  = os.path.join(BASE_DIR, 'predictions', 'output')
threshold   = 0.5

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def format_time(seconds):
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        m = s // 60
        s2 = s % 60
        return f"{m}min {s2}s"
    else:
        h = s // 3600
        m = (s % 3600) // 60
        return f"{h}h {m}min"

def load_model(path, device):
    model = DeepLabV3Plus(
        encoder_name    = 'resnet50',
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1
    )
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()

def pad_to_divisible(frame, d=16):
    h, w = frame.shape[:2]
    ph = (d - h % d) % d
    pw = (d - w % d) % d
    f  = cv2.copyMakeBorder(frame, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(0,0,0))
    return f, h, w


def detect_and_draw_lines(frame, mask_binary):
    # Canny edge detection on mask
    edges = cv2.Canny(mask_binary, 50, 150)
    # Hough line transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                            minLineLength=40, maxLineGap=20)
    detected = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            detected.append((x1, y1, x2, y2))
    return frame, detected

# -----------------------------------------------------------------------------
# Frame reader thread
# -----------------------------------------------------------------------------
def frame_reader(cap, q):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        q.put(frame)
    q.put(None)

# -----------------------------------------------------------------------------
# Batch inference + draw & log
# -----------------------------------------------------------------------------
def process_batch(batch_frames, batch_meta, model, device, writer, frame_idx_start):
    inp = torch.stack(batch_frames).to(device)
    with torch.no_grad(), autocast():
        masks = model(inp)            # [B,1,H,W]
        probs = torch.sigmoid(masks)  # [B,1,H,W]

    for i in range(len(batch_frames)):
        h0, w0, frame = batch_meta[i]
        prob = probs[i,0,:h0,:w0].cpu().numpy()
        mask_np = (prob > threshold).astype(np.uint8) * 255
        # Overlay mask
        m_bgr = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(frame, 0.7, m_bgr, 0.3, 0)
        # Detect lines
        frame_with_lines, lines = detect_and_draw_lines(overlay, mask_np)
        # Log detected lines
        frame_num = frame_idx_start + i + 1
        if lines:
            for ln in lines:
                logger.info(f"Frame {frame_num}: linija {ln}")
        else:
            logger.info(f"Frame {frame_num}: nenustatytos linijos")
        writer.write(frame_with_lines)

# -----------------------------------------------------------------------------
# Single-video processing
# -----------------------------------------------------------------------------
def process_single_video(in_path, out_path, model, device):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"❌ Nepavyko atidaryti {in_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps   = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  // DOWNSCALE_FACTOR
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // DOWNSCALE_FACTOR
    duration = total / fps if fps > 0 else 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))

    logger.info(f"Pradėtas apdorojimas: {os.path.basename(in_path)}, total frames: {total}")
    # start reader
    q = queue.Queue(maxsize=QUEUE_SIZE)
    t = threading.Thread(target=frame_reader, args=(cap, q), daemon=True)
    t.start()

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std =[0.229,0.224,0.225]),
    ])

    batch_frames = []
    batch_meta   = []
    idx = 0
    start = time.time()

    while True:
        frame = q.get()
        if frame is None:
            # galutinis batch
            if batch_frames:
                process_batch(batch_frames, batch_meta, model, device, writer, idx)
                idx += len(batch_frames)
            break

        if DOWNSCALE_FACTOR != 1:
            frame = cv2.resize(frame, (width, height),
                               interpolation=cv2.INTER_AREA)

        padded, h0, w0 = pad_to_divisible(frame)
        batch_frames.append(transform(padded))
        batch_meta.append((h0, w0, frame))

        if len(batch_frames) == BATCH_SIZE:
            process_batch(batch_frames, batch_meta, model, device, writer, idx)
            idx += BATCH_SIZE
            batch_frames.clear()
            batch_meta.clear()

        elapsed = time.time() - start
        eta     = (elapsed / max(1, idx)) * (total - idx)
        percent = idx / total * 100
        avg_ms  = elapsed / max(1, idx) * 1000
        fps_val = idx / elapsed if elapsed > 0 else 0

        print(f"\r{os.path.basename(in_path)}: "
              f"{percent:5.1f}% ({idx}/{total}) | "
              f"Praėjo: {format_time(elapsed)} | "
              f"Likę: ~{format_time(eta)} | "
              f"MS: {avg_ms:.1f} | FPS: {fps_val:.1f} | "
              f"Trukmė: {format_time(duration)}", end='', flush=True)

    cap.release()
    writer.release()
    total_t = time.time() - start
    logger.info(f"Baigta apdorojimas per {format_time(total_t)}")
    print(f"\n✅ {os.path.basename(in_path)} baigėsi per {format_time(total_t)}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"🔌 Įkraunam modelį ant {device} iš {weights_path}")
    model = load_model(weights_path, device)

    os.makedirs(output_dir, exist_ok=True)
    for path in glob.glob(os.path.join(input_dir, '*')):
        if not path.lower().endswith(('.mp4','.avi','.mov','.mkv')):
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        outp = os.path.join(output_dir, name + '.mp4')
        process_single_video(path, outp, model, device)

    print("🏁 Visi video apdoroti.")
