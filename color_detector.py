import cv2
import numpy as np
import argparse
from statistics import median

# --- Thresholds yang gampang di-tune ---
# HSV (OpenCV): H: 0-179, S:0-255, V:0-255
# Coklat ≈ oranye gelap: H ~ 10..25 (kadang melebar 5..30)
HSV_LOWER = np.array([8,  60,  20], dtype=np.uint8)   # batas bawah hue/sat/value
HSV_UPPER = np.array([30, 255, 255], dtype=np.uint8)  # batas atas hue/sat/value

# Klasifikasi berdasar L* (0..100), makin kecil = makin gelap
L_DARK_MAX   = 40   # <= 40 -> coklat tua
L_MID_MAX    = 65   # 40..65 -> coklat muda
# > 65 -> coklat terang

MIN_PIXELS_FOR_CONFIDENCE = 500  # minimal pixel coklat biar hasil valid

def classify_from_lab_l(l_values):
    """Terima list/array nilai Lightness (0..100), balikin label + skor."""
    if len(l_values) == 0:
        return "Tidak terdeteksi", 0.0, None

    # Lebih robust pakai median (anti noise)
    l_med = float(median(l_values))
    if l_med <= L_DARK_MAX:
        label = "Coklat Tua"
    elif l_med <= L_MID_MAX:
        label = "Coklat Muda"
    else:
        label = "Coklat Terang"

    # Confidence sederhana: seberapa rapat distribusinya
    l_arr = np.array(l_values, dtype=np.float32)
    iqr = np.percentile(l_arr, 75) - np.percentile(l_arr, 25) + 1e-6
    spread_score = np.clip(1.0 - (iqr / 40.0), 0.0, 1.0)  # 0..1
    # Tambah bobot dari ukuran sampel
    size_score = np.clip(len(l_values) / 5000.0, 0.0, 1.0)
    confidence = 0.6 * spread_score + 0.4 * size_score

    return label, confidence, l_med

def analyze_frame(bgr):
    """Balikin (label, confidence, l_med, vis) dari 1 frame BGR."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Mask area coklat di HSV
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # Bersihin noise dikit
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Ambil L* dari Lab (0..255 di OpenCV), konversi ke 0..100
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)

    L_masked = L[mask > 0]

    if L_masked.size < MIN_PIXELS_FOR_CONFIDENCE:
        label, conf, l_med = "Tidak terdeteksi", 0.0, None
    else:
        label, conf, l_med = classify_from_lab_l(L_masked.tolist())

    # Visual overlay
    vis = bgr.copy()
    overlay = vis.copy()
    overlay[mask > 0] = (0, 255, 0)  # tandai area coklat
    vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0)

    # Teks info
    h, w = vis.shape[:2]
    panel_h = 48
    cv2.rectangle(vis, (0, 0), (w, panel_h), (0, 0, 0), -1)
    text = f"{label} | conf={conf:.2f}"
    if l_med is not None:
        text += f" | L*~{l_med:.1f}"
    cv2.putText(vis, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return label, conf, l_med, vis

def run_webcam(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Webcam tidak bisa dibuka. Coba --camera-index lain atau cek perizinan kamera.")

    print("[INFO] Tekan 'q' untuk keluar. Tekan 's' untuk snapshot hasil.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        label, conf, l_med, vis = analyze_frame(frame)
        cv2.imshow("Brown Classifier (Webcam)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite("snapshot_result.jpg", vis)
            print("[INFO] Snapshot disimpan: snapshot_result.jpg")
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def run_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
    label, conf, l_med, vis = analyze_frame(img)
    print(f"Hasil: {label} (conf={conf:.2f})", f"| L* median={l_med:.1f}" if l_med is not None else "")
    cv2.imshow("Brown Classifier (Image)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser(description="Klasifikasi coklat tua/muda/terang (OpenCV)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--webcam", action="store_true", help="Pakai webcam")
    group.add_argument("--image", type=str, help="Path gambar input")
    ap.add_argument("--camera-index", type=int, default=0, help="Index kamera (default 0)")
    args = ap.parse_args()

    if args.webcam:
        run_webcam(args.camera_index)
    else:
        run_image(args.image)

if __name__ == "__main__":
    main()
