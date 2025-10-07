import cv2
import numpy as np

lower_brown = np.array([8,  60,  20])
upper_brown = np.array([30, 255, 255])

def classify_lightness(l):
    if l <= 40:
        return "Coklat Tua"
    elif l <= 65:
        return "Coklat Muda"
    else:
        return "Coklat Terang"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam tidak terdeteksi!")

# print("Tekan 'q' buat keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)
    L_masked = L[mask > 0]

    label = "Tidak terdeteksi"
    if L_masked.size > 0:
        L_med = np.median(L_masked)
        label = classify_lightness(L_med)

    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 0)
    vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    cv2.putText(vis, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.imshow("Deteksi Warna Coklat", vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
