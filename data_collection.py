import cv2
import numpy as np
import os
import mediapipe as mp
import time

# Dataset klasörü oluştur
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# MediaPipe el modeli
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# J ve Z hariç ASL harfleri
alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'

# Kamera aç
cap = cv2.VideoCapture(0)

for letter in alphabet:
    if not os.path.exists(f'dataset/{letter}'):
        os.makedirs(f'dataset/{letter}')
    
    print(f"\n=== Letter: '{letter}' ===")
    print("El hareketini göster ve 's' ile başla, 'q' ile harfi geç")

    image_count = 0
    capture_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        h, w, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Arayüz metinleri
        cv2.putText(frame, f"Letter: {letter} | Count: {image_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if capture_active:
            cv2.putText(frame, "Capturing... 'q' for next letter", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press 's' to start capturing", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow('ASL Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            capture_active = True
            print("Capturing started...")

        if capture_active and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # ROI hesapla
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                xmin, xmax = int(min(xs) * w), int(max(xs) * w)
                ymin, ymax = int(min(ys) * h), int(max(ys) * h)

                # Biraz boşluk ekle (padding)
                pad = 20
                xmin = max(0, xmin - pad)
                ymin = max(0, ymin - pad)
                xmax = min(w, xmax + pad)
                ymax = min(h, ymax + pad)

                # ROI al ve boyutlandır
                hand_crop = frame[ymin:ymax, xmin:xmax]
                hand_crop = cv2.resize(hand_crop, (128, 128))

                image_path = f'dataset/{letter}/{letter}_{image_count}.jpg'
                cv2.imwrite(image_path, hand_crop)
                image_count += 1
                time.sleep(0.2)
                break  # sadece bir el

        if key == ord('q'):
            print(f"Captured {image_count} images for letter {letter}")
            break

        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
print("✅ Veri toplama tamamlandı!")
