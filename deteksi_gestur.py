import cv2
import mediapipe as mp
import pyautogui
import time

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Variabel untuk Anti-Flicker
last_stable_action = "NETRAL" 
action_candidate = "NETRAL"
confirm_counter = 0
FRAMES_TO_CONFIRM = 5 

# Fungsi untuk mendeteksi gestur
def get_gesture_from_finger_count(hand_landmarks):
    landmarks = hand_landmarks.landmark
    
    fingers_up = []
    if landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
        fingers_up.append(1)
    if landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
        fingers_up.append(1)
    if landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y:
        fingers_up.append(1)
    if landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y:
        fingers_up.append(1)
        
    total_fingers = len(fingers_up)
    
    # --- PERUBAHAN ADA DI SINI ---
    if total_fingers == 1:
        return "KIRI" # DIUBAH
    elif total_fingers == 2:
        return "KANAN" # DIUBAH
    elif total_fingers == 3:
        return "LOMPAT"
    elif total_fingers == 4:
        return "BERGULING"
    else:
        return "NETRAL"

# Memulai kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

print("Program dimulai... Bersiap-siap dalam 3 detik.")
time.sleep(3)
print("OKE, MULAI!")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    current_detected_action = "NETRAL"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            current_detected_action = get_gesture_from_finger_count(hand_landmarks)

    # Logika Anti-Flicker
    if current_detected_action == action_candidate:
        confirm_counter += 1
    else:
        action_candidate = current_detected_action
        confirm_counter = 0

    if confirm_counter > FRAMES_TO_CONFIRM:
        if action_candidate != last_stable_action and action_candidate != "NETRAL":
            if action_candidate == "KANAN":
                pyautogui.press('right')
            elif action_candidate == "KIRI":
                pyautogui.press('left')
            elif action_candidate == "LOMPAT":
                pyautogui.press('up')
            elif action_candidate == "BERGULING":
                pyautogui.press('down')
            
            last_stable_action = action_candidate
        
        elif action_candidate == "NETRAL":
            last_stable_action = "NETRAL"

    cv2.putText(image, action_candidate, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
    
    cv2.imshow('Kontrol Game Anti-Flicker', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()