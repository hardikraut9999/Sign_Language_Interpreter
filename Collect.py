import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create dataset directory
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Get label from user
label = input("Enter label for this sign: ")  # Example: "Hello"
label_path = os.path.join(dataset_path, label)
if not os.path.exists(label_path):
    os.makedirs(label_path)

count = 0  # Image counter
max_images = 150  # Capture limit

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or count >= max_images:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract bounding box
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Extract hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                hand_img = cv2.resize(hand_img, (150, 150))  # Resize for consistency
                cv2.imwrite(os.path.join(label_path, f"{count}.jpg"), hand_img)
                count += 1
                print(f"Captured {count}/{max_images} images for label '{label}'")

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Dataset collection complete for label '{label}'")
