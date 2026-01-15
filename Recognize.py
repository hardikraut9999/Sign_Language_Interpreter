import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import mediapipe as mp
import os
from PIL import Image  # Import PIL for image conversion

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model (must match the trained architecture)
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 37 * 37, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Get class labels from dataset directory
class_names = sorted(os.listdir("dataset"))
num_classes = len(class_names)
print(f"Recognizing signs for: {class_names}")

# Load the trained model
model = SignLanguageModel(num_classes).to(device)
model_path = "sign_language_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set to evaluation mode (FIXED)

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Resize to match model input
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize (adjust if needed)
])

# Function to process video frames
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box around the hand
            h, w, _ = frame.shape
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
            x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
            y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)

            # Ensure valid crop
            if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
                continue

            # Crop the hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Convert NumPy OpenCV image to PIL Image for PyTorch processing
            hand_img = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))

            # Apply transformations
            hand_img = transform(hand_img).unsqueeze(0).to(device)  # Add batch dimension

            # Make predictions
            with torch.no_grad():
                outputs = model(hand_img)
                class_id = torch.argmax(outputs, dim=1).item()

            word_detected = class_names[class_id]

            # Display the prediction on the frame
            cv2.putText(frame, f'Word: {word_detected}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Main function to capture video
def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow('Sign Language Recognition', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
