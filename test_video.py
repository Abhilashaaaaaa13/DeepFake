import cv2
import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import Image
import numpy as np

#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features,2)
model.load_state_dict(torch.load("deepfake_detector.pth",weights_only=True))
model.to(device).eval()

#2 transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def test_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_scores=[]
    count =0 
    print(f"analyzing video: {video_path}")

    while cap.isOpened() and len(frame_scores)<20:
        ret, frame = cap.read()
        if not ret: break

        #har 10th frame checkkro speed k liye
        if count % 10 ==0:
            #simple preprocess(mediapipe yha bhi lga skte for better accuracy)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output,dim=1)
                #fake hone ki probabilitu
                fake_prob = probs[0][1].item()
                frame_scores.append(fake_prob)

        count += 1
    cap.release()
    if not frame_scores:
        return "Error: Could not process video"
    
    final_score = np.mean(frame_scores)
    result = "🔴 FAKE" if final_score > 0.5 else "🟢 REAL"
    
    print(f"--- Final Result ---")
    print(f"Prediction: {result}")
    print(f"Confidence: {final_score:.2%}")

test_video('Deepfake_Video_Generation_For_Model_Testing.mp4')