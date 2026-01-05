import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import mediapipe as mp

# --- CONFIG ---
MODEL_PATH = "deepfake_image_model.pth"
VIDEO_PATH = "00005.mp4"  # <-- Apni video ka file name yahan badlo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FACE DETECTION SETUP ---
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# --- LOAD MODEL ---
def load_deepfake_model(path):
    backbone = models.efficientnet_b0()
    num_feats = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    
    classifier = nn.Sequential(
        nn.BatchNorm1d(num_feats),
        nn.Linear(num_feats, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    
    checkpoint = torch.load(path, map_location=DEVICE)
    backbone.load_state_dict(checkpoint['backbone'])
    classifier.load_state_dict(checkpoint['classifier'])
    
    return backbone.to(DEVICE).eval(), classifier.to(DEVICE).eval()

backbone, classifier = load_deepfake_model(MODEL_PATH)

# --- PREPROCESSING ---
val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def get_face_crop(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bbox.xmin*iw), int(bbox.ymin*ih), int(bbox.width*iw), int(bbox.height*ih)
        face = frame[max(0,y):y+h, max(0,x):x+w]
        if face.size > 0:
            return face
    return None

# --- TESTING LOGIC ---
def run_test(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logits_list = []
    
    print(f"🎬 Processing video: {video_path}...")
    
    # Video se 20 frames uthayenge patterns check karne ke liye
    sampled_count = 0
    for i in range(20):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total_frames / 20))
        ret, frame = cap.read()
        if not ret: break
        
        face = get_face_crop(frame)
        if face is not None:
            face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            img_t = val_tfms(face_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = classifier(backbone(img_t))
                logits_list.append(out)
                sampled_count += 1

    cap.release()
    
    if len(logits_list) == 0:
        print("❌ Error: Video mein koi chehra nahi mila!")
        return

    # Average logits and calculate probability
    avg_logits = torch.cat(logits_list, dim=0).mean(dim=0)
    probs = torch.softmax(avg_logits / 1.5, dim=0) # Temperature scaling for stability
    
    real_p = probs[0].item() * 100
    fake_p = probs[1].item() * 100

    print("\n" + "="*30)
    print(f"🔍 FINAL REPORT ({sampled_count} frames analyzed)")
    print("="*30)
    print(f"🟢 REAL Confidence: {real_p:.2f}%")
    print(f"🔴 FAKE Confidence: {fake_p:.2f}%")
    print("-" * 30)
    
    if real_p > fake_p:
        print("✅ VERDICT: ORIGINAL VIDEO")
    else:
        print("🚨 VERDICT: DEEPFAKE DETECTED!")
    print("="*30 + "\n")

if __name__ == "__main__":
    run_test(VIDEO_PATH)