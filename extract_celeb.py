import cv2
import os
import mediapipe as mp
from tqdm import tqdm

# --- CONFIG (Inhe ek baar dhyan se dekho) ---
CELEB_DIR = './Celeb-real'  
REAL_DIR = './processed_data/real'
TARGET_COUNT = 1808 

# Check if folder exists
if not os.path.exists(CELEB_DIR):
    print(f"❌ ERROR: '{CELEB_DIR}' folder nahi mila! Check karo ki unzip kahan kiya hai.")
    exit()

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1)

def start_fixing():
    current_files = [f for f in os.listdir(REAL_DIR) if f.endswith('.jpg')]
    current_count = len(current_files)
    needed = TARGET_COUNT - current_count
    
    if needed <= 0:
        print(f"✅ Already balanced! Total: {current_count}")
        return

    videos = [f for f in os.listdir(CELEB_DIR) if f.endswith('.mp4')]
    
    if len(videos) == 0:
        print(f"❌ ERROR: '{CELEB_DIR}' ke andar koi .mp4 videos nahi mili!")
        return

    print(f"🚀 Found {len(videos)} videos. Extracting {needed} faces...")
    
    count = 0
    pbar = tqdm(total=needed)
    
    for vid in videos:
        if count >= needed: break
        cap = cv2.VideoCapture(os.path.join(CELEB_DIR, vid))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Har video se 5 frames uthayenge
        for pos in [0.2, 0.4, 0.6, 0.8]:
            if count >= needed: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * pos))
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)
                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bbox.xmin*iw), int(bbox.ymin*ih), int(bbox.width*iw), int(bbox.height*ih)
                    
                    face = frame[max(0,y):y+h, max(0,x):x+w]
                    if face.size > 0:
                        face = cv2.resize(face, (224, 224))
                        # Filename check: celeb_ prefix zaroori hai
                        cv2.imwrite(os.path.join(REAL_DIR, f"celeb_fix_{vid}_{pos}.jpg"), face)
                        count += 1
                        pbar.update(1)
        cap.release()
    pbar.close()
    print(f"🎉 Success! Real folder now has {current_count + count} images.")

if __name__ == "__main__":
    start_fixing()