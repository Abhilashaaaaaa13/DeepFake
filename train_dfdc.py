import os
import json
import cv2
import mediapipe as mp
from tqdm import tqdm

#configuration
DFDC_DIR='./train_sample_videos'
METADATA_PATH=os.path.join(DFDC_DIR,'metadata.json')
OUTPUT_DIR='./processed_data'

#mediapipe face detection
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1,min_detection_confidence=0.5)

def extract_face(frame):
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        ih,iw,_ = frame.shape
        x,y,w,h = int(bbox.xmin*iw),int(bbox.ymin*ih),int(bbox.width*iw),int(bbox.height*ih)
        face = frame[max(0,y):y+h,max(0,x):x+w]
        if face.size>0:
            return cv2.resize(face,(224,224))
    return None

#processing
with open(METADATA_PATH,'r') as f:
    metadata = json.load(f)

os.makedirs(os.path.join(OUTPUT_DIR,'real'),exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR,'fake'),exist_ok=True)

#count check krne k liye taki dataset balance rhe
counts = {'real':0,'fake':0}
LIMIT_PER_CLASS = 400

print("Starting DFDC Balanced Extraction")

for filename,info in tqdm(metadata.items()):
    label = info['label'].lower()  #real or fake

    #check if we reached limit of this class
    if counts[label] >= LIMIT_PER_CLASS:
        continue

    video_path = os.path.join(DFDC_DIR,filename)
    if not os.path.exists(video_path):
        continue

    cap = cv2.VideoCapture(video_path)
    ret,frame = cap.read()  #pehla frame process krnege
    if ret:
        face = extract_face(frame)
        if face is not None:
            save_name = f"dfdc_{filename}_{counts[label]}.jpg"
            save_path = os.path.join(OUTPUT_DIR,label,save_name)
            cv2.imwrite(save_path,face)
            counts[label] += 1
    cap.release()

print(f"DFDC Processing Complete! Added: {counts['real']} Real, {counts['fake']} Fake")