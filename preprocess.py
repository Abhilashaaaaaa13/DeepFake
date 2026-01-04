import cv2
import os
import mediapipe as mp
from tqdm import tqdm

#Configurationss
INPUT_ROOT = './FaceForensics++_C23'
OUTPUT_ROOT = './processed_data'

#sabhi manipulation types
CATEGORIES = {
    'real':['original','DeepFakeDetection'],
    'fake':['Deepfakes','Face2Face','FaceSwap','NeuralTextures','FaceShifter']

}

#mediapipe face detection
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1,min_detection_confidence=0.5)

def save_face(frame,save_path):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)
    if results.detections:
       bbox = results.detections[0].location_data.relative_bounding_box
       ih, iw, _ = frame.shape
       #coordinate nikalana
       x, y, w, h = int(bbox.xmin*iw), int(bbox.ymin*ih), int(bbox.width*iw), int(bbox.height*ih)
        #crop with safety boundaries
       face = frame[max(0,y):y+h,max(0,x):x+w]
       if face.size > 0:
           face = cv2.resize(face,(224,224))
           cv2.imwrite(save_path,face)
           return True
       
    return False
    
   

def start_preprocessing():
    print("starting phase 1: extractig fac chips...")
    for label, folders in CATEGORIES.items():
        save_path = os.path.join(OUTPUT_ROOT,label)
        os.makedirs(save_path,exist_ok=True)

        for folder in folders:
            folder_path = os.path.join(INPUT_ROOT,folder)
            if not os.path.exists(folder_path):
                print(f"Warning : Folder {folder} not found at {folder_path}")
                continue
            #har category s 30 video sample krenge
            videos = [v for v in os.listdir(folder_path) if v.endswith('.mp4')][:30]
            for vid in tqdm(videos, desc=f"Processing {folder}"):
                cap = cv2.VideoCapture(os.path.join(folder_path,vid))
                saved, frame_no = 0,0 

                while saved< 10 :
                    ret, frame = cap.read()
                    if not ret: break

               

                    #har 15th frame skip
                    if frame_no % 15 == 0:
                        name = f"{folder}_{vid}_{saved}.jpg"
                        if save_face(frame, os.path.join(save_path,name)):
                            saved += 1
                    frame_no += 1
                cap.release()

if __name__ == "__main__":
    start_preprocessing()
    print("\n phase 1 sucess")
