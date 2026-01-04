import torch
import torch.nn as nn
from torchvision import models,transforms
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features,2)
model.load_state_dict(torch.load("deepfake_detector.pth",weights_only=True))
model.to(device).eval()

#efficientnet k liye target layer
target_layers = [model.features[-1]]

#2 image processing
def get_cam_result(img_path):
    img = cv2.imread(img_path)[:,:,::-1]
    img_float = np.float32(img)/255
    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])(img).unsqueeze(0).to(device)

    cam = GradCAM(model=model,target_layers=target_layers)
    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=input_tensor,targets=targets)[0,:]

    visualization = show_cam_on_image(img_float,grayscale_cam,use_rgb=True)
    cv2.imwrite('gradcam_output.jpg',cv2.cvtColor(visualization,cv2.COLOR_RGB2BGR))
    print("Grad-CAM saved as gradcam_output.jpg")
    get_cam_result('./processed_data/fake/Deepfakes_000_003.mp4_0.jpg')
