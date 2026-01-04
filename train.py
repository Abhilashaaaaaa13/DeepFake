import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models,transforms
from torch.utils.data import DataLoader
import os

#settings n gpu check
device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#data transforms
#images ko model k layak banana
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

])
#load dataset
data_dir = './processed_data'
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

#train or validation m 80-20 rules
train_size = int(0.8*len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,[train_size,val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=False)

#build the model(efficientnet)
model = models.efficientnet_b0(weights='DEFAULT')
#last layer ko 2 classes (real bs fake) k liye change krnaa
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs,2)
model = model.to(device)

#loss n optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

#training loop
epochs = 10
print("starting training")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs,labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss:{running_loss/len(train_loader):.4f}")

#save model
torch.save(model.state_dict(),"deepfake_detector.pth")
print("Model saved as deepfake_detector.pth")