import torch
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
import torch.nn as nn
import uvicorn
from PIL import Image
import io

class Rock_Scissors(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16*16, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
app = FastAPI()
model = Rock_Scissors()
model.load_state_dict(
    torch.load('rock_scissors.pth', map_location='cpu', weights_only=True)
)

model.eval()
classes = ['stone', 'scissors', 'paper']
бумага
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = outputs.argmax(dim=1).item()

    confidence = probabilities[0][prediction].item() * 100
    return {
        'prediction': classes[prediction],
        'class_index': prediction,
        'confidence': round(confidence, 2)
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
