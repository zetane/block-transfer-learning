import os
import torch
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import requests

# Download ImageNet pre-trained model
model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
model_path = 'resnet50_imagenet.pt'

if not os.path.exists(model_path):
    print("Downloading ImageNet pre-trained model...")
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    print("Download complete.")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations for the input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load datasets
test_dataset = datasets.ImageFolder(root='test_set', transform=transform)

# Data loaders
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# Load the model
model = models.resnet50()
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Load ImageNet class names
imagenet_classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(imagenet_classes_url)
imagenet_classes = response.text.strip().split('\n')

# Evaluation function
def compute():
    """Evaluate the model and return predictions in the specified format."""

    result = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, _ = data
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            original_image_path = test_dataset.imgs[i][0]
            original_image_name = os.path.basename(original_image_path)
            original_image = Image.open(original_image_path).convert("RGB")

            # Draw the predicted labels on the image
            draw = ImageDraw.Draw(original_image)
            font = ImageFont.load_default()
            predicted_label = imagenet_classes[predicted.item()]
            text = f"Pred: {predicted_label}"
            draw.text((10, 10), text, fill="red", font=font)

            # Save the image
            save_path = f"output/{original_image_name}"
            os.makedirs("output", exist_ok=True)
            original_image.save(save_path)

            result.append(save_path)

    return {"result": result}

def test():
    """Test the compute function."""

    print("Running test")
