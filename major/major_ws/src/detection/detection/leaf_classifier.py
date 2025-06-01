import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

class LeafDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(LeafDiseaseClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def load_classification_model(model_path, num_classes=2):

    model = LeafDiseaseClassifier(num_classes)
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading classification model: {str(e)}")
        return None

def preprocess_image(image):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def classify_leaf(cropped_image, model):
    
    if model is None or cropped_image is None or cropped_image.size == 0:
        return None, None
        
    try:
        # Check image dimensions
        height, width = cropped_image.shape[:2]
        if height < 32 or width < 32:  # Minimum size check
            print(f"Warning: Cropped image too small ({width}x{height})")
            return None, None

        enhanced = cropped_image
            
        # Convert BGR to RGB and then to PIL Image
        pil_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        # Add padding to maintain aspect ratio
        target_size = 224
        ratio = float(target_size) / max(pil_img.size)
        new_size = tuple([int(x * ratio) for x in pil_img.size])
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        new_img.paste(pil_img, ((target_size - new_size[0]) // 2,
                               (target_size - new_size[1]) // 2))
        
        # Preprocess the image
        input_tensor = preprocess_image(new_img)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probs = nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            return predicted.item(), confidence.item()
            
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return None, None

def get_classification_label(prediction, confidence):
    """Convert prediction and confidence to a human-readable label."""
    if prediction is None or confidence is None:
        return "Unknown"
    status = "Healthy" if prediction == 0 else "Unhealthy"
    return f"{status} ({confidence:.2f})" 