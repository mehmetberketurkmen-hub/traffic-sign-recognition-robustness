import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from imagecorruptions import corrupt

PROJECT_DIR = r"C:\Users\Berke\OneDrive\Masaüstü\Bitirme_Projesi"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes=43):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def find_weight_file(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if 'mobilenet' in os.path.basename(root) and 'model_weights.pth' in files:
            return os.path.join(root, 'model_weights.pth')
    raise FileNotFoundError("Could not find model_weights.pth for mobilenet_v3_large")

class GaussianNoiseTransform:
    def __init__(self, severity):
        self.severity = severity
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        
    def __call__(self, img):
        img = self.resize(img)
        img_np = np.array(img)
        corrupted_np = corrupt(img_np, corruption_name='gaussian_noise', severity=self.severity)
        return self.to_tensor(corrupted_np)

def main():
    print(f"Device: {DEVICE}")
    model = get_model().to(DEVICE)
    weight_path = find_weight_file(os.path.join(PROJECT_DIR, 'results'))
    print(f"Found weights at: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.eval()

    severities = [1, 3, 5]
    results_dir = os.path.join(PROJECT_DIR, 'results', 'phase2_evaluation')
    os.makedirs(results_dir, exist_ok=True)

    for sev in severities:
        print(f"\nEvaluating MobileNetV3 on Gaussian Noise - Severity {sev}...")
        
        transform = GaussianNoiseTransform(severity=sev)
        test_dataset = datasets.GTSRB(root=DATA_DIR, split='test', transform=transform, download=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(cm, cmap='Oranges', xticklabels=False, yticklabels=False)
        plt.title(f'MobileNetV3 Confusion Matrix - Gaussian Noise Severity {sev}')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        
        save_path = os.path.join(results_dir, f'mobilenet_cm_noise_sev{sev}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix plot to: {save_path}")

if __name__ == "__main__":
    main()