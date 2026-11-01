import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os

PROJECT_DIR = r"C:\Users\Berke\OneDrive\Masaüstü\Bitirme_Projesi"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name, num_classes=43):
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError("Invalid model name")
    return model

def find_weight_file(base_dir, model_name):
    for root, dirs, files in os.walk(base_dir):
        if model_name in os.path.basename(root) and 'model_weights.pth' in files:
            return os.path.join(root, 'model_weights.pth')
    raise FileNotFoundError(f"Could not find model_weights.pth for {model_name} in {base_dir}")

def main():
    print(f"Device: {DEVICE}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    test_dataset = datasets.GTSRB(
        root=DATA_DIR, 
        split='test', 
        transform=transform, 
        download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model_names = ['resnet18', 'mobilenet_v3_large', 'convnext_tiny']
    results_list = []
    
    for model_name in model_names:
        print(f"\nEvaluating: {model_name} (Clean Data)")
        model = get_model(model_name).to(DEVICE)
        
        weight_path = find_weight_file(os.path.join(PROJECT_DIR, 'results'), model_name)
        print(f"Found weights at: {weight_path}")
        
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='macro') * 100
        
        print(f"Accuracy: {acc:.2f}% | Macro-F1: {f1:.2f}%")
        results_list.append({'Model': model_name, 'Clean_Accuracy': round(acc, 2), 'Clean_Macro_F1': round(f1, 2)})
        
    results_dir = os.path.join(PROJECT_DIR, 'results', 'phase2_evaluation')
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(results_dir, 'clean_test_metrics.csv'), index=False)
    print("\nClean test results saved to: clean_test_metrics.csv")

if __name__ == "__main__":
    main()