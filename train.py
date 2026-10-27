import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from dataset import get_dataloaders
from datetime import datetime
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(model_name, num_classes):
    """Initializes pretrained models and modifies the final classification layer for GTSRB."""
    if model_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model.to(device)

def train_and_evaluate(model_name, epochs=5):
    print(f"\n{'='*50}\n[STARTING TRAINING] Model: {model_name}\n{'='*50}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results', f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, num_classes = get_dataloaders(batch_size=32)
    model = create_model(model_name, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training phase started...")
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")
        
    train_time = time.time() - start_time
    
    print("Validation phase started. Calculating metrics...")
    model.eval()
    all_preds = []
    all_labels = []
    
    infer_start_time = time.time()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    infer_time = time.time() - infer_start_time

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    results = {
        "Model": model_name,
        "Accuracy": round(acc * 100, 2),
        "Macro-F1": round(macro_f1 * 100, 2),
        "Training Time (sec)": round(train_time, 2),
        "Inference Time (sec)": round(infer_time, 2)
    }
    
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    np.savetxt(os.path.join(save_dir, "confusion_matrix.txt"), conf_matrix, fmt='%d')
    torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {results['Accuracy']}%")
    print(f"Macro-F1: {results['Macro-F1']}%")
    print(f"Training Time: {results['Training Time (sec)']} seconds")
    print(f"All files saved to: {save_dir}\n")
    
    return results

if __name__ == "__main__":
    
    
    models_to_train = ['resnet18', 'mobilenet_v3_large', 'convnext_tiny']
    
    if torch.cuda.is_available():
        print(f"GPU is available! Using: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: GPU not found. Using CPU.")
        
    all_results = {}
    for m in models_to_train:
        res = train_and_evaluate(m, epochs=5)
        all_results[m] = res
        
    print("\n" + "="*60)
    print("ALL MODELS COMPLETED (FINAL METRICS TABLE)")
    print("="*60)
    for m, res in all_results.items():
        print(f"{m:20} | Accuracy: %{res['Accuracy']:<5} | F1: %{res['Macro-F1']:<5} | Train Time: {res['Training Time (sec)']}s")
    print("="*60)