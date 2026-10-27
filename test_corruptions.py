import os
import glob
import torch
import torch.nn as nn
import numpy as np

# --- YAZILIMCI HİLESİ (MONKEY PATCH) ---
# Eski kütüphanelerin NumPy 2.0 ile çökmesini engeller
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
# ---------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from imagecorruptions import corrupt
from tqdm import tqdm
from PIL import Image
# ---------------------------------------------------------
# 1. AYARLAR VE SABİTLER
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORRUPTIONS = ['gaussian_noise', 'motion_blur', 'fog', 'brightness']
SEVERITIES = [1, 2, 3, 4, 5]
MODELS_TO_TEST = ['resnet18', 'mobilenet_v3_large', 'convnext_tiny']
NUM_CLASSES = 43
BATCH_SIZE = 64

# Sonuçların kaydedileceği ana klasör
PHASE2_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase2_evaluation')
os.makedirs(PHASE2_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. VERİ SETİ VE CORRUPTION (BOZULMA) ADAPTÖRÜ
# ---------------------------------------------------------
class CorruptedGTSRB(Dataset):
    """Resimleri yüklerken on the fly (anında) bozulma ekleyen özel sınıf"""
    def __init__(self, root, split='test', corruption_name=None, severity=1):
        self.base_dataset = GTSRB(root=root, split=split, download=False)
        self.corruption_name = corruption_name
        self.severity = severity
        
        # Bozulmadan önce tüm resimler aynı boyuta getirilmeli (Gürültü piksel boyutu için kritik)
        self.resize = transforms.Resize((224, 224))
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = self.resize(img)
        
        # Eğer bir bozulma (corruption) isteniyorsa, numpy'a çevirip uygula
        if self.corruption_name:
            img_np = np.array(img)
            corrupted_np = corrupt(img_np, corruption_name=self.corruption_name, severity=self.severity)
            img = Image.fromarray(corrupted_np)
            
        img_tensor = self.to_tensor_norm(img)
        return img_tensor, label, idx

# ---------------------------------------------------------
# 3. YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------
def create_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def find_latest_weights(model_name):
    """Klasördeki en son eğitilmiş model ağırlığını (.pth) otomatik bulur."""
    search_pattern = os.path.join(os.path.dirname(__file__), '..', 'results', f"{model_name}_*", "model_weights.pth")
    matches = glob.glob(search_pattern)
    if not matches:
        raise FileNotFoundError(f"Model weights for {model_name} not found! Check 'results' folder.")
    return sorted(matches)[-1]

def generate_visual_examples():
    """Hocanın istediği 'Representative examples' görselini oluşturur."""
    print("\nGenerating visual examples of corruptions...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    base_dataset = GTSRB(root=data_path, split='test', download=False)
    
    # Veri setinden rastgele güzel bir trafik levhası seç (Örn: indeks 1000)
    orig_img, _ = base_dataset[1000]
    orig_img = orig_img.resize((224, 224))
    orig_np = np.array(orig_img)
    
    fig, axes = plt.subplots(len(CORRUPTIONS), len(SEVERITIES) + 1, figsize=(15, 10))
    
    for i, c_name in enumerate(CORRUPTIONS):
        axes[i, 0].imshow(orig_np)
        axes[i, 0].set_title("Clean Image")
        axes[i, 0].axis('off')
        
        for j, sev in enumerate(SEVERITIES):
            c_img = corrupt(orig_np, corruption_name=c_name, severity=sev)
            axes[i, j+1].imshow(c_img)
            axes[i, j+1].set_title(f"{c_name}\nSev: {sev}")
            axes[i, j+1].axis('off')
            
    plt.tight_layout()
    plot_path = os.path.join(PHASE2_DIR, "representative_examples.png")
    plt.savefig(plot_path)
    print(f"Visual examples saved to: {plot_path}")

# ---------------------------------------------------------
# 4. ANA TEST DÖNGÜSÜ
# ---------------------------------------------------------
def evaluate_all():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    all_results = []
    
    # Hocanın istediği Per-Image Predictions (Her resim için tahmin) verisini tutacak liste
    per_image_predictions = []

    print(f"\n{'='*50}\nSTARTING PHASE 2: CORRUPTION EVALUATION\n{'='*50}")
    
    for model_name in MODELS_TO_TEST:
        print(f"\nLoading {model_name}...")
        model = create_model(model_name, NUM_CLASSES)
        weight_path = find_latest_weights(model_name)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        for c_name in CORRUPTIONS:
            for sev in SEVERITIES:
                print(f"Testing [{model_name}] -> {c_name} (Severity {sev})")
                
                dataset = CorruptedGTSRB(root=data_path, split='test', corruption_name=c_name, severity=sev)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                
                all_preds = []
                all_labels = []
                all_indices = []
                
                with torch.no_grad():
                    for images, labels, indices in tqdm(loader, leave=False):
                        images, labels = images.to(DEVICE), labels.to(DEVICE)
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_indices.extend(indices.numpy())
                
                acc = accuracy_score(all_labels, all_preds)
                macro_f1 = f1_score(all_labels, all_preds, average='macro')
                
                # Tablo için ana sonuçları kaydet
                all_results.append({
                    "Model": model_name,
                    "Corruption": c_name,
                    "Severity": sev,
                    "Accuracy": round(acc * 100, 2),
                    "Macro-F1": round(macro_f1 * 100, 2)
                })
                
                # Hocanın istediği "per-image prediction" (Her bir resmin tek tek tahmini) kaydı
                for idx, t_label, p_label in zip(all_indices, all_labels, all_preds):
                    per_image_predictions.append({
                        "Model": model_name,
                        "Corruption": c_name,
                        "Severity": sev,
                        "Image_ID": idx,
                        "True_Label": t_label,
                        "Predicted_Label": p_label
                    })

    # 1. Ana Metrikleri CSV olarak kaydet
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(PHASE2_DIR, "corruption_metrics.csv"), index=False)
    
    # 2. Her resim için olan devasa tahmin verisini CSV olarak kaydet
    df_preds = pd.DataFrame(per_image_predictions)
    df_preds.to_csv(os.path.join(PHASE2_DIR, "per_image_predictions.csv"), index=False)
    
    print(f"\nAll evaluations complete! Results saved to {PHASE2_DIR}")
    return df_results

# ---------------------------------------------------------
# 5. ÇALIŞTIRMA
# ---------------------------------------------------------
if __name__ == "__main__":
    # 1. Hocanın görmek istediği resim örneklerini çiz
    generate_visual_examples()
    
    # 2. Bütün modelleri, bütün bozulma tiplerinde test et
    evaluate_all()