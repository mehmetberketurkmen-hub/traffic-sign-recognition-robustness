import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots():
    
    phase2_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase2_evaluation')
    csv_path = os.path.join(phase2_dir, 'corruption_metrics.csv')
    
    if not os.path.exists(csv_path):
        print("Error: corruption_metrics.csv not found! Please run test_corruptions.py first.")
        return

    
    df = pd.read_csv(csv_path)
    
   
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    corruptions = ['gaussian_noise', 'motion_blur', 'fog', 'brightness']
    titles = ['Gaussian Noise', 'Motion Blur', 'Fog', 'Brightness']
    
    for i, (corr, title) in enumerate(zip(corruptions, titles)):
        subset = df[df['Corruption'] == corr]
        
        
        sns.lineplot(
            data=subset, x='Severity', y='Accuracy', hue='Model', 
            style='Model', markers=True, dashes=False, linewidth=2.5, markersize=8, ax=axes[i]
        )
        
        axes[i].set_title(f'Accuracy vs. Severity: {title}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Severity Level (1-5)', fontsize=12)
        axes[i].set_ylabel('Accuracy (%)', fontsize=12)
        axes[i].set_xticks([1, 2, 3, 4, 5])
        axes[i].set_ylim(0, 105) 
        axes[i].legend(title='Model Architecture')

    plt.tight_layout()
    
    
    save_path = os.path.join(phase2_dir, 'performance_plots.png')
    plt.savefig(save_path, dpi=300)
    print(f"Success! Plot generated and saved to: {save_path}")

if __name__ == "__main__":
    generate_plots()