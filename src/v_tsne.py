import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from tqdm import tqdm


def tsne_image(model, val_loader, device, class_names, base_dir, perplexity = 30):
    print("\n")
    print("\n")
    
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="[Extracting Latent Features]")
        for images, labels in val_bar:
            images = images.to(device)
            
            _, _, proj_out = model(images)
            
            #將特徵從GPU移至CPU並轉為NumPy陣列
            all_features.append(proj_out.cpu().numpy())
            all_labels.extend(labels.numpys())
    
    #合併所有Batch的特徵        
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    
    print(f"共 {all_features.shape[0]} 筆樣本")
    
    tsne = TSNE(
        n_components = 2,
        perplexity = perplexity,
        n_iter = 1500,
        random_state = 42,
        init = 'pca',
        learning_rate = 'auto'
    )
    tsne_coords = tsne.fit_transform(all_features)
    
    plt.figure(figsize = (12, 10))
    
    sns.scatterplot(
        x = tsne_coords[:, 0],
        y = tsne_coords[:, 1],
        hue = [class_names[i] for i in all_labels],
        hur_order = class_names,
        palette = "tab10",
        s = 70,
        alpha = 0.85,
        edgcolor = 'none'
    )
    
    plt.title("t-SNE / UMAP Latent Space Clusters", fontsize = 16, fontewight = 'bold', pad = 15)
    plt.xlabel("t-SNE Dimension 1", fontsize = 12)
    plt.ylabel("t-SNE Dimension 2", fontsize = 12)
    plt.legend(title = "Cinical Grade", bbox_to_anchor = (1.02, 1), loc = 'upper left', borderaxepad = 0, fontsize = 11)
    plt.grid(True, linestyle = '--', alpha = 0.4)
    plt.tight_layout()
    
    save_dir = base_dir / "evaluation_results"
    save_dir.mkdir(exist_ok = True)
    save_path = save_dir / "tsne_latent_clusters.png"
    plt.savefig(save_path, dpi = 400, bbox_inches = 'tight')
    plt.close()


