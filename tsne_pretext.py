import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.config import create_config
from utils.common_config import (
    get_model,
    get_train_dataset,
    get_val_dataset,
    get_train_transformations,
    get_val_transformations1,
    get_val_dataloader,
    inject_sub_anomaly,
)

device = torch.device("cuda:3")

p = create_config(
    "configs/env.yml",
    "configs/pretext/carla_pretext_msl.yml",
    "M-6",
)
p["batch_size"] = 1
p["num_heads"] = 1
p["setup"] = "classification"

model = get_model(p, p["pretext_model"]).to(device)
sanomaly = inject_sub_anomaly(p)

train_transforms = get_train_transformations(p)
val_transforms = get_val_transformations1(p)
train_dataset = get_train_dataset(
    p, train_transforms, sanomaly, to_augmented_dataset=True, split="train"
)
val_dataset = get_val_dataset(
    p, val_transforms, sanomaly, False, train_dataset.mean, train_dataset.std
)
val_dataloader = get_val_dataloader(p, val_dataset)
train_dataloader = get_val_dataloader(p, train_dataset)

x_data = []
y_data = []

for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
    ts_anc = batch["ts_org"].float().to(device, non_blocking=True)

    target = batch["target"].detach().cpu().numpy()

    output = model(ts_anc.view(1, ts_anc.shape[2], ts_anc.shape[1]))[0].detach().cpu().numpy()

    x_data.append(output[0])
    y_data.append(target)

tsne = TSNE(n_components=2, learning_rate=100, init="pca", perplexity=25)
transform_data = tsne.fit_transform(np.array(x_data))
res = pd.DataFrame(
    np.concatenate([transform_data, y_data], axis=1), columns=["x", "y", "cl"]
)
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
sns.scatterplot(data=res, x="x", y="y", style="cl", ax=ax, hue="cl")
plt.savefig("tsne-no-sanomaly.png")
plt.close(fig)
