import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from utils.common_config import (
    get_model,
    get_aug_train_dataset,
    get_val_dataloader,
    get_train_transformations,
)
from utils.config import create_config

p = create_config(
    "configs/env.yml",
    "configs/classification/carla_classification_sre.yml",
    "single-url-method",
)
p["batch_size"] = 1
model = get_model(p, p["pretext_model"]).to("cuda:3")

train_transformations = get_train_transformations(p)
train_dataset = get_aug_train_dataset(
    p, train_transformations, to_neighbors_dataset=True, one_device="cuda:3"
)
tst_dl = get_val_dataloader(p, train_dataset)


negs = []
pos = []

for batch in tqdm(tst_dl):
    ts = batch["anchor"]
    target = batch["target"]

    bs, w, h = ts.shape
    res = model(ts.view(bs, h, w), forward_pass="return_all")
    features = res["features"].detach().cpu().numpy().reshape(1, 8)

    if target[0] == 0:
        pos.append(features)
    else:
        negs.append(features)


np.random.seed(42)
np.random.shuffle(pos)
np.random.shuffle(negs)

pos = pos[:10]
pos = sorted(pos, key=lambda x: np.linalg.norm(x))

negs = negs[:10]
negs = sorted(negs, key=lambda x: np.linalg.norm(x))

sns.heatmap(np.concatenate(negs))
plt.savefig("sre-ng-featuremap.png")
plt.close()


sns.heatmap(np.concatenate(pos))
plt.savefig("sre-ps-featuremap.png")
plt.close()
