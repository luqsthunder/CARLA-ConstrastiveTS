import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import euclidean
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""" 
    AugmentedDataset
    Returns a ts together with an augmentation.
"""


class AugmentedDataset(Dataset):
    def __init__(self, dataset, one_device=None):
        super(AugmentedDataset, self).__init__()
        self.current_epoch = 0
        self.samples = [
            {} for _ in range(len(dataset))
        ]  # Initialized with empty dictionaries
        transform = dataset.transform
        sanomaly = dataset.sanomaly
        dataset.transform = None
        self.dataset = dataset
        self.one_device = one_device

        if isinstance(transform, dict):
            self.ts_transform = transform["standard"]
            self.augmentation_transform = transform["augment"]
        else:
            self.ts_transform = transform
            self.augmentation_transform = transform
            self.subseq_anomaly = sanomaly

        self.create_pairs()

    def __get_device(self):
        return device if self.one_device is None else self.one_device

    def create_pairs(self):
        mmean, sstd = self.dataset.get_info()
        mmean = torch.tensor(mmean, dtype=torch.float32).to(self.__get_device())
        sstd = torch.tensor(sstd, dtype=torch.float32).to(self.__get_device())

        for index in tqdm(range(len(self.dataset)), desc="creating pairs"):
            item = self.dataset.__getitem__(index)

            ts_org = item["ts_org"].clone().detach().to(self.__get_device())
            ts_trg = item["target"].clone().detach().to(self.__get_device())

            # Get random neighbor from windows before time step T
            if index > 10:
                rand_nei = np.random.randint(index - 10, index)
                sample_nei = self.dataset.__getitem__(rand_nei)
                # ts_w_augment = sample_nei['ts_org']
                ts_w_augment = (
                    sample_nei["ts_org"].clone().detach().to(self.__get_device())
                )
            else:
                ts_w_augment = self.augmentation_transform(ts_org)

            ts_ss_augment = self.subseq_anomaly(ts_org)

            sstd = torch.where(sstd == 0.0, torch.tensor(1.0, device=sstd.device), sstd)

            self.samples[index] = {
                "ts_org": (ts_org - mmean) / sstd,
                "ts_w_augment": (ts_w_augment - mmean) / sstd,
                "ts_ss_augment": (ts_ss_augment - mmean) / sstd,
                "target": ts_trg,
            }

    def __len__(self):
        return len(self.dataset)

    def concat_ds(self, new_ds):
        self.dataset.data = np.concatenate(
            (self.dataset.data, new_ds.dataset.data), axis=0
        )
        self.dataset.targets = np.concatenate(
            (self.dataset.targets, new_ds.dataset.targets), axis=0
        )

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, index):
        return self.samples[index]


""" 
    NeighborsDataset
    Returns a ts with one of its neighbors.
"""


class NeighborsDataset(Dataset):
    def __init__(self, dataset, transform, N_indices, F_indices, p, one_device=None):
        super(NeighborsDataset, self).__init__()

        if isinstance(transform, dict):
            self.anchor_transform = transform["standard"]
            self.neighbor_transform = transform["augment"]
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform

        self.one_device = one_device

        dataset.transform = None
        all_data = dataset.data.to(self.__get_device())
        self.dataset = dataset

        NN_indices = (
            N_indices.copy()
        )  # Nearest neighbor indices (np.array  [len(dataset) x k])
        FN_indices = (
            F_indices.copy()
        )  # Nearest neighbor indices (np.array  [len(dataset) x k])
        if p["num_neighbors"] is not None:
            self.NN_indices = NN_indices[:, : p["num_neighbors"]]
            self.FN_indices = FN_indices[:, -p["num_neighbors"] :]
        # assert( int(self.indices.shape[0]/4) == len(self.dataset) )

        self.dataset.data = dataset.data.to(self.__get_device())
        self.dataset.targets = dataset.targets.to(self.__get_device())
        num_samples = self.dataset.data.shape[0]
        NN_index = np.array(
            [np.random.choice(self.NN_indices[i], 1)[0] for i in range(num_samples)]
        )
        FN_index = np.array(
            [np.random.choice(self.FN_indices[i], 1)[0] for i in range(num_samples)]
        )
        self.NNeighbor = all_data[NN_index]
        self.FNeighbor = all_data[FN_index]

    def __get_device(self):
        return device if self.one_device is None else self.one_device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)

        # NN_index = np.random.choice(self.N_indices[index], 1)[0]
        NNeighbor = self.NNeighbor.__getitem__(index)
        # FN_index = np.random.choice(self.F_indices[index], 1)[0]
        FNeighbor = self.FNeighbor.__getitem__(index)

        # anchor['ts_org'] = self.anchor_transform(anchor['ts_org'])
        # NNeighbor['ts_org'] = self.neighbor_transform(NNeighbor['ts_org'])
        # FNeighbor['ts_org'] = self.neighbor_transform(FNeighbor['ts_org'])

        output["anchor"] = anchor["ts_org"]
        output["NNeighbor"] = NNeighbor
        output["FNeighbor"] = FNeighbor
        output["possible_nneighbors"] = torch.from_numpy(self.NN_indices[index])
        output["possible_fneighbors"] = torch.from_numpy(self.FN_indices[index])
        output["target"] = anchor["target"]

        return output

    def concat_ds(self, new_ds):
        self.dataset.data = np.concatenate(
            (self.dataset.data, new_ds.dataset.data), axis=0
        )
        self.dataset.targets = np.concatenate(
            (self.dataset.targets, new_ds.dataset.targets), axis=0
        )
