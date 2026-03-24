import numpy as np
import torch
from torch.utils.data import Dataset
import os


def filter_dataset(dataset, num_classes):
    filtered = [(img, lab) for img, lab in dataset if lab < num_classes]
    return filtered



class QuerySupportTrainDataset(Dataset):
    def __init__(self, dataset, support_size, num_classes):
        # Store original dataset and parameters
        self.dataset = dataset
        self.support_size = support_size
        self.num_classes = num_classes

        # Build a list of valid indices using labels without loading images
        if hasattr(dataset, 'targets'):
            all_targets = list(dataset.targets)
            self.indices = [i for i, lab in enumerate(all_targets) if lab < num_classes]
        elif hasattr(dataset, 'samples'):
            self.indices = [i for i, (_, lab) in enumerate(dataset.samples) if lab < num_classes]
        else:
            # Fallback: may load images, but only labels
            self.indices = [i for i, (_, lab) in enumerate(dataset) if lab < num_classes]

        self.num_samples = len(self.indices)
        # Precompute label-to-indices mapping for efficient support sampling
        self.label_to_indices = {}
        if hasattr(dataset, 'targets'):
            for i in self.indices:
                lab = dataset.targets[i]
                if isinstance(lab, torch.Tensor):
                    lab = lab.item()
                self.label_to_indices.setdefault(lab, []).append(i)
        elif hasattr(dataset, 'samples'):
            for i in self.indices:
                lab = dataset.samples[i][1]
                self.label_to_indices.setdefault(lab, []).append(i)
        else:
            for i in self.indices:
                _, lab = dataset[i]
                self.label_to_indices.setdefault(lab, []).append(i)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        query_img, query_label = self.dataset[real_idx]
        if query_img.dim() == 2:
            query_img = query_img.unsqueeze(0)
        return query_img, query_label, idx


def query_support_train_collate_fn(batch, support_dataset, support_size, num_classes):
    query_imgs, query_labels, query_indices = zip(*batch)
    query_imgs = torch.stack(query_imgs, dim=0)
    query_labels = torch.tensor(query_labels)
    
    # Evenly sample support_size among query labels
    unique_labels = np.unique(query_labels.numpy())
    n_labels = len(unique_labels)
    support_indices = []
    if n_labels > 0:
        base = support_size // n_labels
        extra = support_size % n_labels
        for idx, lab in enumerate(unique_labels):
            candidates = support_dataset.label_to_indices.get(int(lab), [])
            # filtering
            candidates = np.setdiff1d(candidates, query_indices).tolist()
            k = base + (1 if idx < extra else 0)
            if candidates:
                if len(candidates) >= k:
                    chosen = np.random.choice(candidates, k, replace=False)
                else:
                    chosen = np.random.choice(candidates, k, replace=True)
                support_indices.extend(int(i) for i in chosen)

    support_data = [support_dataset[i] for i in support_indices]
    support_imgs = torch.stack([img if img.dim() == 3 else img.unsqueeze(0) for img, lab, _ in support_data], dim=0)
    support_labels = torch.tensor([lab for img, lab, _ in support_data])
    
    perm = torch.randperm(num_classes)
    query_labels = perm[query_labels]
    support_labels = perm[support_labels]
    
    return query_imgs, support_imgs, query_labels, support_labels


class QuerySupportTestDataset(Dataset):
    def __init__(self, query_dataset, support_dataset, support_size, query_digits, support_digits, num_classes):
        # Initialize label sets
        if query_digits is None:
            query_digits = set(range(num_classes))
        if support_digits is None:
            support_digits = set(range(num_classes))
        self.support_size = support_size
        self.num_classes = num_classes
        self.query_dataset = query_dataset

        # Vectorized computation of query indices
        if hasattr(query_dataset, 'targets'):
            targets_arr = np.array(query_dataset.targets)
            mask = np.isin(targets_arr, list(query_digits))
            self.query_indices = mask.nonzero()[0].tolist()
        elif hasattr(query_dataset, 'samples'):
            labs_arr = np.array([lab for _, lab in query_dataset.samples])
            mask = np.isin(labs_arr, list(query_digits))
            self.query_indices = mask.nonzero()[0].tolist()
        else:
            # Fallback to Python loop if necessary
            self.query_indices = [i for i, (_, lab) in enumerate(query_dataset) if lab in query_digits]
        self.num_queries = len(self.query_indices)

        # Build support indices without loading images
        if hasattr(support_dataset, 'targets'):
            targets_arr = np.array(support_dataset.targets)
            mask = np.isin(targets_arr, list(support_digits))
            self.support_indices = mask.nonzero()[0].tolist()
        elif hasattr(support_dataset, 'samples'):
            self.support_indices = [i for i, (_, lab) in enumerate(support_dataset.samples) if lab in support_digits]
        else:
            self.support_indices = [i for i, (_, lab) in enumerate(support_dataset) if lab in support_digits]

        # Keep reference to raw support dataset for on-the-fly sampling
        self.support_dataset = support_dataset


    def __len__(self):
        return self.num_queries

    def __getitem__(self, idx):
        real_idx = self.query_indices[idx]
        query_img, query_label = self.query_dataset[real_idx]
        if query_img.dim() == 2:
            query_img = query_img.unsqueeze(0)
        return query_img, query_label, idx


def query_support_test_collate_fn(batch, support_dataset, support_size):
    query_imgs, query_labels, _ = zip(*batch)
    query_imgs = torch.stack(query_imgs, dim=0)
    query_labels = torch.tensor(query_labels)
    
    available_indices = np.array(support_dataset.support_indices)
    if not support_size:
        support_size = len(available_indices)
    indices = np.random.choice(available_indices, support_size, replace=False)
    support_data = [support_dataset.support_dataset[i] for i in indices]
    support_imgs = torch.stack([img if img.dim() == 3 else img.unsqueeze(0) for img, _ in support_data], dim=0)
    support_labels = torch.tensor([lab for _, lab in support_data])
    
    return query_imgs, support_imgs, query_labels, support_labels