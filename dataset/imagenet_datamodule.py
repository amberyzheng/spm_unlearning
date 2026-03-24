import os
import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.dataset import QuerySupportTrainDataset, QuerySupportTestDataset, query_support_train_collate_fn, query_support_test_collate_fn


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, support_size_train, support_size_eval, batch_size, num_classes=1000, query_digits=None, support_digits=None, normalize=True, model_type='edm', data_dir='/path/to/imagenet', **kwargs):
        super().__init__()
        self.support_size_train = support_size_train
        self.support_size_eval = support_size_eval
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.query_digits = query_digits
        self.support_digits = support_digits
        self.data_dir = data_dir
        if model_type != 'flow':
            if normalize:
                self.train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                self.test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            else:
                self.train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                self.test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])
        else:
            from torchvision.transforms.v2 import ToDtype, ToImage
            self.train_transform = transforms.Compose([
                ToImage(),
                transforms.RandomHorizontalFlip(),
                ToDtype(torch.float32, scale=True),
            ])
            self.test_transform = transforms.Compose([
                ToImage(),
                ToDtype(torch.float32, scale=True),
            ])
        self.dataset = torchvision.datasets.ImageNet


    def prepare_data(self):
        # Data is assumed to be downloaded and extracted manually; no download flag supported
        self.dataset(root=self.data_dir, split='train')
        self.dataset(root=self.data_dir, split='val')

    def setup(self, stage=None):
        if stage in ("fit", None):
            train_set = self.dataset(root=self.data_dir, split='train', transform=self.train_transform)
            val_set = self.dataset(root=self.data_dir, split='val', transform=self.test_transform)
            if hasattr(self, 'no_train_digits'):
                from torch.utils.data import Subset
                targets = torch.as_tensor(train_set.targets)
                mask = ~torch.isin(targets, torch.tensor(self.no_train_digits))
                rest_idx = mask.nonzero(as_tuple=True)[0].tolist()
                train_set = Subset(train_set, rest_idx)
                train_set.targets = targets[rest_idx].tolist()
                val_targets = torch.as_tensor(val_set.targets)
                val_mask = ~torch.isin(val_targets, torch.tensor(self.no_train_digits))
                val_rest_idx = val_mask.nonzero(as_tuple=True)[0].tolist()
                val_set = Subset(val_set, val_rest_idx)
                val_set.targets = val_targets[val_rest_idx].tolist()
                print(f"[DataModule] Exclude digits {self.no_train_digits} "
                    f"from training, total {len(train_set)} samples left.")
                    
            self.train_dataset = QuerySupportTrainDataset(train_set, self.support_size_train, self.num_classes)
            self.val_dataset = QuerySupportTestDataset(val_set, train_set, self.support_size_eval, self.query_digits, self.support_digits, self.num_classes)
        if stage == "test" or stage == "finetune":
            self.train_dataset = self.dataset(root=self.data_dir, split='train', transform=self.train_transform)
            self.test_dataset = self.dataset(root=self.data_dir, split='val', transform=self.test_transform)
            # self.train_dataset = QuerySupportTrainDataset(train_set, self.support_size_train, self.num_classes)
            # self.test_dataset = QuerySupportTestDataset(val_set, train_set, self.support_size_eval, self.query_digits, self.support_digits, self.num_classes)
    
    def set_no_train_digits(self, query_digits):
        self.no_train_digits = query_digits

    def train_dataloader(self):
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
            collate_fn=lambda batch: query_support_train_collate_fn(batch, self.train_dataset, self.support_size_train, self.num_classes)
        )

    def val_dataloader(self):
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=6,
            collate_fn=lambda batch: query_support_test_collate_fn(batch, self.val_dataset, self.support_size_eval)
        )

    def test_dataloader(self):
        from torch.utils.data.distributed import DistributedSampler
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=6,
        )
    
    def support_loader(self):
        from torch.utils.data import Subset, DataLoader
        if not self.support_digits:
            self.support_digits = set(range(self.num_classes))
        if not hasattr(self, '_support_indices'):
            import numpy as np
            targets = np.array(self.train_dataset.targets)
            mask = np.isin(targets, list(self.support_digits))
            self._support_indices = np.nonzero(mask)[0].tolist()
        subset = Subset(self.train_dataset, self._support_indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
        )
    
    def unlearn_train_dataloader(self):
        from torch.utils.data import Subset, DataLoader
        if not self.query_digits:
            self.query_digits = set(range(self.num_classes))
        if not hasattr(self, '_unlearn_train_indices'):
            import numpy as np
            # Use underlying dataset targets for faster indexing
            base_targets = np.array(self.train_dataset.targets)
            mask = np.isin(base_targets, list(self.query_digits))
            self._unlearn_train_indices = np.nonzero(mask)[0].tolist()
        subset = Subset(self.train_dataset, self._unlearn_train_indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
        )

    def rest_train_dataloader(self):
        from torch.utils.data import Subset, DataLoader
        if not self.query_digits:
            self.query_digits = set(range(self.num_classes))
        if not hasattr(self, '_rest_train_indices'):
            import numpy as np
            base_targets = np.array(self.train_dataset.targets)
            mask = ~np.isin(base_targets, list(self.query_digits))
            self._rest_train_indices = np.nonzero(mask)[0].tolist()
        subset = Subset(self.train_dataset, self._rest_train_indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            
        )

    def unlearn_test_dataloader(self):
        from torch.utils.data import Subset, DataLoader
        if not self.query_digits:
            self.query_digits = set(range(self.num_classes))
        if not hasattr(self, '_unlearn_test_indices'):
            import numpy as np
            base_targets = np.array(self.test_dataset.targets)
            mask = np.isin(base_targets, list(self.query_digits))
            self._unlearn_test_indices = np.nonzero(mask)[0].tolist()
        subset = Subset(self.test_dataset, self._unlearn_test_indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
            
        )

    def rest_test_dataloader(self):
        from torch.utils.data import Subset, DataLoader
        if not self.query_digits:
            self.query_digits = set(range(self.num_classes))
        if not hasattr(self, '_rest_test_indices'):
            import numpy as np
            base_targets = np.array(self.test_dataset.targets)
            mask = ~np.isin(base_targets, list(self.query_digits))
            self._rest_test_indices = np.nonzero(mask)[0].tolist()
        subset = Subset(self.test_dataset, self._rest_test_indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            
        )
