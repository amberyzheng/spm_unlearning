import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from utils.dataset import QuerySupportTrainDataset, QuerySupportTestDataset, query_support_train_collate_fn, query_support_test_collate_fn
import random
from torch.utils.data import Subset

class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, support_size_train, support_size_eval, batch_size, num_classes=10, query_digits=None, support_digits=None, dataset_name="cifar10", model_type='edm', **kwargs):
        super().__init__()
        self.support_size_train = support_size_train
        self.support_size_eval = support_size_eval
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.query_digits = query_digits
        self.support_digits = support_digits
        self.dataset_name = dataset_name.lower()

        if model_type == 'classifier' or model_type == 'ablation':
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif model_type == 'edm' or model_type == 'ddpm':
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if self.dataset_name == "cifar100":
            self.dataset = torchvision.datasets.CIFAR100
        else:
            self.dataset = torchvision.datasets.CIFAR10

    def prepare_data(self):
        self.dataset(root='./data', train=True, download=True)
        self.dataset(root='./data', train=False, download=True)

    def setup(self, stage=None):
        if stage in ("fit", None):
            cifar_train = self.dataset(root='./data', train=True, transform=self.train_transform)
            self.train_dataset = QuerySupportTrainDataset(cifar_train, self.support_size_train, self.num_classes)
            cifar_val = self.dataset(root='./data', train=False, transform=self.test_transform)
            self.val_dataset = QuerySupportTestDataset(cifar_val, cifar_train, self.support_size_eval, self.query_digits, self.support_digits, self.num_classes)
            self.test_dataset = self.dataset(root='./data', train=False, transform=self.test_transform)
        if stage == "test" or stage == "finetune":
            self.train_dataset = self.dataset(root='./data', train=True, transform=self.train_transform)
            self.test_dataset = self.dataset(root='./data', train=False, transform=self.test_transform)
    
    def set_no_train_digits(self, query_digits):
        self.no_train_digits = query_digits
    
    def set_random_unlearn_portion(self, portion):
        self.random_unlearn_portion = portion

    def set_indexes_to_replace(self, indexes):
        self.indexes_to_replace = indexes

    def train_dataloader(self):
        if hasattr(self, 'no_train_digits'): # alist of digits to exclude from training
            from torch.utils.data import Subset
            all_idx = list(range(len(self.train_dataset.dataset)))
            rest_idx = [i for i in all_idx if self.train_dataset.dataset[i][1] not in self.no_train_digits]
            subset = Subset(self.train_dataset.dataset, rest_idx)
            self.train_dataset = QuerySupportTrainDataset(subset, self.support_size_train, self.num_classes)
            print('[DataModule] Exclude digits {} from train_dataloader, total {} samples left.'.format(self.no_train_digits, len(self.train_dataset)))

        if hasattr(self, 'random_unlearn_portion'): # a portion of data to exclude from training
            from torh.utils.data import Subset
            all_idx = list(range(len(self.train_dataset.dataset)))
            num_exclude = int(len(all_idx) * self.random_unlearn_portion)
            self.exclude_idx = set(random.sample(all_idx, num_exclude))
            rest_idx = [i for i in all_idx if i not in self.exclude_idx]
            subset = Subset(self.train_dataset.dataset, rest_idx)
            self.train_dataset = QuerySupportTrainDataset(subset, self.support_size_train, self.num_classes)
            print('[DataModule] Exclude random {} from training, total {} samples left.'.format(len(self.exclude_idx), len(self.train_dataset)))

        if hasattr(self, 'indexes_to_replace'): # a list of indexes to exclude from training
            from torch.utils.data import Subset
            all_idx = list(range(len(self.train_dataset.dataset)))
            rest_idx = [i for i in all_idx if i not in self.indexes_to_replace]
            subset = Subset(self.train_dataset.dataset, rest_idx)
            self.train_dataset = QuerySupportTrainDataset(subset, self.support_size_train, self.num_classes)
            print('[DataModule] Exclude {} specified samples from training, total {} samples left.'.format(len(self.indexes_to_replace), len(self.train_dataset)))
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            collate_fn=lambda batch: query_support_train_collate_fn(batch, self.train_dataset, self.support_size_train, self.num_classes)
        )

    def val_dataloader(self):
        if hasattr(self, 'no_train_digits'): # alist of digits to exclude from validation
            from torch.utils.data import Subset
            all_idx = list(range(len(self.val_dataset.query_dataset)))
            rest_idx = [i for i in all_idx if self.val_dataset.query_dataset[i][1] not in self.no_train_digits]
            subset = Subset(self.val_dataset.query_dataset, rest_idx)
            self.val_dataset = QuerySupportTestDataset(subset, self.val_dataset.support_dataset, self.support_size_eval, self.query_digits, self.support_digits, self.num_classes)
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
            collate_fn=lambda batch: query_support_test_collate_fn(batch, self.val_dataset, self.support_size_eval)
        )

    def test_dataloader(self):
        # from torch.utils.data.distributed import DistributedSampler
        # sampler = None
        # if torch.distributed.is_available() and torch.distributed.is_initialized():
        #     self.test_dataset = self.train_dataset
        #     sampler = DistributedSampler(self.test_dataset, shuffle=False)
        # return DataLoader(
        #     self.test_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,  # set to False because sampler takes care of shuffling if needed
        #     sampler=sampler,
        #     num_workers=6,
        #     collate_fn=lambda batch: query_support_test_collate_fn(batch, self.test_dataset, self.support_size_eval)
        # )
         return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
        )
    

    def support_loader(self):
        from torch.utils.data import Subset, DataLoader
        if hasattr(self, 'random_unlearn_portion'):
            if not hasattr(self, 'exclude_idx'):
                all_idx = list(range(len(self.train_dataset)))
                num_exclude = int(len(all_idx) * self.random_unlearn_portion)
                self.exclude_idx = set(random.sample(all_idx, num_exclude))
                rest_idx = [i for i in all_idx if i not in self.exclude_idx]
                print('[DataModule] Exclude random {} from training, total {} samples left.'.format(len(self.exclude_idx), len(rest_idx)))
            
            # V1: directly exclude the samples
            indices = [i for i in range(len(self.train_dataset)) if i not in self.exclude_idx]
            
            # V2: rewrtie the dataset labels of the excluded samples to 10+label (and exclude them after attention, that is, ignore labels >= num_classes)
            # self.train_dataset.targets = [
            #     label+10 if i in self.exclude_idx else label
            #     for i, label in enumerate(self.train_dataset.targets)
            # ]
            # indices = [i for i, (_, label) in enumerate(self.train_dataset)]

            print('[DataModule] Exclude random {} samples from support loader.'.format(len(self.exclude_idx)))
        elif hasattr(self, 'indexes_to_replace'):
            indices = [i for i in range(len(self.train_dataset)) if i not in self.indexes_to_replace]
            print('[DataModule] Exclude {} specified samples from support loader.'.format(len(self.indexes_to_replace)))
        else:
            if not self.support_digits:
                self.support_digits = set(range(self.num_classes))
            indices = [
                i for i, (_, label) in enumerate(self.train_dataset)
                if label in self.support_digits
            ]
        subset = Subset(self.train_dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6
        )
    
    def unlearn_train_dataloader(self): # for UA
        from torch.utils.data import Subset, DataLoader
        if hasattr(self, 'random_unlearn_portion'):
            indices = [i for i in range(len(self.train_dataset)) if i in self.exclude_idx]
            print('[DataModule] Unlearn random {} samples from train_dataloader.'.format(len(indices)))
        elif hasattr(self, 'indexes_to_replace'):
            indices = [i for i in range(len(self.train_dataset)) if i in self.indexes_to_replace]
            print('[DataModule] Unlearn {} specified samples from train_dataloader.'.format(len(indices)))
        else:
            if not self.query_digits:
                self.query_digits = set(range(self.num_classes))
            indices = [
                i for i, (_, label) in enumerate(self.train_dataset)
                if label in self.query_digits
            ]
        subset = Subset(self.train_dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6
        )

    def rest_train_dataloader(self): # for RA
        from torch.utils.data import Subset, DataLoader
        if hasattr(self, 'random_unlearn_portion'):
            all_idx = list(range(len(self.train_dataset)))
            rest_idx = [i for i in all_idx if i not in self.exclude_idx]
            print('[DataModule] Rest random {} samples from train_dataloader.'.format(len(rest_idx)))
        elif hasattr(self, 'indexes_to_replace'):
            all_idx = list(range(len(self.train_dataset)))
            rest_idx = [i for i in all_idx if i not in self.indexes_to_replace]
            print('[DataModule] Rest {} specified samples from train_dataloader.'.format(len(rest_idx)))
        else:
            if not self.query_digits:
                self.query_digits = set(range(self.num_classes))
            all_idx = list(range(len(self.train_dataset)))
            rest_idx = [i for i in all_idx if self.train_dataset[i][1] not in self.query_digits]
        subset = Subset(self.train_dataset, rest_idx)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6
        )

    def unlearn_test_dataloader(self): # for TA
        from torch.utils.data import Subset, DataLoader
        if not self.query_digits:
            self.query_digits = set(range(self.num_classes))
        indices = [
            i for i, (_, label) in enumerate(self.test_dataset)
            if label in self.query_digits
        ]
        subset = Subset(self.test_dataset, indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6
        )

    def rest_test_dataloader(self): # for TA
        from torch.utils.data import Subset, DataLoader
        if hasattr(self, 'random_unlearn_portion'):
            rest_idx = list(range(len(self.test_dataset)))
        elif hasattr(self, 'indexes_to_replace'):
            rest_idx = list(range(len(self.test_dataset)))
        else:
            if not self.query_digits:
                self.query_digits = set(range(self.num_classes))
            all_idx = list(range(len(self.test_dataset)))
            rest_idx = [i for i in all_idx if self.test_dataset[i][1] not in self.query_digits]
        subset = Subset(self.test_dataset, rest_idx)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6
        )
