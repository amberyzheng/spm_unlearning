import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import random


import faiss
import math

class PairwiseRelationExpert(nn.Module):
    def __init__(self, embed_dim_q, embed_dim_s, num_classes, hidden_dim=None):
        super(PairwiseRelationExpert, self).__init__()
        self.num_classes = num_classes
        if hidden_dim is None:
            hidden_dim = embed_dim_q
        self.linear_q = nn.Linear(embed_dim_q, hidden_dim, bias=False)
        self.linear_s = nn.Linear(embed_dim_s, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
    
    def forward(self, q, S, support_labels):
        # Compute one-hot for current batch classes
        unique_labels, inv_indices = torch.unique(support_labels, sorted=True, return_inverse=True)
        V_small = F.one_hot(inv_indices, num_classes=unique_labels.size(0)).float()
        # Transform query and support embeddings
        q_trans = self.linear_q(q)    # (B, hidden_dim)
        S_trans = self.linear_s(S)    # (C, hidden_dim)
        # Compute pairwise relation scores
        # h = F.relu(q_trans.unsqueeze(1) + S_trans.unsqueeze(0) + self.bias)
        h = torch.matmul(q_trans, S_trans.transpose(0, 1)) # (B, C)
        h = h / math.sqrt(self.hidden_dim) 
        weights = F.softmax(h, dim=1)       # (B, C)
        # Aggregate on reduced label set
        out_small = torch.matmul(weights, V_small)  # (B, C)
        # Map back to full class dimension
        B = q.size(0)
        full_out = torch.zeros(B, self.num_classes, device=out_small.device)
        # Map class logits back to full dimension, zero elsewhere
        for i, lab in enumerate(unique_labels):
            if lab < self.num_classes: # To ignore random unlearning labels
                full_out[:, lab] = out_small[:, i]
        return full_out

class Feature_Aggregation(nn.Module):
    # Averge the support embeddings with the same labels
    def __init__(self, num_classes):
        super(Feature_Aggregation, self).__init__()
        self.num_classes = num_classes
        
    def forward(self, S, support_labels):
        # S: (N, embed_dim), support_labels: (N,)
        # output: S_mean: (num_classes, embed_dim), unique_labels: (num_classes,)
        unique_labels, inv_indices = torch.unique(support_labels, sorted=True, return_inverse=True)
        # Compute mean for each class
        S_mean = torch.zeros(len(unique_labels), S.size(1), device=S.device)
        for i, label in enumerate(unique_labels):
            mask = (support_labels == label)
            S_mean[i] = S[mask].mean(dim=0)
        return S_mean, unique_labels

class GatingNetwork(nn.Module):
    def __init__(self, embed_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(2 * embed_dim, num_experts)
        
    def forward(self, q, S):
        # q: (B, embed_dim), S: (N, embed_dim)
        S_mean = S.mean(dim=0)                     # (embed_dim)
        S_mean_exp = S_mean.unsqueeze(0).expand(q.size(0), -1)  # (B, embed_dim)
        combined = torch.cat([q, S_mean_exp], dim=-1)           # (B, 2 * embed_dim)
        gating_logits = self.fc(combined)            # (B, num_experts)
        gating_weights = F.softmax(gating_logits, dim=-1)
        return gating_weights
    

class MoEModel(nn.Module):
    def __init__(self, embed_dim, num_experts, num_heads=1, num_classes=10):
        super().__init__()
        expert_class = PairwiseRelationExpert
        self.experts = nn.ModuleList([
            expert_class(embed_dim, embed_dim, num_classes)
            for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(embed_dim, num_experts)
        
    def forward(self, q, S, support_labels):
        # compute each expert's full-class output
        expert_outputs = [expert(q, S, support_labels) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, C)
        gate = self.gating(q, S).unsqueeze(-1)               # (B, num_experts, 1)
        out = torch.sum(gate * expert_outputs, dim=1)       # (B, C)
        return out


class SemiParametricClassifier(pl.LightningModule):
    def __init__(self, embed_dim=None, num_experts=1, num_heads=1, num_classes=10, encoder_type="cnn", learning_rate=1e-3, pretrained_imagenet=True, pretrained_cifar=False, ckpt_path=None):
        super(SemiParametricClassifier, self).__init__()
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.encoder_type = encoder_type
        self.pretrained_imagenet = pretrained_imagenet
        self.pretrained_cifar = pretrained_cifar
        if encoder_type.lower() == "resnet18":
            if pretrained_imagenet:
                import torchvision.models as models
                resnet = models.resnet18(pretrained=True)
                embed_dim = resnet.fc.in_features
                resnet.fc = nn.Identity(resnet.fc.in_features, resnet.fc.in_features)
                self.encoder = resnet
            else:
                from nets.resnet import ResNet18
                self.encoder = ResNet18(embed_dim=embed_dim)
                if pretrained_cifar:
                    assert ckpt_path is not None, "Checkpoint path must be provided for pretrained CIFAR ResNet"
                    self.encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
                    print(f"Loaded pretrained ResNet18 from {ckpt_path}")
        else:
            raise ValueError("encoder_type is not supported")
        self.moe = MoEModel(embed_dim, num_experts, num_heads, num_classes)
        self.feature_aggregation = Feature_Aggregation(num_classes)
        self.loss_fn = nn.NLLLoss()
        self.encoder_params = list(self.encoder.parameters())
        self.other_params = list(self.moe.parameters()) + list(self.loss_fn.parameters())
        self.num_classes = num_classes

    def forward(self, query_img, support_imgs, support_labels):
        q_emb = self.encoder(query_img)  # [B, D]
        support_emb = self.encoder(support_imgs)
        if support_imgs.shape[0] != self.num_classes:
            support_emb, support_labels = self.feature_aggregation(support_emb, support_labels)
        moe_output = self.moe(q_emb, support_emb, support_labels)
        return torch.log(moe_output.clamp(min=1e-8))
        # return torch.log(q_emb.clamp(min=1e-8))
    
    def training_step(self, batch, batch_idx):
        query_img, support_imgs, query_label, support_labels = batch
        logits = self(query_img, support_imgs, support_labels)
        loss = self.loss_fn(logits, query_label)
        self.log("train_loss", loss, prog_bar=True)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == query_label).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        query_img, support_imgs, query_label, support_labels = batch
        logits = self(query_img, support_imgs, support_labels)
        loss = self.loss_fn(logits, query_label)
        self.log("val_loss", loss, prog_bar=True)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == query_label).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        query_img, query_label = batch
        B = query_img.size(0)
        k = getattr(self, "support_size_eval", 1024)  # default k for evaluation

        # compute query embedding
        q_emb = self.encoder(query_img)

        # build FAISS index and load precomputed embeddings once
        if not hasattr(self, "_faiss_index"):
            # Load embeddings and labels from file
            if hasattr(self, "embedding_file"):
                emb_file = getattr(self, "embedding_file", "train_embeddings.pt")
                print(f"Loading embeddings from {emb_file}")
                data = torch.load(emb_file, map_location="cpu")
                embeddings = data["embeddings"]  # Tensor [N, D]
                labels = data["labels"]          # Tensor [N]
            else:
                # compute embeddings from training data
                train_loader = self.trainer.datamodule.train_dataloader()
                emb_list, label_list = [], []
                for imgs, lbls in train_loader:
                    emb = self.encoder(imgs.to(self.device))
                    emb_list.append(emb.detach().cpu())
                    label_list.append(lbls)
                embeddings = torch.cat(emb_list, dim=0)
                labels = torch.cat(label_list, dim=0)

            # Create FAISS index
            emb_np = embeddings.numpy().astype("float32")
            index = faiss.IndexFlatL2(emb_np.shape[1])
            index.add(emb_np)
            self._faiss_index = index
            self._support_embeddings = embeddings.to(self.device)
            self._support_labels = labels.to(self.device)

        # retrieve k nearest neighbors with FAISS
        q_np = q_emb.detach().cpu().numpy().astype("float32")
        dists, idxs = self._faiss_index.search(q_np, k)
        idxs_tensor = torch.from_numpy(idxs).to(self.device)
        # gather support embeddings and labels
        supp_embs = self._support_embeddings[idxs_tensor]  # (B, k, D)
        supp_labels = self._support_labels[idxs_tensor]    # (B, k)

        # flatten for the MoE module
        supp_embs_flat = supp_embs.view(B * k, -1)
        supp_labels_flat = supp_labels.view(B * k)

        # forward through the mixture-of-experts
        moe_out = self.moe(q_emb, supp_embs_flat, supp_labels_flat)
        logits = torch.log(moe_out.clamp(min=1e-8))

        # compute accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == query_label).float().mean()
        loss = self.loss_fn(logits, query_label)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss": loss, "test_acc": acc}

    def load_pretrained_encoder(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        encoder_state = {k.replace("encoder.", ""): v for k, v in state["state_dict"].items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"Pretrained encoder loaded from {checkpoint_path}")

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen.")

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder parameters unfrozen.")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        if self.encoder_type.lower() == "resnet18" and self.pretrained_imagenet is True:
            encoder_params = self.encoder_params
            other_params = self.other_params
            optimizer = optim.SGD([
                {"params": encoder_params, "lr": self.learning_rate / 100},
                {"params": other_params, "lr": self.learning_rate}
                ], 
                momentum=0.9, 
                weight_decay=1e-4
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            return optimizer
        elif (self.encoder_type.lower() == 'resnet18' and not self.pretrained_imagenet) or self.encoder_type.lower() == "dla":
            optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[91, 36], gamma=0.1
            )
            return [optimizer], [scheduler]
        elif self.encoder_type.lower() != "cnn":
            optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            return [optimizer], [scheduler]
        else:
            return optim.Adam(params, lr=self.learning_rate)