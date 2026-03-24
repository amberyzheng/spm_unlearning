import torch
import faiss
import time
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)

def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0] if target_train is not None else 0
    n_target_test = target_test.shape[0] if target_test is not None else 0

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

    return np.mean(accs) if accs else 0.0

def SVC_MIA(shadow_train, shadow_test, target_train, target_test):
    # shadow_train, shadow_test, target_train, target_test are tuples (probs, labels)
    shadow_train_prob, shadow_train_labels = shadow_train
    shadow_test_prob, shadow_test_labels   = shadow_test
    target_train_prob, target_train_labels = target_train if target_train is not None else (None, None)
    target_test_prob, target_test_labels   = target_test

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None]) if target_train_prob is not None else None
    target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])

    m_conf = SVC_fit_predict(shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf)
    m_prob = SVC_fit_predict(shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob)

    return {
        "confidence_svc": m_conf,
        "prob_svc": m_prob,
    }


class UnlearnEvaluator:
    def __init__(self):
        self.metrics = {
            'shadow_train_rest_correct': 0, 'shadow_train_rest_total': 0,
            'train_unlearn_correct': 0, 'train_unlearn_total': 0,
            'train_rest_correct': 0,    'train_rest_total': 0,
            'test_unlearn_correct': 0,  'test_unlearn_total': 0,
            'test_rest_correct': 0,     'test_rest_total': 0, 'test_all_rest_correct': 0, 'test_all_rest_total': 0,
        }
        self.start_time = None

    def start_timer(self):
        self.start_time = time.perf_counter()

    def stop_timer(self):
        self.metrics['inference_time'] = time.perf_counter() - self.start_time

    def update(self, preds, labels, split, is_unlearn):
        key = f"{split}_{'unlearn' if is_unlearn else 'rest'}"
        self.metrics[f"{key}_correct"] += (preds == labels).sum().item()
        self.metrics[f"{key}_total"] += labels.size(0)
        print(f"Updated {key}: {self.metrics[f'{key}_correct']}/{self.metrics[f'{key}_total']} correct")

    def compute(self):
        return {
            'train_acc_unlearn':    self.metrics['train_unlearn_correct'] / self.metrics['train_unlearn_total'],
            'train_acc_rest':       self.metrics['train_rest_correct'] / self.metrics['train_rest_total'],
            # 'test_acc_unlearn':     self.metrics['test_unlearn_correct'] / self.metrics['test_unlearn_total'],
            'test_acc_rest':        self.metrics['test_rest_correct'] / self.metrics['test_rest_total'],
            'inference_time':       self.metrics.get('inference_time', 0.0)
        }


# ---- FAISS-based evaluation ----

def build_faiss_index(model, support_loader, device=None, eval_mode=True):
    """
    Build a FAISS index from model embeddings of the support set.
    support_loader should yield (query_imgs, support_imgs, query_labels, support_labels).
    """
    device = device or next(model.parameters()).device
    model.to(device)
    if eval_mode:
        model.eval()
    all_emb = []
    all_labels = []
    with torch.no_grad():
        for support_imgs, support_labels in tqdm(support_loader, desc="Building FAISS index", leave=False):
            support_imgs = support_imgs.to(device) #, dtype=model.dtype)
            try:
                emb = model.encoder(support_imgs)  # (B, D) on GPU
            except:
                emb = model(support_imgs)  # (B, D) on GPU
                emb = F.normalize(emb, p=2, dim=1)
            all_emb.append(emb)
            all_labels.append(support_labels.view(-1).to(device))
    embeddings = torch.cat(all_emb, dim=0)       # still on GPU
    labels = torch.cat(all_labels, dim=0)        # on GPU
    emb_np = embeddings.cpu().numpy().astype('float32')  # one GPU→CPU copy
    index = faiss.IndexFlatL2(emb_np.shape[1])
    index.add(emb_np)
    return index, embeddings, labels

def evaluate_all_test_data(model, datamodule, save_dir, cluster_portion, knn=False, knn_c=False, k=None):
    """
    Evaluate the model on all test data using the datamodule.
    Returns a dict with accuracy and inference time.
    """
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    test_loader = datamodule.test_dataloader()
    print(f"Evaluating {len(test_loader.dataset)} test samples...")

    if cluster_portion < 1.0:
        # Random smaple a portion of support data for clustering
        total_support = len(datamodule.support_loader().dataset)
        sample_size = int(total_support * cluster_portion)
        indices = np.random.choice(total_support, sample_size, replace=False)
        sampled_support = torch.utils.data.Subset(datamodule.support_loader().dataset, indices)
        sampled_loader = torch.utils.data.DataLoader(sampled_support, batch_size=64, shuffle=False)
        datamodule.support_loader = lambda: sampled_loader
        print(f'Building aggregated features for support data ({len(datamodule.support_loader().dataset)} samples)...')

        # Build and save
        start_index_time = time.perf_counter()
        mean_vector = {}
        each_class_count = {}
        with torch.no_grad():
            for support_imgs, support_labels in tqdm(datamodule.support_loader(), desc="Building aggregated features", leave=False):
                support_imgs = support_imgs.to(device, dtype=model.dtype)
                support_emb = model.encoder(support_imgs)  # (B, D)
                for i in range(support_labels.size(0)):
                    label = int(support_labels[i])
                    emb = support_emb[i].detach().cpu()
                    if label not in mean_vector:
                        each_class_count[label] = 1
                        mean_vector[label] = emb.clone()
                    else:
                        each_class_count[label] += 1
                        mean_vector[label] += (emb - mean_vector[label]) / each_class_count[label]
        support_embeddings = torch.stack(list(mean_vector.values()), dim=0).to(device)  # (N, D)
        support_labels = torch.tensor(list(mean_vector.keys()), device=device)  # (N,)
        build_index_time = time.perf_counter() - start_index_time
        print(f"Built support data with {len(support_embeddings)} unique classes in {build_index_time:.2f} seconds.")
    correct = 0
    total = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for query_imgs, query_labels in tqdm(test_loader, desc="Evaluating test data", leave=False):
            query_imgs = query_imgs.to(device)
            q_emb = model.encoder(query_imgs)  # (B, D)
            if knn:
                # search k nearest
                q_np = q_emb.cpu().numpy().astype('float32')
                if knn_c: # use clustering, so just pick the one closest in support embeddings
                    q_tensor = torch.from_numpy(q_np).to(device)
                    dists_all = torch.cdist(q_tensor, support_embeddings)  # (B, N)
                    dists, idxs_tensor = torch.topk(dists_all, 1, largest=False)  # (B, k)  
                else:                        
                    dists, idxs = index.search(q_np, k)
                    idxs_tensor = torch.from_numpy(idxs).to(device)
                supp_embs = support_embeddings[idxs_tensor]           # (B, k, D)
                supp_labels_batch = support_labels[idxs_tensor]       # (B, k)
                n_classes = datamodule.num_classes
                B = q_emb.size(0)
                one_hot = torch.zeros(B, n_classes, device=device)
                ones = torch.ones(B, k, device=device)
                one_hot.scatter_add_(1, supp_labels_batch, ones)
                moe_out = one_hot / k

            else:
                moe_out = model.moe(q_emb, support_embeddings, support_labels)

            logits = torch.log(moe_out.clamp(min=1e-8))
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == query_labels.to(device)).sum().item()
            total += query_labels.size(0)
    inference_time = time.perf_counter() - start_time
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        'test_acc': accuracy,
        'inference_time': inference_time,
        'build_index_time': build_index_time,
    }

    return metrics

def evaluate_unlearn(model, datamodule, k, device=None, knn=False, ablation=False, save_dir=None, use_random_sampling=False):
    """
    Evaluate unlearn and rest splits on train and test, using k-NN over support embeddings.
    Returns a dict with metrics and inference_time.
    
    Args:
        model: The model to evaluate
        datamodule: The datamodule containing the data
        k: Number of nearest neighbors to use
        device: Device to use for evaluation
        knn: Whether to use k-NN instead of MoE
        ablation: Whether to use ablation mode (random sampling)
        save_dir: Directory to save FAISS index and support data
        use_random_sampling: Whether to use random sampling instead of retrieval
    """

    device = device or next(model.parameters()).device
    model.to(device)
    model.eval()
    split_results = {}
    emb_time_unlearn_train = 0.0
    search_time_unlearn_train = 0.0
    moe_time_unlearn_train = 0.0
    support_loader = datamodule.support_loader()
    # prepare shadow training loaders
    test_len = len(datamodule.rest_test_dataloader().dataset)
    full_dataset = datamodule.rest_train_dataloader().dataset
    num_samples = len(full_dataset)
    random_indices = torch.randperm(num_samples)[:test_len].tolist()
    print('Dataset sizes:')
    print(f'  Rest train: {len(datamodule.rest_train_dataloader().dataset)}')
    print(f'  Unlearn train: {len(datamodule.unlearn_train_dataloader().dataset)}')
    print(f'  Rest test: {len(datamodule.rest_test_dataloader().dataset)}')
    print(f'  Unlearn test: {len(datamodule.unlearn_test_dataloader().dataset)}')

    save_dir = os.path.join(save_dir, f"cluster_{datamodule.query_digits}") if save_dir else None
    # os.makedirs(save_dir, exist_ok=True)
    data_path = os.path.join(save_dir or os.getcwd(), "support_data.pt")
    if  os.path.exists(data_path):
        # Load from disk
        data = torch.load(data_path, map_location=device)
        support_embeddings = data["embeddings"]
        support_labels = data["labels"]
        build_index_time = 0.0
        print(f"Loaded support data from {save_dir}")
    elif len(datamodule.support_loader().dataset) < 100000:
        # Build and save (smaller dataset, aggregation in one pass)
        start_index_time = time.perf_counter()
        all_S_emb = []
        all_S_labels = []
        print(f'Building aggregated features for support data ({len(support_loader.dataset)} samples)...')
        with torch.no_grad():
            for support_imgs, support_labels in tqdm(support_loader, desc="Building aggregated features", leave=False):
                support_imgs = support_imgs.to(device, dtype=model.dtype)
                support_emb = model.encoder(support_imgs)  # (B, D)
                all_S_emb.append(support_emb)
                all_S_labels.append(support_labels.view(-1).to(device))
        support_embeddings = torch.cat(all_S_emb, dim=0)  # (N, D)
        support_labels = torch.cat(all_S_labels, dim=0)  # (N,)
        support_embeddings, support_labels = model.feature_aggregation(support_embeddings, support_labels)
        build_index_time = time.perf_counter() - start_index_time
        print(f"Built support data with {len(support_embeddings)} unique classes in {build_index_time:.2f} seconds.")
    else:
        # Build and save (large dataset, aggregation in multiple passes)
        start_index_time = time.perf_counter()
        print(f'Building aggregated features for support data ({len(support_loader.dataset)} samples)...')
        mean_vector = {}
        each_class_count = {}
        with torch.no_grad():
            for support_imgs, support_labels in tqdm(support_loader, desc="Building aggregated features", leave=False):
                support_imgs = support_imgs.to(device, dtype=model.dtype)
                support_emb = model.encoder(support_imgs)  # (B, D)
                for i in range(support_labels.size(0)):
                    label = int(support_labels[i])
                    emb = support_emb[i].detach().cpu()
                    if label not in mean_vector:
                        each_class_count[label] = 1
                        mean_vector[label] = emb.clone()
                    else:
                        each_class_count[label] += 1
                        mean_vector[label] += (emb - mean_vector[label]) / each_class_count[label]
        support_embeddings = torch.stack(list(mean_vector.values()), dim=0).to(device)  # (N, D)
        support_labels = torch.tensor(list(mean_vector.keys()), device=device).to(device)  # (N,)
        build_index_time = time.perf_counter() - start_index_time
        print(f"Built support data with {len(support_embeddings)} unique classes in {build_index_time:.2f} seconds.")
        # save to disk
        save_base = save_dir or os.getcwd()
        os.makedirs(save_base, exist_ok=True)

    evaluator = UnlearnEvaluator()
    evaluator.start_timer()
    # Dictionary to hold collected probabilities and labels per split

    for split, loader, is_unlearn in [
        ('train', datamodule.unlearn_train_dataloader(), True),
        ('train', datamodule.rest_train_dataloader(), False),
        ('test', datamodule.rest_test_dataloader(), False),
    ]:
        if split == 'train' and is_unlearn:
            infer_start_time_unlearn_train = time.perf_counter()
        model.eval()
        # initialize probability collector
        probs = []
        labels_all = []
        with torch.no_grad():
            for query_imgs, query_labels in tqdm(loader, desc=f"{split}_{'unlearn' if is_unlearn else 'rest'}", leave=False):
                query_imgs = query_imgs.to(device)
                # embed query
                t0 = time.perf_counter()
                q_emb = model.encoder(query_imgs)  # (B, D)
                B = q_emb.size(0)
                t1 = time.perf_counter()
                if split == 'train' and is_unlearn:
                    emb_time_unlearn_train += (t1 - t0)

                t2 = time.perf_counter()
                try: # fusion is only used for ablation !!!
                    moe_out = model.moe(q_emb, support_embeddings, support_labels)
                except:
                    moe_out = model.fusion(q_emb, support_embeddings, support_labels)
                t3 = time.perf_counter()
                if split == 'train' and is_unlearn:
                    moe_time_unlearn_train += (t3 - t2)
                logits = torch.log(moe_out.clamp(min=1e-8))

                # collect probabilities and true labels
                probs.append(moe_out.cpu())
                labels_all.append(query_labels)
                preds = torch.argmax(logits, dim=-1).cpu()
                evaluator.update(preds, query_labels, split, is_unlearn)
        if split == 'train' and is_unlearn:
            infer_time_unlearn_train = time.perf_counter() - infer_start_time_unlearn_train

        # aggregate collected probabilities and labels for this split
        probs = torch.cat(probs, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        # store results for this split
        split_results[f"{split}_{'unlearn' if is_unlearn else 'rest'}"] = (probs, labels_all)
                
    evaluator.stop_timer()
    # Compute primary metrics
    metrics = evaluator.compute()
    metrics['build_index_time'] = build_index_time
    metrics['emb_time_unlearn_train'] = emb_time_unlearn_train
    metrics['moe_time_unlearn_train'] = moe_time_unlearn_train
    metrics['emb_moe_time_unlearn_train'] = emb_time_unlearn_train + moe_time_unlearn_train
    metrics['infer_time_unlearn_train'] = infer_time_unlearn_train


    return metrics

def evaluate_retrain(model, datamodule, k=0, knn=False, device=None, save_dir=None, config_path=None, ablation=False,  unlearn_portion=None, indexes_to_replace=[], use_random_sampling=False, acc_v2=False, retrained_model=None):
    # Retrain reference model on rest data only
    print("="*60)
    print("RETRAINING REFERENCE MODEL ON REST DATA")
    print("="*60)
    
    from utils.config import load_config, instantiate_from_config
    import pytorch_lightning as pl
    
    # Load config and modify it for rest-only training
    config = load_config(config_path)
    # Get rest classes (all classes except query_digits)
    all_classes = set(range(datamodule.num_classes))
    if datamodule.query_digits:
        if isinstance(datamodule.query_digits, (list, tuple)):
            unlearn_classes = set(datamodule.query_digits)
        else:
            unlearn_classes = {datamodule.query_digits}
    else:
        unlearn_classes = set()
    rest_classes = list(all_classes - unlearn_classes)
    
    # Modify data config to use only rest classes
    config['data']['params']['query_digits'] = rest_classes
    config['data']['params']['support_digits'] = rest_classes

    # Create model and datamodule for retraining
    reference_model = instantiate_from_config(config['model'])
    rest_datamodule = instantiate_from_config(config['data'])
    if datamodule.query_digits: # class-specific unlearning
        rest_datamodule.set_no_train_digits(list(unlearn_classes))
    elif unlearn_portion is not None: # random unlearning by randomly selecting a portion of training data to unlearn
        rest_datamodule.set_random_unlearn_portion(unlearn_portion)
    if len(indexes_to_replace) > 0: # unlearning specific indexes
        rest_datamodule.set_indexes_to_replace(indexes_to_replace)
    
    
    # Setup training callbacks (simplified)
    lightning_config = config.get('lightning', {})
    callbacks = []
    if 'callbacks' in lightning_config and lightning_config['callbacks'] is not None:
        # Only use essential callbacks
        if 'checkpoint_callback' in lightning_config['callbacks']:
            cb_config = lightning_config['callbacks']['checkpoint_callback'].copy()
            if unlearn_portion is not None: # random unlearning by randomly selecting a portion of training data to unlearn
                cb_config['params']['dirpath'] = os.path.join(save_dir, f"retrained_model/unl_{int(unlearn_portion*100)}")
            elif len(indexes_to_replace) > 0: # unlearning specific indexes
                cb_config['params']['dirpath'] = os.path.join(save_dir, f"retrained_model/unl_{len(indexes_to_replace)}")
            else:
                cb_config['params']['dirpath'] = os.path.join(save_dir, f"retrained_model/unl_{''.join(map(str, sorted(unlearn_classes)))}")
            callbacks.append(instantiate_from_config(cb_config))
    # Create trainer exactly as in main.py - use config parameters
    logger = instantiate_from_config(lightning_config['logger']) if 'logger' in lightning_config else None
    from pytorch_lightning.loggers import WandbLogger
    if logger is not None and isinstance(logger, WandbLogger):
        logger.log_hyperparams(config)
    
    trainer_config = lightning_config.get('trainer', {})
    trainer_config['logger'] = logger
    trainer_config['callbacks'] = callbacks
    trainer_config['enable_model_summary'] = False
    
    # Use available GPUs for retraining
    available_gpus = torch.cuda.device_count()
    trainer_config['devices'] = min(available_gpus, trainer_config.get('devices', 1))
    print(f"Using {trainer_config['devices']} GPUs for retraining (available: {available_gpus}, config requested: {lightning_config.get('trainer', {}).get('devices', 'unknown')})")

    trainer = pl.Trainer(**trainer_config)
    
    t0 = time.perf_counter()
    # if retained model checkpoint exists, load it
    ckpt_path = os.path.join(cb_config['params']['dirpath'], 'last-v1.ckpt')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print(f"Loading existing checkpoint from {ckpt_path}")
        reference_model.load_state_dict(checkpoint['state_dict'], strict=False)
    elif os.path.exists(ckpt_path.replace('last-v1.ckpt', 'last.ckpt')):
        checkpoint = torch.load(ckpt_path.replace('last-v1.ckpt', 'last.ckpt'), map_location='cpu')
        print(f"Loading existing checkpoint from {ckpt_path.replace('last-v1.ckpt', 'last.ckpt')}")
        reference_model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print(f"No existing checkpoint found in {ckpt_path}, retraining from scratch...")
        trainer.fit(reference_model, datamodule=rest_datamodule)
    t1 = time.perf_counter()
    training_time = t1 - t0
    print(f"Training completed in {training_time:.2f} seconds.")
    spm_metrics = get_predict(model, datamodule, device, knn=knn, k=k, cluster_portion=1.0, ours=True)
    del model
    if acc_v2: # Show our proposed metrics
        # Evaluate reference model on rest test data
        metrics = get_predict(reference_model, datamodule, device, knn=knn, k=k, cluster_portion=1.0, ours=True)
        ModelClass = type(reference_model)
        # Calculate cross entorpy of predictions
        predictions = metrics['preds']
        spm_predictions = spm_metrics['preds']

        ce = F.cross_entropy(
            F.one_hot(spm_predictions.argmax(dim=1), num_classes=datamodule.num_classes).float(),
            F.one_hot(predictions.argmax(dim=1), num_classes=datamodule.num_classes).float(),
        ).item()
        kl = F.kl_div(
            spm_predictions.log_softmax(dim=1),
            predictions.softmax(dim=1),
            reduction="batchmean",
        ).item()
        # view retrained model as the ground truth
        agreement_rate = (spm_predictions.argmax(dim=1) == predictions.argmax(dim=1)).float().mean().item() * 100

        print(f"CE: {ce:.4f}")
        print(f"KL: {kl:.4f}")
        print(f"Agreement rate: {agreement_rate:.4f}")

        
    unlearn_metric = evaluate_unlearn(
        reference_model, datamodule, k, device=device, knn=knn, ablation=ablation, save_dir=save_dir, use_random_sampling=use_random_sampling
    ) 
    all_merics = {
        'spm_model_acc': spm_metrics['test_acc'] if acc_v2 else None,
        'spm_inference_time': spm_metrics['inference_time'] if acc_v2 else None,
        'retrain_spm_model_acc': metrics['test_acc'] if acc_v2 else None,
        'retrain_spm_inference_time': metrics['inference_time'] if acc_v2 else None,
        'ce': ce if acc_v2 else None,
        'kl': kl if acc_v2 else None,
        'agreement_rate': agreement_rate if acc_v2 else None,
        'retrain_training_time': training_time,
        'retain_spm_unl_train_acc_unlearn': unlearn_metric['train_acc_unlearn'],
        'retain_spm_unl_train_acc_rest': unlearn_metric['train_acc_rest'],
        'retain_spm_unl_test_acc_rest': unlearn_metric['test_acc_rest'],
        'retain_spm_unl_inference_time': unlearn_metric['inference_time'],
    }

    return all_merics


def get_predict(model, datamodule, device, knn=False, k=None, cluster_portion=1.0, ours=False):
    device = next(model.parameters()).device
    model.to(device)
    model.eval()
    

    if cluster_portion < 1.0:
        # Random smaple a portion of support data for clustering
        total_support = len(datamodule.support_loader().dataset)
        sample_size = int(total_support * cluster_portion)
        indices = np.random.choice(total_support, sample_size, replace=False)
        sampled_support = torch.utils.data.Subset(datamodule.support_loader().dataset, indices)
        sampled_loader = torch.utils.data.DataLoader(sampled_support, batch_size=64, shuffle=False)
        datamodule.support_loader = lambda: sampled_loader
        print(f'Building aggregated features for support data ({len(datamodule.support_loader().dataset)} samples)...')

    
    print(f'Building aggregated features for support data ({len(datamodule.support_loader().dataset)} samples)...')
    start_index_time = time.perf_counter()
    mean_vector = {}
    each_class_count = {}
    with torch.no_grad():
        for support_imgs, support_labels in tqdm(datamodule.support_loader(), desc="Building aggregated features", leave=False):
            support_imgs = support_imgs.to(device, dtype=model.dtype)
            support_emb = model.encoder(support_imgs)  # (B, D)
            for i in range(support_labels.size(0)):
                label = int(support_labels[i])
                emb = support_emb[i].detach().cpu()
                if label not in mean_vector:
                    each_class_count[label] = 1
                    mean_vector[label] = emb.clone()
                else:
                    each_class_count[label] += 1
                    mean_vector[label] += (emb - mean_vector[label]) / each_class_count[label]
    support_embeddings = torch.stack(list(mean_vector.values()), dim=0).to(device)  # (N, D)
    support_labels = torch.tensor(list(mean_vector.keys()), device=device)  # (N,)
    build_index_time = time.perf_counter() - start_index_time
    print(f"Built support data with {len(support_embeddings)} unique classes in {build_index_time:.2f} seconds.")

    correct = 0
    total = 0
    start_time = time.perf_counter()
    all_preds = []
    test_loader = datamodule.test_dataloader()
    print(f"Evaluating {len(test_loader.dataset)} test samples...")
    with torch.no_grad():
        for query_imgs, query_labels in tqdm(test_loader, desc="Evaluating test data", leave=False):
            query_imgs = query_imgs.to(device)
            q_emb = model.encoder(query_imgs)  # (B, D)
            moe_out = model.moe(q_emb, support_embeddings, support_labels)

            logits = torch.log(moe_out.clamp(min=1e-8))
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(logits.cpu())
            correct += (preds == query_labels.to(device)).sum().item()
            total += query_labels.size(0)
    inference_time = time.perf_counter() - start_time
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        'test_acc': accuracy,
        'inference_time': inference_time,
        'build_index_time': build_index_time,
        'preds': torch.cat(all_preds, dim=0)
    }
    return metrics
