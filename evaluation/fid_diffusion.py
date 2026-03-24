import os
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from PIL import Image
from torchvision.utils import make_grid, save_image
import random
import numpy as np
import time
from tqdm import tqdm


def generate_and_save_samples_ddpm(model, datamodule, config):
    datamodule.setup(stage='test')
    support_size_eval = getattr(config, "support_size_eval", 1024)
    if hasattr(config, "seed"):
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
    # save_path = f'results/images/ddpm/{support_size_eval}/{config.support_digits}'
    save_path = config.save_path
    os.makedirs(save_path, exist_ok=True)
    model.eval()
    device = model.device
    model.n_classes = getattr(config, "num_classes", 10)
    steps = getattr(config, "steps", 50)
    eta = getattr(config, "eta", 0.0)
    guide_w = getattr(config, "guide_w", 0.3)
    support_digits = datamodule.support_digits if hasattr(datamodule, 'support_digits') else [int(digit) for digit in config.support_digits.split(',')]
    query_digits = datamodule.query_digits if hasattr(datamodule, 'query_digits') else [int(digit) for digit in config.query_digits.split(',')]



    total_samples_per_class = getattr(config, "total_samples_per_class", 5000)
    batch_size = config.batch_size  # or any value that fits in GPU memory
    num_batches = total_samples_per_class // batch_size
    if model.sample_shape is not None:
        sample_shape = torch.Size(model.sample_shape)
    else:
        img_size = getattr(config, "img_size", 32)
        sample_shape = torch.Size((3, img_size, img_size))

    def get_support_images_for_class(target_class, support_size_eval):
        if not hasattr(datamodule, '_support_indices'):
            targets = np.array(datamodule.train_dataset.targets)
            mask = np.isin(targets, list(support_digits))
            datamodule._support_indices = np.nonzero(mask)[0].tolist()
        subset = torch.utils.data.Subset(datamodule.train_dataset, datamodule._support_indices)
        if target_class in support_digits:
            subset_targets = np.array(datamodule.train_dataset.targets)
            mask = (subset_targets == target_class)
            indices = np.intersect1d(np.nonzero(mask)[0], datamodule._support_indices).tolist()
            chosen_indices = random.sample(indices, support_size_eval) if len(indices) >= support_size_eval else random.choices(indices, k=support_size_eval)
            support_images = [datamodule.train_dataset[i][0] for i in chosen_indices]
        else:
            subset_targets = np.array(datamodule.train_dataset.targets)
            mask = (subset_targets == 0) if target_class !=0 else (subset_targets == 1)
            indices = np.intersect1d(np.nonzero(mask)[0], datamodule._support_indices).tolist()
            chosen_indices = random.sample(indices, support_size_eval)
            support_images = [datamodule.train_dataset[i][0] for i in chosen_indices]
        support_labels = [target_class] * support_size_eval
        support_imgs = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        return support_imgs, support_labels

    start_time = time.perf_counter()
    support_time = 0.0
    inference_time = 0.0
    save_img_time = 0.0 
    for class_label in query_digits:
        print(f"Generating samples for class {class_label}...")
        class_save_dir = os.path.join(save_path, f"class_{class_label}")
        os.makedirs(class_save_dir, exist_ok=True)
        # Check how many images are already saved in the directory
        existing_files = os.listdir(class_save_dir)
        existing_image_count = len([f for f in existing_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # We need to generate remaining images
        remaining_images = total_samples_per_class - existing_image_count
        print(f"Already {existing_image_count} images generated for class {class_label}. Generating {remaining_images} more images.")
        
        saved_count = existing_image_count  # Start from the count of already saved images

        for batch_idx in tqdm(range(num_batches)):
            if saved_count >= total_samples_per_class:
                break  
            # Sample a new batch of support
            t1 = time.perf_counter()
            support_imgs, support_labels = get_support_images_for_class(class_label, support_size_eval)
            t2 = time.perf_counter()
            support_time += t2 - t1
            support_imgs = support_imgs.to(device)
            support_labels = support_labels.to(device)

            cond = torch.full((batch_size,), class_label, device=device)
            with torch.no_grad():
                samples = model.ema.ema_model.ddim_sample(
                    batch_size,
                    sample_shape,
                    guide_w=guide_w,
                    steps=steps,
                    eta=eta,
                    cond=cond,
                    support=support_imgs,
                    cond_support=support_labels
                )
            t3 = time.perf_counter()
            inference_time += t3 - t2
            samples = samples.cpu()
            for i in range(samples.size(0)):
                save_image(
                    samples[i],
                    os.path.join(class_save_dir, f"{saved_count + i:05d}.png"),
                    padding=0
                )
            t4 = time.perf_counter()
            save_img_time += t4 - t3
            saved_count += samples.size(0)
            # os.makedirs(os.path.join(class_save_dir, "support"), exist_ok=True)
            # for i in range(support_imgs.size(0)):
            #     save_image(
            #         support_imgs[i].cpu(),
            #         os.path.join(class_save_dir, f"support/{i:03d}.png"),
            #         padding=0
            #     )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    metrics = {
        "total_time": elapsed_time,
        "support_time": support_time,
        "inference_time": inference_time,
        "save_img_time": save_img_time}
    print(metrics)
    return metrics


from tqdm import tqdm
def compute_fid(support_digits, device, real_root, gene_root, time_metric=None):

    if isinstance(support_digits, str):
        support_digits = [int(digit) for digit in support_digits.split(',')]

    fid = FrechetInceptionDistance(feature=2048).to(device)
    # to_tensor = transforms.ToTensor()

    # update with all real images from support classes
    for class_label in tqdm(support_digits):
        real_dir = os.path.join(real_root, str(class_label)) if 'cifar' in real_root else real_root
        for fname in sorted(os.listdir(real_dir)):
            img = Image.open(os.path.join(real_dir, fname)).convert('RGB')
            # img_t = to_tensor(img).unsqueeze(0).to(device) * 2 - 1
            img_np = np.array(img)  # shape H,W,3, dtype uint8
            img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)
            fid.update(img_t, real=True)

    # update with all generated images from support classes
    for class_label in tqdm(support_digits):
        class_dir = os.path.join(gene_root, f"class_{class_label}")
        for fname in sorted(os.listdir(class_dir)):
            img = Image.open(os.path.join(class_dir, fname)).convert('RGB')
            # img_t = to_tensor(img).unsqueeze(0).to(device) * 2 - 1
            img_np = np.array(img)  # shape H,W,3, dtype uint8
            img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)
            fid.update(img_t, real=False)

    # compute and print one overall FID
    score = fid.compute()
    print(f"FID for support classes {support_digits}: {score.item():.4f}")

    # save in gene root
    fid_file = os.path.join(gene_root, 'fid_score.txt')
    with open(fid_file, 'w') as f:
        f.write(f"{score.item():.4f}\n")
        if time_metric is not None:
            f.write(f"Total generation time: {time_metric['total_time']:.2f} seconds\n")
            f.write(f"Support selection time: {time_metric['support_time']:.2f} seconds\n")
            f.write(f"Inference time: {time_metric['inference_time']:.2f} seconds\n")
            f.write(f"Image saving time: {time_metric['save_img_time']:.2f} seconds\n")
    print(f"FID score saved to {fid_file}")