import argparse
import os
import torch
from torchvision import transforms
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--support_digits",
        type=str,
        default=None,
        help="Comma-separated support digits, e.g. '0,1,2'"
    )
    parser.add_argument(
        "--query_digits",
        type=str,
        required=True,
        help="Comma-separated query digits, e.g. '3,4,5'"
    )
    parser.add_argument(
        "--generated_root",
        type=str,
        required=True,
        help="Root folder where generated images are saved"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for classifier inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    test_classes = [int(d) for d in args.query_digits.split(",")]

    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_resnet20",
        pretrained=True
    ).to(args.device).eval()

    norm = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        norm
    ])

    total_correct = 0
    total_count = 0

    for cls in test_classes:
        if args.support_digits == None:
            folder = os.path.join(args.generated_root, "class_samples", f"{cls}")
        else:
            folder = os.path.join(args.generated_root, args.support_digits, f"class_{cls}")
        files = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
        correct = 0
        for i in range(0, len(files), args.batch_size):
            batch_files = files[i : i + args.batch_size]
            imgs = [
                transform(Image.open(os.path.join(folder, f)).convert("RGB"))
                for f in batch_files
            ]
            batch = torch.stack(imgs, dim=0).to(args.device)
            with torch.no_grad():
                preds = model(batch).argmax(dim=1).cpu()
            correct += (preds == cls).sum().item()
        count = len(files)
        print(f"Class {cls} accuracy: {correct/count:.4f}")
        total_correct += correct
        total_count += count

    overall = 1 - total_correct / total_count if total_count else 0
    print(f"Overall accuracy on unlearned classes: {overall:.4f}")
    # save the results to a file
    if args.support_digits == None:
        results_file = os.path.join(args.generated_root, "class_samples", "cls_accuracy.txt")
    else:
        results_file = os.path.join(args.generated_root, args.support_digits, "cls_accuracy.txt")
    with open(results_file, "w") as f:
        f.write(f"Overall accuracy on unlearned classes: {overall:.4f}\n")
        for cls in test_classes:
            f.write(f"Class {cls} accuracy: {1 - correct/count:.4f}\n")
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()