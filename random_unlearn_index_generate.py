import random
import argparse
import os

def generate_random_unlearn_indices(dataset_size, unlearn_ratio, seed=42):
    num_unlearn = int(dataset_size * unlearn_ratio)
    random.seed(seed)
    unlearn_indices = random.sample(range(dataset_size), num_unlearn)
    return unlearn_indices

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Generate random unlearn indices for a dataset.")
    
    parser.add_argument("--ratio", type=float, default=0.5, 
                        help="The ratio of the dataset to unlearn (0.0 to 1.0).")
    parser.add_argument("--outpath", type=str, default="random_unlearn_idx", 
                        help="The directory where the output file will be saved.")
    parser.add_argument("--size", type=int, default=50000, 
                        help="Total size of the dataset.")
    
    args = parser.parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)
        print(f"Created directory: {args.outpath}")

    all_unlearn_indices = []
    
    # Generate indices across 10 seeds
    for seed in range(10):
        indices = generate_random_unlearn_indices(args.size, args.ratio, seed)
        all_unlearn_indices.append(indices)

    # Save to a txt file
    filename = f"random_unlearn_indices_{int(args.ratio * 100)}.txt"
    full_path = os.path.join(args.outpath, filename)
    
    with open(full_path, "w") as f:
        for indices in all_unlearn_indices:
            f.write(",".join(map(str, indices)) + "\n")
            
    print(f"Successfully saved indices to {full_path}")

if __name__ == "__main__":
    main()