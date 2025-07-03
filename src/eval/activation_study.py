import argparse
import os
import shutil
from collections import OrderedDict
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from dadapy.data import Data

from src.models.llla_model import LaplaceApproxModel
from src.utils.data import get_llla_dataloader
from src.utils.environment import (
    get_device,
    load_activations_by_timestep,
    load_pretrained_model,
)
from src.utils.plots import plot_uncertainty_metric


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Laplace Approximation Sampling with Uncertainty Visualization"
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help="Number of samples to generate per class",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-name", type=str, default="unet", help="Model name to use from registry"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["diffusion", "flow"],
        default="diffusion",
        help="Method type",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="FashionMNIST", help="Dataset name"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Sample batch size")

    parser.add_argument(
        "--slice-start", type=int, default=0, help="Start index for image slice"
    )
    parser.add_argument(
        "--slice-end", type=int, default=2, help="End index for image slice"
    )
    parser.add_argument(
        "--cov-samples",
        type=int,
        default=100,
        help="Number of samples for covariance estimation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of generation steps (diffusion/flow)",
    )

    return parser.parse_args()


def estimate_intrinsic_dimensionality(data_np, verbose=True):
    data = Data(data_np)
    data.compute_distances(maxk=10)
    id_est, id_err, avg_rs = data.compute_id_2NN()
    if verbose:
        print(
            f"Estimated intrinsic dimension: {id_est:.2f} ± {id_err:.2f}, avg NN distance: {avg_rs:.4f}"
        )
    return id_est


def main():
    args = parse_args()
    device = get_device()
    num_classes = 10

    model_kwargs = {
        "num_classes": num_classes,
        "time_embedding_type": "mlp" if args.method == "flow" else "sinusoidal",
    }

    # Load pretrained model
    model = load_pretrained_model(
        model_name=args.model_name,
        ckpt_path=args.ckpt,
        device=device,
        model_kwargs=model_kwargs,
        use_wandb=True,
    )

    # Data loader
    train_loader, _ = get_llla_dataloader(
        batch_size=args.batch_size, mode=args.method, dataset_name=args.dataset_name
    )

    # Config for MNIST or custom dataset
    config = SimpleNamespace()
    config.data = SimpleNamespace(image_size=28)

    # Fit Laplace approximation model
    laplace_model = LaplaceApproxModel(model, train_loader, args=None, config=config)

    # Choose generative method
    if args.method == "diffusion":
        from src.models.diffusion import UQDiffusion

        method_instance = UQDiffusion(img_size=config.data.image_size, device=device)
    else:
        from src.models.flow import UQFlowMatching

        method_instance = UQFlowMatching(img_size=config.data.image_size, device=device)

    activations_dir = "activations"
    if os.path.exists(activations_dir):
        shutil.rmtree(activations_dir)
    os.makedirs(activations_dir)

    y = torch.arange(num_classes, device=device).repeat_interleave(
        args.samples_per_class
    )

    all_samples_grouped, uncertainties = method_instance.sample_with_uncertainty(
        model=laplace_model,
        y=y,
        cov_num_sample=args.cov_samples,
        num_steps=args.steps,
        log_intermediate=True,
        log_activations=True,
    )

    # Load activations grouped by timestep
    activations_by_timestep = load_activations_by_timestep(activations_dir)

    id_values = []
    timesteps = []

    timestep_keys = sorted(activations_by_timestep.keys())
    if args.method == "diffusion":
        timestep_keys = list(reversed(timestep_keys))

    for i, t in enumerate(timestep_keys):
        activations = activations_by_timestep[t]
        id_est = estimate_intrinsic_dimensionality(
            activations.cpu().numpy(), verbose=False
        )
        id_values.append(id_est)
        timesteps.append(i / (len(timestep_keys) - 1))  # normalized [0, 1]
        print(f"Normalized Timestep {i}: Intrinsic Dimensionality = {id_est:.2f}")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(timesteps, id_values, marker="o")
    plt.xlabel("Normalized Time (0=start, 1=end)")
    plt.ylabel("Estimated Intrinsic Dimensionality")
    plt.title(f"Intrinsic Dimensionality over Time ({args.method})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/intrinsic_dim_vs_time_{args.method}.png")


if __name__ == "__main__":
    main()
