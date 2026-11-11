"""Evaluate the bank-1 sub-bank-0 autoencoder on the raw subset files."""

import os

import numpy as np
import torch
import torch.nn as nn

INPUT_DIM: int = 10
HIDDEN_DIM: int = 64
LATENT_DIM: int = 2
N_LAYERS: int = 4

RAW_FILES: tuple[str, ...] = (
    "ca_bank_1_subbank_0-first_10_only_10_percent.txt",
    "phases_bank_1_subbank_0-first_10_only_10_percent.txt",
    "Vph_bank_1_subbank_0-first_10.txt",
    "mean_phase_bank_1_subbank_0.txt",
    "weights_1.txt",
    "autoencoder_b_1_sb_0.pth",
    "norm_factor_b_1_sb_0.txt",
)


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
) -> nn.Sequential:
    """Linear, a fresh GELU, ``layers`` hidden blocks, then a linear map to ``output_dim``."""
    modules: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
    ]
    for _ in range(layers):
        modules.append(nn.Linear(hidden_dim, hidden_dim))
        modules.append(nn.GELU())
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


class Autoencoder_suppress(nn.Module):
    """MLP autoencoder: input -> latent -> input, GELU hidden units."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        layers: int = N_LAYERS,
    ) -> None:
        super(Autoencoder_suppress, self).__init__()
        self.encoder: nn.Sequential = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            layers=layers,
        )
        self.decoder: nn.Sequential = build_mlp(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            layers=layers,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded: torch.Tensor = self.encoder(x)
        decoded: torch.Tensor = self.decoder(encoded)
        return decoded, encoded


def script_directory() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def require_raw_files(script_dir: str) -> None:
    """Fail if a raw input is missing. This script does not unpack archives."""
    missing: list[str] = [
        name
        for name in RAW_FILES
        if not os.path.isfile(os.path.join(script_dir, name))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing raw files in "
            + script_dir
            + ": "
            + ", ".join(missing)
        )


def main() -> None:
    torch.set_default_dtype(torch.float32)
    script_dir: str = script_directory()
    require_raw_files(script_dir)

    ca: np.ndarray = np.loadtxt(
        os.path.join(script_dir, "ca_bank_1_subbank_0-first_10_only_10_percent.txt")
    )
    phases: np.ndarray = np.loadtxt(
        os.path.join(
            script_dir,
            "phases_bank_1_subbank_0-first_10_only_10_percent.txt",
        )
    )
    Vph: np.ndarray = np.loadtxt(
        os.path.join(script_dir, "Vph_bank_1_subbank_0-first_10.txt")
    )
    mean_phase: np.ndarray = np.loadtxt(
        os.path.join(script_dir, "mean_phase_bank_1_subbank_0.txt")
    )
    wt: np.ndarray = np.loadtxt(os.path.join(script_dir, "weights_1.txt"))

    ca_unnormalized: torch.Tensor = torch.tensor(
        np.ascontiguousarray(ca[:, :INPUT_DIM], dtype=np.float64),
        dtype=torch.float32,
    )
    norm_factor_value: np.float64 = np.float64(
        np.loadtxt(os.path.join(script_dir, "norm_factor_b_1_sb_0.txt"))
    )
    norm_factor: torch.Tensor = torch.tensor(norm_factor_value, dtype=torch.float32)
    ca_normalized: torch.Tensor = ca_unnormalized / norm_factor

    model: Autoencoder_suppress = Autoencoder_suppress(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        layers=N_LAYERS,
    )
    state_dict: dict[str, torch.Tensor] = torch.load(
        os.path.join(script_dir, "autoencoder_b_1_sb_0.pth"),
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model.float()
    model.eval()

    with torch.no_grad():
        decoded_norm: torch.Tensor
        encoded: torch.Tensor
        decoded_norm, encoded = model(ca_normalized)
        ca_hat: torch.Tensor = decoded_norm * norm_factor
        n_waveforms: int = int(ca_hat.shape[0])
        mse_value: float = torch.mean((ca_hat - ca_unnormalized) ** 2).item()
        ca_hat_np: np.ndarray = ca_hat.detach().numpy()

    phases_hat: np.ndarray = ca_hat_np @ Vph[:INPUT_DIM, :] + mean_phase
    cos: np.ndarray = np.real(
        np.einsum(
            "ij,ij->i",
            np.exp(1j * phases_hat / wt) * wt,
            np.conj(np.exp(1j * phases / wt) * wt),
        )
    )
    median_overlap: float = float(np.median(cos))

    print("number of waveforms:", n_waveforms)
    print(
        "mean squared error between ca_hat and the original (unnormalized) ca[:, :10]:",
        mse_value,
    )
    print("median phase overlap:", median_overlap)


if __name__ == "__main__":
    main()
