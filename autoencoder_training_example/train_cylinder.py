"""Single-process laptop demo of the cylinder metric-preserving autoencoder.

The loss and the MLP layout are the ones in
``autoencoder_training_example_appendix_B.py``. This file does not use DDP.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import jacrev, vmap
from torch.utils.data import DataLoader, TensorDataset

# Match cylinder_riemann.pth. Must run before any tensor is created.
torch.set_default_dtype(torch.float32)

# Short laptop default. The cluster script uses 10000.
N_EPOCHS: int = 20
BATCH_SIZE: int = 256

INPUT_DIM: int = 3
HIDDEN_DIM: int = 64
LATENT_DIM: int = 2
N_LAYERS: int = 4

LEARNING_RATE: float = 1.0e-3
WEIGHT_DECAY: float = 1.0e-6
LOSS_ALPHA: float = 1.0
LOSS_BETA: float = 1.0
SEED: int = 42

N_TRAIN_EXPECTED: int = 4111
N_VALID_EXPECTED: int = 4081

TRAIN_DATA_NAME: str = "cylinder_train_data.pt"
VALID_DATA_NAME: str = "cylinder_valid_data.pt"
DEMO_CHECKPOINT_NAME: str = "cylinder_riemann_demo.pth"
FINISHED_CHECKPOINT_NAME: str = "cylinder_riemann.pth"


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
) -> nn.Sequential:
    """Linear, GELU, then ``layers`` hidden Linear-GELU blocks, then Linear.

    Each activation is a new ``nn.GELU`` instance. ``nn.Sequential`` rejects
    the same module object if it is inserted more than once.
    """
    modules: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
    ]
    for _ in range(layers):
        modules.append(nn.Linear(hidden_dim, hidden_dim))
        modules.append(nn.GELU())
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        layers: int = N_LAYERS,
    ) -> None:
        super().__init__()
        self.encoder = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            layers=layers,
        )
        self.decoder = build_mlp(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            layers=layers,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded: torch.Tensor = self.encoder(x)
        decoded: torch.Tensor = self.decoder(encoded)
        return decoded, encoded


def _induced_metrics(
    decoder: nn.Module,
    latent_points: torch.Tensor,
) -> torch.Tensor:
    """Batched J^T J. One jacrev per latent point, vmap over the batch."""

    def metric_one(e_point: torch.Tensor) -> torch.Tensor:
        def decode_at(z: torch.Tensor) -> torch.Tensor:
            decoded: torch.Tensor = decoder(z)
            return decoded

        jacobian_j: torch.Tensor = jacrev(decode_at)(e_point)
        metric: torch.Tensor = torch.einsum("Ai,Aj->ij", jacobian_j, jacobian_j)
        return metric

    metrics: torch.Tensor = vmap(
        metric_one, in_dims=0, randomness="same"
    )(latent_points)
    return metrics


class CustomLoss_Jacobian_at_p(nn.Module):
    """Reconstruction MSE plus mean Frobenius norm of (J^T J - I).

    ``J`` is the per-sample 3x2 Jacobian of the decoder at the encoded point,
    vectorized with ``jacrev`` and ``vmap``. The training pass keeps grad
    enabled, so ``backward`` reaches the decoder and the encoder through ``J``.
    The validation pass does not call ``backward``.
    """

    def __init__(self, alpha: float = LOSS_ALPHA, beta: float = LOSS_BETA) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        decoder: nn.Module,
        latent_point: torch.Tensor,
        metric_target: torch.Tensor,
        npts: int,
        create_graph: bool = True,
    ) -> torch.Tensor:
        reconstruction_loss: torch.Tensor = self.mse_loss(y_pred, y_true)
        if int(latent_point.shape[0]) != npts:
            raise RuntimeError(
                f"latent batch {int(latent_point.shape[0])} != npts {npts}."
            )
        # Per-point J, then vmap over the batch. No cross-sample blocks.
        # jacrev stays on the autograd graph whenever grad is enabled,
        # which is the training pass. create_graph is kept for the callers.
        if create_graph and not torch.is_grad_enabled():
            raise RuntimeError(
                "Training Jacobian requested with grad disabled."
            )
        metric: torch.Tensor = _induced_metrics(decoder, latent_point)
        j_loss: torch.Tensor = torch.mean(
            torch.norm(metric - metric_target, p="fro", dim=(1, 2))
        )

        combined_loss: torch.Tensor = (
            self.alpha * reconstruction_loss + self.beta * j_loss
        )
        return combined_loss


def _load_points(path: str, expected_shape: tuple[int, int]) -> torch.Tensor:
    loaded: torch.Tensor = torch.load(path, map_location="cpu")
    points: torch.Tensor = loaded.to(dtype=torch.float32).contiguous()
    if tuple(points.shape) != expected_shape:
        raise ValueError(
            f"Expected {path} to have shape {expected_shape}, got {tuple(points.shape)}."
        )
    return points


def _flat_metric(n_points: int) -> torch.Tensor:
    metric: torch.Tensor = (
        torch.eye(LATENT_DIM, dtype=torch.float32).unsqueeze(0).repeat(n_points, 1, 1)
    )
    return metric.contiguous()


def _assert_float32(module: nn.Module, label: str) -> None:
    for name, param in module.named_parameters():
        if param.dtype != torch.float32:
            raise RuntimeError(
                f"{label} parameter '{name}' is {param.dtype}, expected float32."
            )


def main() -> None:
    torch.manual_seed(SEED)

    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    train_path: str = os.path.join(script_dir, TRAIN_DATA_NAME)
    valid_path: str = os.path.join(script_dir, VALID_DATA_NAME)
    demo_checkpoint_path: str = os.path.join(script_dir, DEMO_CHECKPOINT_NAME)
    finished_checkpoint_path: str = os.path.join(script_dir, "paper_training_run", FINISHED_CHECKPOINT_NAME)

    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    train_data: torch.Tensor = _load_points(
        train_path, (N_TRAIN_EXPECTED, INPUT_DIM)
    )
    valid_data: torch.Tensor = _load_points(
        valid_path, (N_VALID_EXPECTED, INPUT_DIM)
    )
    n_train: int = int(train_data.shape[0])
    n_valid: int = int(valid_data.shape[0])
    train_metric: torch.Tensor = _flat_metric(n_train)
    valid_metric: torch.Tensor = _flat_metric(n_valid)

    train_loader: DataLoader = DataLoader(
        TensorDataset(train_data, train_metric),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    valid_loader: DataLoader = DataLoader(
        TensorDataset(valid_data, valid_metric),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model: Autoencoder = Autoencoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        layers=N_LAYERS,
    )
    model = model.to(device)
    _assert_float32(model, "training model")

    criterion: CustomLoss_Jacobian_at_p = CustomLoss_Jacobian_at_p(
        alpha=LOSS_ALPHA, beta=LOSS_BETA
    ).to(device)
    optimizer: optim.AdamW = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.99995, patience=50
    )

    print(f"Device: {device}", flush=True)
    print(f"Train points: {n_train} | Valid points: {n_valid}", flush=True)
    print(f"Epochs: {N_EPOCHS} (cluster script uses 10000)", flush=True)

    best_val_loss: float = float("inf")

    for epoch in range(N_EPOCHS):
        model.train()
        total_train_loss: float = 0.0

        for train_batch, metric_batch_train in train_loader:
            train_batch = train_batch.to(device)
            metric_batch_train = metric_batch_train.to(device)
            npts_train: int = int(train_batch.shape[0])

            optimizer.zero_grad()
            decoded_train, encoded_train = model(train_batch)
            loss: torch.Tensor = criterion(
                decoded_train,
                train_batch,
                model.decoder,
                encoded_train,
                metric_batch_train,
                npts_train,
                create_graph=True,
            )
            total_train_loss += float(loss.item())
            loss.backward()
            optimizer.step()

        avg_train_loss: float = total_train_loss / float(len(train_loader))

        model.eval()
        total_val_loss: float = 0.0
        with torch.no_grad():
            for valid_batch, metric_batch_valid in valid_loader:
                valid_batch = valid_batch.to(device)
                metric_batch_valid = metric_batch_valid.to(device)
                npts_valid: int = int(valid_batch.shape[0])

                decoded_valid, encoded_valid = model(valid_batch)
                # Jacobian still needs autograd. create_graph stays False:
                # this pass must not train.
                with torch.enable_grad():
                    val_loss: torch.Tensor = criterion(
                        decoded_valid,
                        valid_batch,
                        model.decoder,
                        encoded_valid,
                        metric_batch_valid,
                        npts_valid,
                        create_graph=False,
                    )
                total_val_loss += float(val_loss.item())

        avg_val_loss: float = total_val_loss / float(len(valid_loader))
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{N_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.16f} | "
            f"Validation Loss: {avg_val_loss:.16f}",
            flush=True,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), demo_checkpoint_path)
            print(
                f"New best validation loss: {best_val_loss:.16f}. "
                f"Saved {DEMO_CHECKPOINT_NAME}",
                flush=True,
            )

    finished_model: Autoencoder = Autoencoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        layers=N_LAYERS,
    )
    finished_state = torch.load(finished_checkpoint_path, map_location="cpu")
    finished_model.load_state_dict(finished_state)
    finished_model = finished_model.to(device)
    finished_model.eval()
    _assert_float32(finished_model, "finished checkpoint")

    valid_points: torch.Tensor = valid_data.to(device)
    with torch.no_grad():
        decoded_finished, _encoded_finished = finished_model(valid_points)
        finished_mse: torch.Tensor = nn.MSELoss()(decoded_finished, valid_points)
    print(
        "Finished checkpoint reconstruction MSE "
        f"({FINISHED_CHECKPOINT_NAME}): {finished_mse.item():.16f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
