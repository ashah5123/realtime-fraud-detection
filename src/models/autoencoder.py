"""
Autoencoder for fraud detection (reconstruction-based anomaly detection).
Train on non-fraud only; score = normalized reconstruction error (high = likely fraud).
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/model_config.yaml"
DEFAULT_ARTIFACT_PATH = "models/artifacts/autoencoder.pt"


def _load_ae_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load autoencoder section from model config YAML."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config not found at %s; using defaults", path)
        return {}
    with open(path) as f:
        full = yaml.safe_load(f) or {}
    return full.get("autoencoder", {})


class _AutoencoderModule(nn.Module):
    """
    MLP autoencoder: encoder_dims and decoder_dims from config.
    BatchNorm + Dropout between layers, ReLU, sigmoid on output.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int],
        decoder_dims: list[int],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        # Encoder: input -> e1 -> e2 -> ... -> latent
        enc_layers: list[nn.Module] = []
        dims = [input_dim] + list(encoder_dims)
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            enc_layers.append(nn.BatchNorm1d(dims[i + 1]))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)
        # Decoder: latent -> d1 -> d2 -> ... -> input
        dec_layers: list[nn.Module] = []
        dims_dec = list(decoder_dims) + [input_dim]
        for i in range(len(dims_dec) - 1):
            dec_layers.append(nn.Linear(dims_dec[i], dims_dec[i + 1]))
            if i < len(dims_dec) - 2:
                dec_layers.append(nn.BatchNorm1d(dims_dec[i + 1]))
                dec_layers.append(nn.ReLU())
                dec_layers.append(nn.Dropout(dropout))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class FraudAutoencoder:
    """
    Autoencoder for fraud detection. Trains on non-fraud only; anomaly score
    is normalized reconstruction error (higher = more likely fraud).
    """

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
        device: str | None = None,
    ) -> None:
        """
        Initialize from config. input_dim is set at fit() from data.

        Args:
            config_path: Path to model_config.yaml.
            artifact_path: Path to save/load .pt artifact.
            device: torch device (default: cuda if available else cpu).
        """
        self.artifact_path = Path(artifact_path)
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = _load_ae_config(config_path)
        self._encoder_dims = list(cfg.get("encoder_dims", [128, 64, 32]))
        self._decoder_dims = list(cfg.get("decoder_dims", [32, 64, 128]))
        self._dropout = float(cfg.get("dropout", 0.2))
        self._learning_rate = float(cfg.get("learning_rate", 0.001))
        self._batch_size = int(cfg.get("batch_size", 256))
        self._epochs = int(cfg.get("epochs", 50))
        self._patience = int(cfg.get("early_stopping_patience", 5))
        self._input_dim: int | None = None
        self._model: _AutoencoderModule | None = None
        self._input_scaler = MinMaxScaler(feature_range=(0, 1))
        self._score_p0: float = 0.0
        self._score_p99: float = 1.0
        self._fitted_ = False

    def _build_model(self, input_dim: int) -> _AutoencoderModule:
        """Build the PyTorch module given input dimension."""
        return _AutoencoderModule(
            input_dim=input_dim,
            encoder_dims=self._encoder_dims,
            decoder_dims=self._decoder_dims,
            dropout=self._dropout,
        ).to(self._device)

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        X_val: np.ndarray | torch.Tensor | None = None,
        y_val: np.ndarray | torch.Tensor | None = None,
    ) -> "FraudAutoencoder":
        """
        Train on non-fraud transactions only. Uses internal 80/20 val split if X_val not provided.

        Args:
            X: Feature matrix.
            y: Labels (0 = non-fraud, 1 = fraud).
            X_val: Optional validation features (for early stopping).
            y_val: Optional validation labels (non-fraud only used for val loss).

        Returns:
            self (for chaining).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        mask = y == 0
        n_total = len(y)
        X_normal = X[mask]
        if X_normal.size == 0:
            raise ValueError("No non-fraud samples (y==0) to fit on.")
        n_normal = len(X_normal)

        # Scale inputs to [0,1] for stable reconstruction (sigmoid output)
        self._input_scaler.fit(X_normal)
        X_normal = self._input_scaler.transform(X_normal)

        self._input_dim = X_normal.shape[1]
        self._model = self._build_model(self._input_dim)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        # Validation: use provided or last 20% of normal
        if X_val is not None and y_val is not None:
            y_val = np.asarray(y_val).ravel()
            val_mask = y_val == 0
            X_val_n = self._input_scaler.transform(np.asarray(X_val, dtype=np.float64)[val_mask])
        else:
            n_val = max(1, n_normal // 5)
            X_val_n = X_normal[-n_val:]
            X_normal = X_normal[:-n_val]
            n_normal = len(X_normal)

        train_ds = TensorDataset(
            torch.from_numpy(X_normal).float(),
            torch.from_numpy(X_normal).float(),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=min(self._batch_size, len(X_normal)),
            shuffle=True,
            drop_last=len(X_normal) > self._batch_size,
        )
        val_t = torch.from_numpy(X_val_n).float().to(self._device)

        best_val_loss = float("inf")
        patience_counter = 0
        criterion = nn.MSELoss(reduction="none")

        for epoch in range(self._epochs):
            self._model.train()
            epoch_loss = 0.0
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(self._device)
                optimizer.zero_grad()
                recon = self._model(x_batch)
                loss_per_sample = criterion(recon, x_batch).mean(dim=1)
                loss = loss_per_sample.mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x_batch.size(0)
            train_loss = epoch_loss / len(X_normal)

            self._model.eval()
            with torch.no_grad():
                recon_val = self._model(val_t)
                val_loss = criterion(recon_val, val_t).mean().item()
            scheduler.step(val_loss)
            logger.info("Epoch %d  train_loss=%.6f  val_loss=%.6f", epoch + 1, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self._patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Fit score normalization: 0th and 99th percentile of reconstruction error on train normal
        self._model.eval()
        with torch.no_grad():
            train_t = torch.from_numpy(X_normal).float().to(self._device)
            recon = self._model(train_t)
            errors = (recon - train_t).pow(2).mean(dim=1).cpu().numpy()
        self._score_p0 = float(np.percentile(errors, 0))
        self._score_p99 = float(np.percentile(errors, 99))
        if self._score_p99 <= self._score_p0:
            self._score_p99 = self._score_p0 + 1e-6
        self._fitted_ = True
        logger.info(
            "Autoencoder fit on %d non-fraud samples (of %d total); score scale [p0=%.6f, p99=%.6f]",
            n_normal,
            n_total,
            self._score_p0,
            self._score_p99,
        )
        return self

    def _reconstruction_errors(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute per-sample MSE reconstruction error (input is scaled to [0,1])."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = self._input_scaler.transform(X)
        self._model.eval()
        t = torch.from_numpy(X).float().to(self._device)
        with torch.no_grad():
            recon = self._model(t)
            errors = (recon - t).pow(2).mean(dim=1).cpu().numpy()
        return errors

    def score(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Return anomaly score in [0, 1]; higher = more likely fraud (percentile-based scaling).

        Args:
            X: Feature matrix.

        Returns:
            1D array of scores in [0, 1].
        """
        if not self._fitted_ or self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        errors = self._reconstruction_errors(X)
        scaled = (errors - self._score_p0) / (self._score_p99 - self._score_p0)
        return np.clip(scaled, 0.0, 1.0).astype(np.float64)

    def predict_single(self, features: np.ndarray) -> float:
        """
        Score one transaction.

        Args:
            features: 1D feature vector.

        Returns:
            Fraud score in [0, 1].
        """
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return float(self.score(arr)[0])

    def save(self, path: str | Path | None = None) -> None:
        """
        Save model state_dict, architecture config, and score normalization params.

        Args:
            path: Override default artifact path.
        """
        if not self._fitted_ or self._model is None:
            raise ValueError("Model not fitted. Cannot save.")
        path = Path(path) if path is not None else self.artifact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "state_dict": self._model.state_dict(),
            "input_dim": self._input_dim,
            "encoder_dims": self._encoder_dims,
            "decoder_dims": self._decoder_dims,
            "dropout": self._dropout,
            "input_scaler": self._input_scaler,
            "score_p0": self._score_p0,
            "score_p99": self._score_p99,
        }
        torch.save(state, path)
        logger.info("Saved autoencoder to %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_ARTIFACT_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        device: str | None = None,
    ) -> "FraudAutoencoder":
        """
        Load a fitted autoencoder from disk.

        Args:
            path: Path to the .pt file.
            config_path: Path to model config (for other hyperparams).
            device: Device to load model onto.

        Returns:
            Loaded FraudAutoencoder instance.
        """
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(config_path=config_path, artifact_path=path, device=device)
        obj._input_dim = state["input_dim"]
        obj._encoder_dims = state["encoder_dims"]
        obj._decoder_dims = state["decoder_dims"]
        obj._dropout = state["dropout"]
        obj._input_scaler = state.get("input_scaler", MinMaxScaler(feature_range=(0, 1)))
        obj._score_p0 = state["score_p0"]
        obj._score_p99 = state["score_p99"]
        obj._model = obj._build_model(obj._input_dim)
        obj._model.load_state_dict(state["state_dict"])
        obj._model.to(obj._device)
        obj._fitted_ = True
        logger.info("Loaded autoencoder from %s", path)
        return obj


def _main() -> None:
    """Load data, split, engineer features, train on non-fraud, score val, print sample scores."""
    import logging
    from src.data.loader import load_transactions
    from src.data.splitter import TimeAwareSplitter
    from src.features.feature_store import SimpleFeatureStore

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    df = load_transactions()
    splitter = TimeAwareSplitter()
    train_df, val_df, _test_df = splitter.split(df)
    store = SimpleFeatureStore()
    store.fit(train_df)
    X_train, y_train = store.transform(train_df)
    X_val, y_val = store.transform(val_df)

    model = FraudAutoencoder()
    model.fit(X_train.values, y_train.values)
    scores_val = model.score(X_val.values)
    y_val_arr = np.asarray(y_val).ravel()

    fraud_scores = scores_val[y_val_arr == 1]
    non_fraud_scores = scores_val[y_val_arr == 0]
    print("Sample scores (validation set):")
    print(
        "  Non-fraud: mean = %.4f, min = %.4f, max = %.4f"
        % (non_fraud_scores.mean(), non_fraud_scores.min(), non_fraud_scores.max())
    )
    print(
        "  Fraud:     mean = %.4f, min = %.4f, max = %.4f"
        % (fraud_scores.mean(), fraud_scores.min(), fraud_scores.max())
    )
    print("  (higher score = more likely fraud)")


if __name__ == "__main__":
    _main()
