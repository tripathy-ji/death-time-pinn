# ============================================================
# Inverse Physics-Informed Neural Network (PMI Estimation)
# Fully corrected, stable, Rich-instrumented
# ============================================================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn
from rich.panel import Panel

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
console = Console()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(Panel(f"[bold green]Using device:[/bold green] {device}"))

# ------------------------------------------------------------
# Hyperparameters (anti-overfitting tuned)
# ------------------------------------------------------------
EPOCHS = 4000
BATCH_SIZE = 128
LR = 1e-3
LAMBDA_PHYS = 10.0
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 300

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
console.print(Panel("[bold cyan]Loading dataset[/bold cyan]"))

df = pd.read_csv("output/body_cooling_records.csv")

X = torch.tensor(
    df[["T_body", "T_env", "BMI", "clothing", "airflow"]].values,
    dtype=torch.float32,
    device=device
)

y_time = torch.tensor(
    df["time"].values,
    dtype=torch.float32,
    device=device
).view(-1, 1)

dataset = torch.utils.data.TensorDataset(X, y_time)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ------------------------------------------------------------
# Inverse PINN Model
# ------------------------------------------------------------
class InversePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

model = InversePINN().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

mse = nn.MSELoss()

console.print(Panel("[bold green]Model initialized[/bold green]"))

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
best_val_loss = float("inf")
patience_counter = 0

console.print(Panel("[bold magenta]Training inverse PINN (PMI estimator)[/bold magenta]"))

with Progress(
    SpinnerColumn(),
    BarColumn(),
    console=console
) as progress:

    task = progress.add_task("Training", total=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.clone().detach().requires_grad_(True)
            yb = yb.to(device)

            optimizer.zero_grad()

            # Predict time since death
            t_pred = model(xb)

            # -------------------------
            # Data loss
            # -------------------------
            data_loss = mse(t_pred, yb)

            # -------------------------
            # Physics loss (CORRECT)
            # -------------------------
            T_body = xb[:, 0:1]
            T_env = xb[:, 1:2]
            BMI = xb[:, 2:3]
            clothing = xb[:, 3:4]
            airflow = xb[:, 4:5]

            bmi_norm = (BMI - 18.0) / (35.0 - 18.0)
            k = 0.075 - 0.045 * bmi_norm - 0.045 * clothing + 0.045 * airflow
            k = torch.clamp(k, 0.01, 0.12)

            # Reconstructed temperature from predicted time
            T_recon = T_env + (T_body - T_env) * torch.exp(-k * t_pred)

            # dT_recon / dt_pred
            dTdt = torch.autograd.grad(
                outputs=T_recon,
                inputs=t_pred,
                grad_outputs=torch.ones_like(T_recon),
                create_graph=True
            )[0]

            physics_loss = torch.mean((dTdt + k * (T_recon - T_env)) ** 2)

            # -------------------------
            # Total loss
            # -------------------------
            loss = data_loss + LAMBDA_PHYS * physics_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                t_pred = model(xb)
                val_loss += mse(t_pred, yb).item()

        val_loss /= len(val_loader)

        progress.advance(task)

        if epoch % 200 == 0:
            console.log(
                f"Epoch {epoch} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        # -------------------------
        # Early stopping
        # -------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > EARLY_STOPPING_PATIENCE:
                console.print("[bold red]Early stopping triggered[/bold red]")
                break

console.print(Panel("[bold green]Training complete[/bold green]"))
