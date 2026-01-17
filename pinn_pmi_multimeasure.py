# ============================================================
# Multi-Measurement Inverse PINN for PMI Estimation
# Research-grade, deployable, GPU-ready
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
NUM_BODIES = 300
POINTS_PER_BODY = 5
EPOCHS = 5000
BATCH_SIZE = 128
LR = 1e-3
LAMBDA_PHYS = 15.0
WEIGHT_DECAY = 1e-5
EARLY_STOPPING = 400

# ------------------------------------------------------------
# STEP 1: Generate multi-measurement dataset
# ------------------------------------------------------------
console.print(Panel("[bold cyan]Generating multi-measurement dataset[/bold cyan]"))

records = []

for body_id in range(NUM_BODIES):
    T0 = np.random.uniform(36.5, 38.0)
    T_env = np.random.uniform(0, min(35, T0 - 0.5))
    BMI = np.random.uniform(18, 35)
    clothing = np.random.uniform(0, 1)
    airflow = np.random.choice([0, 1])

    bmi_norm = (BMI - 18) / (35 - 18)
    k = 0.075 - 0.045*bmi_norm - 0.045*clothing + 0.045*airflow
    k = np.clip(k, 0.01, 0.12)

    times = np.sort(np.random.uniform(0.5, 24, POINTS_PER_BODY))

    for t in times:
        T_body = T_env + (T0 - T_env) * np.exp(-k * t)
        T_body += np.random.uniform(-0.15, 0.15)
        T_body = max(T_body, T_env)

        records.append([
            body_id, t, T_body, T_env, BMI, clothing, airflow
        ])

df = pd.DataFrame(
    records,
    columns=["body_id", "time", "T_body", "T_env", "BMI", "clothing", "airflow"]
)

DATA_PATH = "output/multimeasure_body_cooling.csv"
df.to_csv(DATA_PATH, index=False)

console.print(f"[bold green]Dataset saved to:[/bold green] {DATA_PATH}")

# ------------------------------------------------------------
# STEP 2: Prepare tensors
# ------------------------------------------------------------
X = torch.tensor(
    df[["T_body", "T_env", "BMI", "clothing", "airflow"]].values,
    dtype=torch.float32,
    device=device
)

y = torch.tensor(
    df["time"].values,
    dtype=torch.float32,
    device=device
).view(-1, 1)

dataset = torch.utils.data.TensorDataset(X, y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ------------------------------------------------------------
# STEP 3: Model
# ------------------------------------------------------------
class MultiMeasurePINN(nn.Module):
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

model = MultiMeasurePINN().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

mse = nn.MSELoss()

# ------------------------------------------------------------
# STEP 4: Training
# ------------------------------------------------------------
console.print(Panel("[bold magenta]Training multi-measurement inverse PINN[/bold magenta]"))

best_val = float("inf")
patience = 0

with Progress(SpinnerColumn(), BarColumn(), console=console) as progress:
    task = progress.add_task("Training", total=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.clone().detach().requires_grad_(True)
            yb = yb.to(device)

            optimizer.zero_grad()

            t_pred = model(xb)
            data_loss = mse(t_pred, yb)

            T_body = xb[:, 0:1]
            T_env = xb[:, 1:2]
            BMI = xb[:, 2:3]
            clothing = xb[:, 3:4]
            airflow = xb[:, 4:5]

            bmi_norm = (BMI - 18) / (35 - 18)
            k = 0.075 - 0.045*bmi_norm - 0.045*clothing + 0.045*airflow
            k = torch.clamp(k, 0.01, 0.12)

            T_recon = T_env + (T_body - T_env) * torch.exp(-k * t_pred)

            dTdt = torch.autograd.grad(
                T_recon, t_pred,
                grad_outputs=torch.ones_like(T_recon),
                create_graph=True
            )[0]

            physics_loss = torch.mean((dTdt + k*(T_recon - T_env))**2)

            loss = data_loss + LAMBDA_PHYS * physics_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += mse(model(xb), yb).item()

        val_loss /= len(val_loader)
        progress.advance(task)

        if epoch % 200 == 0:
            console.log(
                f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), "models/pinn_pmi_multimeasure.pt")
        else:
            patience += 1
            if patience > EARLY_STOPPING:
                console.print("[bold red]Early stopping triggered[/bold red]")
                break

console.print(Panel("[bold green]Training complete — model saved[/bold green]"))
# ------------------------------------------------------------
# STEP 5: Evaluation Metrics (Forensic Accuracy)
# ------------------------------------------------------------
console.print(Panel("[bold cyan]Evaluating model accuracy[/bold cyan]"))

model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for xb, yb in val_loader:
        preds = model(xb)
        y_true.append(yb.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

y_true = np.vstack(y_true).flatten()
y_pred = np.vstack(y_pred).flatten()

errors = np.abs(y_pred - y_true)

mae = np.mean(errors)
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

acc_1h = np.mean(errors <= 1.0) * 100
acc_2h = np.mean(errors <= 2.0) * 100
acc_3h = np.mean(errors <= 3.0) * 100

console.print(f"[bold green]MAE:[/bold green] {mae:.2f} hours")
console.print(f"[bold green]RMSE:[/bold green] {rmse:.2f} hours")

console.print(f"[bold yellow]±1 hour accuracy:[/bold yellow] {acc_1h:.1f}%")
console.print(f"[bold yellow]±2 hour accuracy:[/bold yellow] {acc_2h:.1f}%")
console.print(f"[bold yellow]±3 hour accuracy:[/bold yellow] {acc_3h:.1f}%")
