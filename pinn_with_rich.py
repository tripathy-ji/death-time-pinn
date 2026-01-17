# ==========================================
# Physics-Informed Neural Network with Rich
# Full pipeline: Data Generation + Training
# ==========================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from rich.console import Console
from rich.progress import track, Progress, SpinnerColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.status import Status

# ------------------------------------------
# Rich console
# ------------------------------------------
console = Console()

# ------------------------------------------
# Device setup
# ------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(Panel(f"[bold green]Using device:[/bold green] {device}"))

# ------------------------------------------
# Paths
# ------------------------------------------
DATA_PATH = "output/body_cooling_records.csv"
os.makedirs("output", exist_ok=True)

# ------------------------------------------
# STEP 1: Generate dataset (if missing)
# ------------------------------------------
def generate_dataset(N=1000):
    console.print(Panel("[bold cyan]Generating synthetic forensic dataset[/bold cyan]"))

    data = []

    for _ in track(range(N), description="Generating samples"):
        T0 = np.random.uniform(36.5, 38.0)
        T_env = np.random.uniform(0, min(35, T0 - 0.5))
        BMI = np.random.uniform(18, 35)
        clothing = np.random.uniform(0, 1)
        airflow = np.random.choice([0, 1])
        time = np.random.uniform(0, 24)
        t_lag = np.random.uniform(0, 5)

        bmi_norm = (BMI - 18) / (35 - 18)
        k = 0.075 - 0.045 * bmi_norm - 0.045 * clothing + 0.045 * airflow
        k = np.clip(k, 0.01, 0.12)
        k_plateau = k / 10

        if time < t_lag:
            T_body = T_env + (T0 - T_env) * np.exp(-k_plateau * time)
        else:
            T_lag = T_env + (T0 - T_env) * np.exp(-k_plateau * t_lag)
            T_body = T_env + (T_lag - T_env) * np.exp(-k * (time - t_lag))

        T_body += np.random.uniform(-0.2, 0.2)
        T_body = max(T_body, T_env)

        data.append([time, T_body, T_env, BMI, clothing, airflow])

    df = pd.DataFrame(
        data,
        columns=["time", "T_body", "T_env", "BMI", "clothing", "airflow"]
    )
    df.to_csv(DATA_PATH, index=False)

    console.print(f"[bold green]Dataset saved to:[/bold green] {DATA_PATH}")
    return df

# ------------------------------------------
# Load or generate dataset
# ------------------------------------------
if not os.path.exists(DATA_PATH):
    df = generate_dataset()
else:
    console.print(Panel("[bold yellow]Dataset found — loading[/bold yellow]"))
    df = pd.read_csv(DATA_PATH)

# ------------------------------------------
# STEP 2: Convert to tensors
# ------------------------------------------
console.print(Panel("[bold cyan]Preparing tensors[/bold cyan]"))

t = torch.tensor(df["time"].values, dtype=torch.float32).view(-1, 1).to(device)
T_body = torch.tensor(df["T_body"].values, dtype=torch.float32).view(-1, 1).to(device)
T_env = torch.tensor(df["T_env"].values, dtype=torch.float32).view(-1, 1).to(device)
BMI = torch.tensor(df["BMI"].values, dtype=torch.float32).view(-1, 1).to(device)
clothing = torch.tensor(df["clothing"].values, dtype=torch.float32).view(-1, 1).to(device)
airflow = torch.tensor(df["airflow"].values, dtype=torch.float32).view(-1, 1).to(device)

t.requires_grad = True

# ------------------------------------------
# STEP 3: PINN model
# ------------------------------------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, t, T_env, BMI, clothing, airflow):
        x = torch.cat([t, T_env, BMI, clothing, airflow], dim=1)
        return self.model(x)

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

console.print(Panel("[bold green]PINN model initialized[/bold green]"))

# ------------------------------------------
# STEP 4: Training
# ------------------------------------------
epochs = 3000
lambda_phys = 1.0

console.print(Panel("[bold magenta]Training PINN[/bold magenta]"))

with Progress(
    SpinnerColumn(),
    "[progress.description]{task.description}",
    BarColumn(),
    TimeElapsedColumn(),
    console=console,
) as progress:

    task = progress.add_task("Training epochs", total=epochs)

    for epoch in range(epochs):
        optimizer.zero_grad()

        T_pred = model(t, T_env, BMI, clothing, airflow)

        data_loss = torch.mean((T_pred - T_body) ** 2)

        dTdt = torch.autograd.grad(
            T_pred,
            t,
            grad_outputs=torch.ones_like(T_pred),
            create_graph=True
        )[0]

        BMI_norm = (BMI - 18) / (35 - 18)
        k = 0.075 - 0.045 * BMI_norm - 0.045 * clothing + 0.045 * airflow
        k = torch.clamp(k, 0.01, 0.12)

        physics_loss = torch.mean((dTdt + k * (T_pred - T_env)) ** 2)

        loss = data_loss + lambda_phys * physics_loss
        loss.backward()
        optimizer.step()

        progress.advance(task)

        if epoch % 300 == 0:
            console.log(
                f"[epoch {epoch}] "
                f"total={loss.item():.5f} "
                f"data={data_loss.item():.5f} "
                f"physics={physics_loss.item():.5f}"
            )

# ------------------------------------------
# STEP 5: Visualization
# ------------------------------------------
console.print(Panel("[bold cyan]Plotting results[/bold cyan]"))

model.eval()
with torch.no_grad():
    T_hat = model(t, T_env, BMI, clothing, airflow).cpu().numpy()

plt.figure(figsize=(8, 5))
plt.scatter(df["time"], df["T_body"], s=10, label="True", alpha=0.6)
plt.scatter(df["time"], T_hat, s=10, label="Predicted", alpha=0.6)
plt.xlabel("Time since death (hours)")
plt.ylabel("Body temperature (°C)")
plt.legend()
plt.show()

console.print(Panel("[bold green]Pipeline completed successfully[/bold green]"))
