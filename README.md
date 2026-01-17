# 🧠 Physics-Informed Neural Networks for Time Since Death Estimation (Algor Mortis)

This repository presents a **Physics-Informed Neural Network (PINN)** framework for estimating **Post-Mortem Interval (PMI)** using **body cooling (algor mortis)**, integrating forensic heat-transfer physics directly into neural network training.

The project combines:
- Synthetic forensic data generation
- Newton’s Law of Cooling
- Plateau-aware post-mortem temperature models
- Deep learning with physical constraints
- GPU-accelerated training (local)

---

## 📌 Key Features

- ✅ **Physics-informed loss** enforcing heat-transfer laws  
- ✅ **Synthetic dataset generation** grounded in forensic literature  
- ✅ **Multi-measurement PMI estimation** (more accurate than single-point)  
- ✅ **Rich-based live training visualization**  
- ✅ **GPU support (CUDA)** for efficient training  
- ✅ **Research-grade evaluation metrics**

---

## 🧪 Models Included

### 1️⃣ `pinn_with_rich.py` (Main Pipeline)
- End-to-end pipeline:
  - Data generation
  - PINN training
  - Physics loss enforcement
  - Live progress tracking (Rich)
  - Visualization
- Best for understanding the full workflow

---

### 2️⃣ `pinn_pmi_multimeasure.py` (Best Model)
- Uses **multiple temperature measurements**
- Trains an **inverse PINN** to estimate PMI
- Includes:
  - Validation split
  - Early stopping
  - Quantitative accuracy metrics

**Performance (typical run):**
- MAE ≈ **0.8 hours**
- RMSE ≈ **1.5 hours**
- ±1 hour accuracy ≈ **82%**
- ±2 hour accuracy ≈ **92%**
- ±3 hour accuracy ≈ **97%**

---

### 3️⃣ `pinn_pmi_inverse.py` (Baseline)
- Single-measurement inverse PINN
- Included for comparison and methodological clarity

---

## 📂 Project Structure
death-time-pinn/
│
├─ pinn_with_rich.py
├─ pinn_pmi_multimeasure.py
├─ pinn_pmi_inverse.py
│
├─ output/
│ ├─ body_cooling_records.csv
│ └─ multimeasure_body_cooling.csv
│
├─ requirements.txt
├─ README.md
└─ .gitignore


---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/death-time-pinn.git
cd death-time-pinn

2. Create a virtual environment
python -m venv .venv

3. Activate the environment

Windows (PowerShell):

.venv\Scripts\Activate.ps1

4. Install dependencies
pip install -r requirements.txt

🚀 Running the Code
Full pipeline with live tracking:
python pinn_with_rich.py

Multi-measurement PMI model:
python pinn_pmi_multimeasure.py

🧠 Physics Model

The PINN enforces:

𝑑
𝑇
𝑑
𝑡
=
−
𝑘
(
𝑇
−
𝑇
𝑒
𝑛
𝑣
)
dt
dT
	​

=−k(T−T
env
	​

)

Where:

𝑘
k is adjusted using BMI, clothing, and airflow

A slow-decay plateau phase models early post-mortem thermal inertia

Loss = Data Loss + Physics Loss

This ensures predictions remain physically plausible, not just statistically accurate.

🧪 Dataset

Synthetic data is generated using:

Initial body temperature (36.5–38°C)

Ambient temperature (0–35°C)

BMI-based insulation

Clothing insulation factor

Airflow (convection)

Measurement noise (±0.2°C)

Datasets are saved under /output.

🧠 Why PINNs for Forensics?

Traditional ML ignores physics.
Pure physics ignores uncertainty.

PINNs combine both, making them ideal for:

Sparse measurements

Ill-posed inverse problems

Forensic time estimation

🔮 Future Work

Web-based PMI estimation tool

REST API deployment

Confidence intervals via Bayesian PINNs

Real-world forensic data validation

⚠️ Disclaimer

This project is intended for research and educational purposes only.
It is not a certified forensic tool and should not be used in legal investigations without validation.

📜 License

MIT License


---

## 5️⃣ Next steps (when you’re ready)

When you come back later, I can help you:
- Design a **professional website UI**
- Build a **PMI prediction web app**
- Deploy inference safely
- Improve the model scientifically

For now, **GitHub push is perfect**.

If you want, next message you can simply ask:
> “Tell me the exact Git commands to push this to GitHub.”

And I’ll give you those step-by-step.
