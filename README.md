# 🧠 Death Time Estimation using Physics-Informed Neural Networks (PINNs)

This repository implements a **Physics-Informed Neural Network (PINN)** framework for estimating the **Post-Mortem Interval (PMI)** using **algor mortis (post-mortem body cooling)**.

Unlike traditional machine learning models, this system explicitly enforces **thermodynamic heat-transfer laws** during training, producing **physically consistent, robust, and generalizable PMI predictions**.

---

## 🚀 Project Highlights

- 🔬 Physics-informed learning (Newton’s Law of Cooling enforced)
- 🧪 Realistic synthetic forensic data generation
- 🧠 Inverse problem solving (temperature → time since death)
- 📊 Multi-measurement PMI estimation
- ⚡ GPU-accelerated training (CUDA)
- 🖥️ Live training visualization using **Rich**
- 📈 Research-grade evaluation metrics

---

## 📌 Problem Statement

Estimating **time since death (PMI)** from body temperature is a classical forensic problem.

Traditional approaches:
- Use simplified nomograms
- Assume constant cooling rates
- Fail under variable conditions (BMI, clothing, airflow)

This project solves PMI estimation as an **inverse heat-transfer problem** using **Physics-Informed Neural Networks**, which combine:
- Data-driven learning
- Governing physical equations

---

## 🧠 Physics Model

The model enforces Newton’s Law of Cooling:

dT/dt = -k (T - T_env)

Where:
- T = body temperature
- T_env = ambient temperature
- k = cooling constant (modulated by BMI, clothing, airflow)

A **plateau phase** is included to model early post-mortem thermal inertia, producing realistic sigmoidal cooling curves.

---

## 📂 Repository Structure

death-time-pinn/
│
├─ pinn_with_rich.py              # Full pipeline with live tracking
├─ pinn_pmi_multimeasure.py       # Best-performing multi-measurement model
├─ pinn_pmi_inverse.py            # Baseline single-measurement model
│
├─ output/
│   ├─ body_cooling_records.csv
│   └─ multimeasure_body_cooling.csv
│
├─ requirements.txt
├─ README.md
└─ .gitignore

---

## 🧪 Models Included

### 1️⃣ pinn_with_rich.py — Full Pipeline
- Synthetic data generation
- Physics-informed training
- Live progress tracking
- Visualization

### 2️⃣ pinn_pmi_multimeasure.py — Primary Model
- Uses multiple temperature measurements
- Inverse PINN to estimate PMI

Typical Performance:
- MAE ≈ 0.8 hours
- RMSE ≈ 1.5 hours
- ±1 hour accuracy ≈ 82%
- ±2 hour accuracy ≈ 92%
- ±3 hour accuracy ≈ 97%

### 3️⃣ pinn_pmi_inverse.py — Baseline
- Single-measurement inverse PINN

---

## ⚙️ Installation

git clone https://github.com/tripathy-ji/death-time-pinn.git
cd death-time-pinn

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

---

## ▶️ Running the Code

python pinn_with_rich.py
python pinn_pmi_multimeasure.py

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only**.
It is **not a certified forensic tool**.

---

## 📜 License

MIT License
