# 🧠 Toy Diffusion Model (2D)

*A minimal working example of a diffusion model implemented from scratch with PyTorch.*

![Training Loss Curve](assets/loss_curve.png)
![Denoising Step](assets/denoise_step.png)

---

## 📘 Overview
This repository provides a **clean, educational implementation** of a 2D diffusion model trained on a synthetic *Swiss Roll* dataset.  
It helps students and researchers understand the **core principles of denoising diffusion models** without heavy dependencies.

### ✨ Key Features
- Custom noise schedule (`ScheduleLogLinear`)
- Sigma embedding with sinusoidal encoding
- Simple **MLP (Multi-Layer Perceptron)** noise predictor
- **Weighted MSE loss** emphasizing small-noise precision
- Fail-fast debugging, gradient clipping, and cosine LR scheduler
- Visualization of denoising direction in 2D

---

## 🧩 File Structure
```
📦 toy_diffusion_2d/
 ┣ 📜 toy_diffusion_2d.py        # main code
 ┣ 📜 requirements.txt           # lightweight pip dependencies
 ┣ 📜 LICENSE                    # MIT license
 ┣ 📜 README.md                  # this file
 ┗ 📂 assets/                    # figures for README
    ┣ 📊 loss_curve.png
    ┗ 🌀 denoise_step.png
```

---

## ⚙️ Installation
```bash
git clone https://github.com/<yourusername>/toy_diffusion_2d.git
cd toy_diffusion_2d
pip install -r requirements.txt
```

---

## 🚀 Run
```bash
python toy_diffusion_2d.py
```

Expected outputs:
- Console training logs  
- `assets/loss_curve.png` (loss curve)  
- `assets/denoise_step.png` (one-step denoising arrows)

---

## 🧮 Core Components

### 1️⃣ Noise Schedule
```python
schedule = ScheduleLogLinear(N=400, sigma_min=0.002, sigma_max=2.0)
```
Log-spaced σ values `[0.002, 2.0]` to cover a wide noise range.

### 2️⃣ Model (MLP)
```python
model = TimeInputMLP(dim=2, hidden_dims=(128,256,256,256,128))
```
Predicts Gaussian noise `ε̂` from noisy inputs `xσ = x0 + σ·ε`.  
Input to the MLP is the concatenation `[x, embed(σ)]` where `embed(σ) = [sin(0.5·log σ), cos(0.5·log σ)]`.

### 3️⃣ Loss Function
Weighted MSE:
```text
L = E[ 1 / (σ^2 + ε0) * || ε̂ - ε ||^2 ]
```
Emphasizes **small-σ (low-noise)** cases where denoising precision matters most.

---

## 📈 Visualization
- **Loss curve** demonstrates convergence.  
- **Denoising arrows** show vectors from noisy points toward their one-step denoised estimates on the spiral manifold.

---

## 💻 Environment

| Dependency | Minimum Version |
|-----------:|:----------------|
| Python     | 3.10            |
| PyTorch    | 2.1.0           |
| NumPy      | 1.24            |
| Matplotlib | 3.7             |

Install via:
```bash
pip install -r requirements.txt
```

---

## 📜 License
Released under the **MIT License**.  
You may use, modify, and distribute this code with attribution.

---

## 👨‍🎓 Author
**Weineng Zhu**  
Biostatistics PhD Student at City University of Hong Kong

---

## 🧩 Citation
```bibtex
@misc{zhu2025toy_diffusion_2d,
  author       = {Weineng Zhu},
  title        = {Toy Diffusion Model (2D)},
  year         = {2025},
  howpublished = {\url{https://github.com/<yourusername>/toy_diffusion_2d}}
}
```

---

*Last updated: 2025-10 — maintained by Weineng Zhu*
