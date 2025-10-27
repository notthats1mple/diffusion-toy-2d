# 🧠 Toy Diffusion Model (2D)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

*A minimal working example of a diffusion model implemented from scratch with PyTorch.*

![Training Loss Curve](assets/loss_curve.png)
![Denoising Step](assets/denoise_step.png)
一个**最小但工程化**的扩散模型示例：在 2D Swissroll 上训练一个去噪器 (denoiser)，
学习从 `xσ = x0 + σ·ε` 预测噪声 `ε̂(xσ, σ)`，并可视化**一步去噪**与**训练曲线**。

> 特色：Fail-Fast 冒烟测试、GPU 自检、加权 MSE、余弦学习率、梯度裁剪、Checkpoint、可视化。

---

## 🧠 背景与直觉
在 manifold 假设下，加噪相当于**正交扰动**，去噪等价于**向流形投影的一步梯度下降**。
本仓库通过 2D 螺旋点集把这件事“可视化地跑通”。

- 训练目标：  
  给 `xσ, σ` 预测噪声 `ε̂`，最小化 **加权 MSE**  
  `w(σ)=1/(σ^2+ε0)`，让**小噪声（更接近 x0）更重要**。
- 一步去噪：`x_denoised = xσ - σ·ε̂(xσ,σ)`。

---

## 🚀 快速开始

### 1) 安装依赖
```bash
# 方式 A：pip
pip install -r requirements.txt

# 方式 B：conda（推荐）
conda env create -f environment.yml
conda activate diffusion-toy-2d
