# toy_diffusion_2d.py — Fail-Fast & Stronger Version
import os, math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace
import matplotlib.pyplot as plt

# ---------- 调试：让 CUDA 错误同步化（调试期有用，稳定后可注释） ----------
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ---------- 基础设置 ----------
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(0)
print(f"[Info] Using device: {device}")

# ---------- 数据 ----------
def make_swissroll(theta_start=np.pi/2, theta_end=5*np.pi, n_points=100):
    theta = np.linspace(theta_start, theta_end, n_points)
    r = np.linspace(0.05, 1.0, n_points)
    x = r * np.cos(theta); y = r * np.sin(theta)
    pts = np.stack([x, y], axis=1).astype(np.float32)  # [N,2]
    return torch.from_numpy(pts)

# ---------- 噪声调度 ----------
class Schedule:
    def __init__(self, sigmas: torch.Tensor):
        self.sigmas = sigmas.float()
    def __len__(self): return self.sigmas.shape[0]
    def __getitem__(self, i): return self.sigmas[i]
    def sample_batch(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        idx = torch.randint(len(self), (B,), device=self.sigmas.device)
        sig = self.sigmas[idx].to(x0)
        if sig.ndim == 1:
            sig = sig.view(B, 1)  # [B] -> [B,1]，与 [B,2] 广播
        return sig

class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float, sigma_max: float, device=None):
        smin = torch.tensor(sigma_min, dtype=torch.float32)
        smax = torch.tensor(sigma_max, dtype=torch.float32)
        sigmas = torch.logspace(torch.log10(smax), torch.log10(smin), steps=N)
        if device is not None:
            sigmas = sigmas.to(device)
        super().__init__(sigmas)

# ---------- σ 的 2D 正弦嵌入 ----------
def get_sigma_embeds(sigma: torch.Tensor) -> torch.Tensor:
    s = sigma.view(-1, 1)
    s = torch.clamp(s, min=1e-12)           # 避免 log(0)
    half_log = 0.5 * torch.log(s)
    return torch.cat([torch.sin(half_log), torch.cos(half_log)], dim=1)  # [B,2]

# ---------- 模型：MLP 预测噪声 ----------
class TimeInputMLP(nn.Module):
    def __init__(self, dim: int, hidden_dims=(128, 256, 256, 256, 128)):
        super().__init__()
        dims = [dim + 2] + list(hidden_dims) + [dim]  # 输入=2(x)+2(embed)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != dim:
                layers.append(nn.SiLU())             # 稳定的激活
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        emb = get_sigma_embeds(sigma)                # [B,2]
        nn_input = torch.cat([x, emb], dim=1)        # [B, 4]
        return self.net(nn_input)                    # [B,2]

# ---------- 训练样品 ----------
def generate_train_sample(x0: torch.Tensor, schedule: Schedule):
    sigma = schedule.sample_batch(x0)                # [B,1]
    eps   = torch.randn_like(x0)                     # [B,2]
    return sigma, eps

# ---------- 断言/体检 ----------
def assert_finite(t: torch.Tensor, name: str):
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"[NaN/Inf detected] tensor='{name}', shape={tuple(t.shape)}")

def assert_shape(t: torch.Tensor, expected_last_dim: int, name: str):
    if t.dim() < 2 or t.shape[-1] != expected_last_dim:
        raise AssertionError(f"[Shape mismatch] {name}.shape={tuple(t.shape)}, expected last dim={expected_last_dim}")

# ---------- 可选：GPU 压力测试（验证监控是否能看到利用率） ----------
@torch.no_grad()
def gpu_burn(seconds=3):
    if device != "cuda":
        print("[gpu_burn] CUDA not available."); return
    x = torch.randn(8192, 8192, device="cuda")
    t0 = time.time()
    while time.time() - t0 < seconds:
        x = x @ x
    torch.cuda.synchronize()
    print("[gpu_burn] done.")

# ---------- GPU 校验打印 ----------
def print_gpu_check(x0, sigma, eps, eps_hat, model, step):
    if step == 0 and torch.cuda.is_available():
        print("[GPU-Check] model on:", next(model.parameters()).device)
        print("[GPU-Check] x0:", x0.device, "sigma:", sigma.device, "eps:", eps.device, "eps_hat:", eps_hat.device)
        print("[GPU-Check] CUDA:", torch.version.cuda, "| cuDNN:", torch.backends.cudnn.version())
        print("[GPU-Check] device name:", torch.cuda.get_device_name(0))

# ---------- 噪声加权的 MSE ----------
def mse_weighted(eps_hat: torch.Tensor, eps: torch.Tensor, sigma: torch.Tensor, eps2=1e-4):
    """
    w = 1 / (sigma^2 + eps2). 小噪声更重要，回归更精细。
    """
    w = 1.0 / (sigma.view(-1) ** 2 + eps2)           # [B]
    err2 = (eps_hat - eps) ** 2                      # [B,2]
    loss = (w * err2.sum(dim=1)).mean()              # 标量
    return loss

# ---------- 冒烟测试（几步就能炸出大多数问题） ----------
@torch.no_grad()
def smoke_test(model, loader, schedule, steps=3):
    model.train()
    seen = 0
    for x0 in loader:
        x0 = x0.to(device)
        sigma, eps = generate_train_sample(x0, schedule)
        assert_shape(x0, 2, "x0"); assert_shape(eps, 2, "eps")
        assert_finite(x0, "x0"); assert_finite(eps, "eps"); assert_finite(sigma, "sigma")
        x_sigma = x0 + sigma * eps
        eps_hat = model(x_sigma, sigma)
        assert_shape(eps_hat, 2, "eps_hat"); assert_finite(eps_hat, "eps_hat")
        seen += 1
        if seen >= steps: break
    print(f"[SmokeTest] passed {seen} mini-steps ✅")

# ---------- 训练循环（余弦 LR、梯度裁剪、Fail-Fast、可早停） ----------
def training_loop(loader, model, schedule, epochs=20000, lr=3e-4, weight_decay=1e-4,
                  clip_grad=1.0, use_weighted_mse=True,
                  max_steps=None, log_every=50, ckpt_every=2000, ckpt_path="ckpt.pt",
                  debug=False):
    if debug:
        torch.autograd.set_detect_anomaly(True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, math.ceil(len(loader.dataset) / loader.batch_size))
    total_steps = (epochs * steps_per_epoch) if max_steps is None else max(max_steps, 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr * 0.1)

    step = 0
    model.train()

    for epoch in range(epochs):
        for x0 in loader:                             # x0: [B,2]
            x0 = x0.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            sigma, eps = generate_train_sample(x0, schedule)      # [B,1], [B,2]
            assert_finite(x0, "x0"); assert_finite(eps, "eps"); assert_finite(sigma, "sigma")

            x_sigma = x0 + sigma * eps
            eps_hat = model(x_sigma, sigma)                       # [B,2]
            loss = mse_weighted(eps_hat, eps, sigma) if use_weighted_mse \
                   else ((eps_hat - eps) ** 2).mean()

            if not torch.isfinite(loss):
                raise FloatingPointError(f"[NaN loss] step={step}, epoch={epoch}")

            # 第一步打印 GPU 校验
            print_gpu_check(x0, sigma, eps, eps_hat, model, step)

            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            opt.step()
            scheduler.step()

            # 日志
            if step % log_every == 0:
                with torch.no_grad():
                    smin = sigma.min().item(); smax = sigma.max().item()
                print(f"[step {step:6d}] loss={loss.item():.6f}  sigma∈[{smin:.4g},{smax:.4g}]  lr={scheduler.get_last_lr()[0]:.2e}")

            # 检查点
            if ckpt_every and step > 0 and step % ckpt_every == 0:
                torch.save({"model": model.state_dict(),
                            "opt": opt.state_dict(),
                            "step": step}, ckpt_path)
                print(f"[Checkpoint] saved to {ckpt_path} at step {step}")

            yield SimpleNamespace(loss=loss.detach(), step=step, epoch=epoch)
            step += 1

            # 早停
            if max_steps is not None and step >= max_steps:
                return

# ---------- 可视化 ----------
def moving_average(x, k=200):
    if k <= 1 or k > len(x): return np.array(x, dtype=float)
    w = np.ones(k, dtype=float) / k
    return np.convolve(np.asarray(x, dtype=float), w, mode="valid")

def plot_losses(losses, smooth=400, show_raw=False):
    plt.figure(figsize=(6,4))
    if show_raw and len(losses) > 0:
        plt.plot(np.arange(len(losses)), losses, alpha=0.25, label="loss (raw)")
    if smooth and smooth > 1 and len(losses) >= smooth:
        y = moving_average(losses, k=smooth)
        xs = np.arange(len(y)) + (smooth - 1) / 2.0
        plt.plot(xs, y, label=f"loss (MA{smooth})")
    else:
        plt.plot(np.arange(len(losses)), losses, label="loss")
    plt.xlabel("training step"); plt.ylabel("loss (MSE)")
    plt.title("Training loss"); plt.legend(); plt.tight_layout(); plt.show()

@torch.no_grad()
def visualize_denoising_batch(x0, x_sigma, eps_hat, sigma,
                              max_sigma_for_plot=0.5, radius=1.5, max_arrows=100):
    mask = (sigma.view(-1) <= max_sigma_for_plot)
    x0 = x0[mask].detach().cpu()
    x_sigma = x_sigma[mask].detach().cpu()
    eps_hat = eps_hat[mask].detach().cpu()
    sigma = sigma[mask].detach().cpu()

    x_denoised = x_sigma - sigma * eps_hat
    plt.figure(figsize=(5,5))
    plt.scatter(x_sigma[:,0], x_sigma[:,1], s=10, label="noisy")
    plt.scatter(x0[:,0], x0[:,1], s=10, label="clean", alpha=0.7)
    for a,b,c,d in list(zip(x_sigma[:,0], x_sigma[:,1], x_denoised[:,0], x_denoised[:,1]))[:max_arrows]:
        plt.arrow(a, b, c-a, d-b, head_width=0.02, length_includes_head=True, alpha=0.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-radius, radius); plt.ylim(-radius, radius)
    plt.title("Denoising step (σ small)")
    plt.legend(); plt.tight_layout(); plt.show()

# ---------- main ----------
def main():
    # ====== 配置区（按显卡改大/改小） ======
    n_points        = 20000
    batch_size      = 4096
    sigma_min, sigma_max, Nsig = 0.002, 2.0, 400
    hidden_dims     = (128,256,256,256,128)
    use_weighted_mse= True
    lr, wd          = 3e-4, 1e-4

    # 可选：让 GPU 曲线明显抬头 3 秒以确认监控（训练正式开始前）
    # gpu_burn(3)

    # 数据与加载器
    dataset = make_swissroll(np.pi/2, 5*np.pi, n_points)            # [N,2]
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         pin_memory=(device=="cuda"))

    # 调度与模型
    schedule = ScheduleLogLinear(N=Nsig, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    model = TimeInputMLP(dim=2, hidden_dims=hidden_dims).to(device)

    # 1) 冒烟测试：快速发现问题
    smoke_test(model, loader, schedule, steps=3)

    # 2) 预热短跑：先跑 400 步看看曲线
    warm_steps = 400
    trainer = training_loop(loader, model, schedule,
                            epochs=999999, lr=lr, weight_decay=wd,
                            clip_grad=1.0, use_weighted_mse=use_weighted_mse,
                            max_steps=warm_steps, log_every=20, ckpt_every=200,
                            ckpt_path="warm_ckpt.pt", debug=True)
    warm_losses = [ns.loss.item() for ns in trainer]
    plot_losses(warm_losses, smooth=50, show_raw=True)

    # 3) 长跑训练： 20000 步
    long_steps = 20000
    trainer = training_loop(loader, model, schedule,
                            epochs=999999, lr=lr, weight_decay=wd,
                            clip_grad=1.0, use_weighted_mse=use_weighted_mse,
                            max_steps=long_steps, log_every=50, ckpt_every=2000,
                            ckpt_path="long_ckpt.pt", debug=False)
    losses = [ns.loss.item() for ns in trainer]
    plot_losses(losses, smooth=800)

    # 画小噪声区的一步去噪
    with torch.no_grad():
        for x0 in loader:
            x0 = x0.to(device)
            sigma, eps = generate_train_sample(x0, schedule)
            x_sigma = x0 + sigma * eps
            eps_hat = model(x_sigma, sigma)
            visualize_denoising_batch(x0, x_sigma, eps_hat, sigma,
                                      max_sigma_for_plot=0.3, radius=1.6)
            break

if __name__ == "__main__":
    main()
