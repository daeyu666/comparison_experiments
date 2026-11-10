# Official SA-RFM core implementation, mirrored verbatim below this attribution.
# Source: https://github.com/ZhangQuan-hub/SA-RFM/blob/main/SB/sb.py
# Integrated as the architectural reference; adapter-specific training is in train.py.

"""
Structure-Aware Flow Matching (SAFM) for Unregistered MHIF

基于 Monge 最优传输的动态表述，通过结构诱导的黎曼几何实现结构保持的配准与融合。

核心创新点：
1. Structure-Induced Riemannian Geometry: 从 HR-MSI 构造结构张量，定义各向异性黎曼度量
2. Minimum-Cost Velocity Field: 将配准问题表述为在黎曼流形上最小化累积动能
3. Riemannian Metric-Weighted Flow Matching Loss: 在向量场回归层面施加几何约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


# ==========================================
# 1. Structure Tensor & Riemannian Metric
# ==========================================

class StructureTensorComputer(nn.Module):
    """
    从 HR-MSI 构造结构张量 S(x)，用于刻画局部空间结构。
    
    结构张量定义：
    S(x) = Σ_k (∇X_m^(k)(x) ∇X_m^(k)(x)^T) * G_σ
    
    其中 G_σ 是高斯平滑核，用于提升结构估计的鲁棒性。
    """
    def __init__(self, sigma: float = 1.0, kernel_size: int = 5):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 预计算高斯平滑核
        self._build_gaussian_kernel(kernel_size, sigma)
        
        # Sobel算子用于计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def _build_gaussian_kernel(self, kernel_size: int, sigma: float):
        """构建高斯平滑核"""
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_2d = gauss_1d.view(-1, 1) @ gauss_1d.view(1, -1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        self.register_buffer('gauss_kernel', gauss_2d.view(1, 1, kernel_size, kernel_size))
    
    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """对多通道图像应用高斯平滑"""
        B, C, H, W = x.shape
        # 对每个通道独立应用高斯核
        kernel = self.gauss_kernel.repeat(C, 1, 1, 1)
        return F.conv2d(x, kernel, padding=self.padding, groups=C)
    
    def forward(self, msi: torch.Tensor) -> torch.Tensor:
        """
        计算结构张量。
        
        Input: msi (B, C, H, W) - HR-MSI 图像
        Output: S (B, 2, 2, H, W) - 结构张量（对称正定）
        """
        B, C, H, W = msi.shape
        
        # 初始化结构张量分量
        S_xx = torch.zeros(B, 1, H, W, device=msi.device)
        S_xy = torch.zeros(B, 1, H, W, device=msi.device)
        S_yy = torch.zeros(B, 1, H, W, device=msi.device)
        
        # 对每个光谱通道计算梯度并累加
        for k in range(C):
            channel_k = msi[:, k:k+1, :, :]  # (B, 1, H, W)
            
            # 计算x和y方向的梯度
            grad_x = F.conv2d(channel_k, self.sobel_x, padding=1)
            grad_y = F.conv2d(channel_k, self.sobel_y, padding=1)
            
            # 累加外积
            S_xx = S_xx + grad_x * grad_x
            S_xy = S_xy + grad_x * grad_y
            S_yy = S_yy + grad_y * grad_y
        
        # 对结构张量分量应用高斯平滑
        S_xx = self._smooth(S_xx)
        S_xy = self._smooth(S_xy)
        S_yy = self._smooth(S_yy)
        
        # 组装结构张量 S = [[S_xx, S_xy], [S_xy, S_yy]]
        # 输出形状: (B, 2, 2, H, W)
        S = torch.stack([
            torch.cat([S_xx, S_xy], dim=1),  # 第一行 [S_xx, S_xy]
            torch.cat([S_xy, S_yy], dim=1)   # 第二行 [S_xy, S_yy]
        ], dim=1)  # (B, 2, 2, H, W)
        
        return S


class RiemannianMetricComputer(nn.Module):
    """
    基于结构张量构造黎曼度量张量。
    
    黎曼度量定义：
    M(x) = α·I + β·S(x)
    
    其中：
    - α 控制全局平滑性（各向同性基础）
    - β 决定结构约束的强度（各向异性调制）
    
    几何含义：
    - 沿高梯度方向（垂直于边缘）的运动被赋予更高代价
    - 沿结构内部（低梯度方向）的运动代价较低
    """
    def __init__(self, alpha: float = 1.0, beta: float = 10.0, sigma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.structure_tensor = StructureTensorComputer(sigma=sigma)
    
    def forward(self, msi: torch.Tensor) -> torch.Tensor:
        """
        计算黎曼度量张量。
        
        Input: msi (B, C, H, W) - HR-MSI 图像
        Output: M (B, 2, 2, H, W) - 黎曼度量张量
        """
        B, C, H, W = msi.shape
        
        # 计算结构张量
        S = self.structure_tensor(msi)  # (B, 2, 2, H, W)
        
        # 构造单位矩阵 I (B, 2, 2, H, W)
        I = torch.zeros_like(S)
        I[:, 0, 0, :, :] = 1.0
        I[:, 1, 1, :, :] = 1.0
        
        # 黎曼度量: M = α·I + β·S
        M = self.alpha * I + self.beta * S
        
        # 确保正定性：添加小的对角正则化
        eps = 1e-4
        M[:, 0, 0, :, :] = M[:, 0, 0, :, :] + eps
        M[:, 1, 1, :, :] = M[:, 1, 1, :, :] + eps
        
        return M


# ==========================================  
# 2. Time Embedding & Network Components
# ==========================================

class TimeEmbedding(nn.Module):
    """正弦位置编码的时间嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)
        self.act = nn.Mish()

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        emb = self.act(self.linear_1(emb))
        emb = self.linear_2(emb)
        return emb



# ==========================================
# 3. Structure-Aware Flow Matching Network
# ==========================================

class ImprovedDiT(nn.Module):
    """
    Structure-Aware Flow Matching Network
    
    基于 ImprovedDiT 的轻量化实现，增加以下功能：
    1. 显式计算黎曼度量张量 M(x) 用于结构感知损失
    2. 输出速度场 v_θ(x_t, t) 用于 Flow Matching
    
    输入：
    - x_t: 当前状态 (B, C_hsi, H, W)
    - t: 时间步 (B,)
    - hsi_lr: 低分辨率 HSI (B, C_hsi, H_lr, W_lr)
    - msi: 高分辨率 MSI (B, C_msi, H, W)
    
    输出：
    - pred: 预测的速度场 v_θ (B, C_hsi, H, W)
    - metric_tensor: 黎曼度量张量 M(x) (B, 2, 2, H, W)
    """
    def __init__(self, hsi_channels=102, msi_channels=5, dim=220, 
                 alpha: float = 1.0, beta: float = 10.0, sigma: float = 1.0):
        super().__init__()
        self.dim = dim
        self.hsi_channels = hsi_channels
        
        # 黎曼度量计算器
        self.metric_computer = RiemannianMetricComputer(alpha=alpha, beta=beta, sigma=sigma)

        # Time embedding
        self.time_mlp = TimeEmbedding(dim)

        # 共享编码器 (用于 x_t 和 hsi_lr)
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(hsi_channels, dim//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim//2, dim, 3, padding=1)
        )

        # MSI 编码器
        self.msi_encoder = nn.Sequential(
            nn.Conv2d(msi_channels, dim//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim//2, dim, 3, padding=1)
        )

        # 轻量级融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),  # 1x1 conv for efficient fusion
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=max(1, dim//20)),  # Depthwise conv
            nn.Conv2d(dim, dim, 1)  # Pointwise conv
        )

        # 输出头：预测速度场
        self.final_conv = nn.Conv2d(dim, hsi_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, 
                hsi_lr: torch.Tensor, msi: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x_t.shape

        # 1. 计算黎曼度量张量 M(x) 从 MSI
        metric_tensor = self.metric_computer(msi)  # (B, 2, 2, H, W)

        # 2. Time embedding
        t_emb = self.time_mlp(t).view(B, self.dim, 1, 1).expand(-1, -1, H, W)

        # 3. 编码输入
        x_t_feat = self.shared_encoder(x_t) + t_emb  # 当前状态 + 时间

        # 上采样 LR HSI 并编码
        hsi_lr_up = F.interpolate(hsi_lr, size=(H, W), mode='bicubic', align_corners=False)
        hsi_lr_feat = self.shared_encoder(hsi_lr_up)

        # 编码 MSI
        msi_feat = self.msi_encoder(msi)

        # 4. 特征融合
        combined = torch.cat([x_t_feat, hsi_lr_feat, msi_feat], dim=1)
        fused = self.fusion(combined)

        # 5. 预测速度场
        pred = self.final_conv(fused)

        return {
            "pred": pred,
            "metric_tensor": metric_tensor,
            # 为了兼容性，同时输出 cost_matrix
            "cost_matrix": metric_tensor
        }


# ==========================================
# 4. Riemannian Flow Matching Loss
# ==========================================

def compute_spatial_gradient(img: torch.Tensor) -> torch.Tensor:
    """
    计算空间梯度 (dy, dx)。
    Input: (B, C, H, W)
    Output: (B, C, 2, H, W) -> [dx, dy]
    """
    dy = torch.zeros_like(img)
    dy[..., 1:, :] = img[..., 1:, :] - img[..., :-1, :]
    
    dx = torch.zeros_like(img)
    dx[..., :, 1:] = img[..., :, 1:] - img[..., :, :-1]
    
    return torch.stack([dx, dy], dim=2)


def riemannian_flow_matching_loss(
    v_pred: torch.Tensor, 
    u_target: torch.Tensor, 
    M: torch.Tensor,
    use_riemannian: bool = True,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Riemannian Metric-Weighted Flow Matching Loss (Eq. 10 in paper)
    
    L_RFM = E_{t, z_0, z_1} [ (v_θ(z_t, t) - u_t)^T M(z_t) (v_θ(z_t, t) - u_t) ]
    
    论文解释：
    - v_pred: 网络预测的速度场 v_θ(z_t, t)  (B, C, H, W)
    - u_target: 目标速度场 u = z_1 - z_0 = x_hr - x_lr_up  (B, C, H, W)
    - M: 黎曼度量张量 M(x) = αI + βS(x)  (B, 2, 2, H, W)
    
    关键洞察：
    M(x) 是 2x2 空间度量张量，编码图像结构信息。
    我们使用 M 的特征值来调制光谱速度场误差的空间权重：
    - 在结构显著区域（大特征值），速度预测误差受到更强惩罚
    - 在平滑区域（小特征值），允许更大的预测容忍度
    
    这样确保学习到的形变流在结构敏感区域具有极高的保真度。
    """
    # 速度场误差 (B, C, H, W)
    diff = v_pred - u_target
    
    if not use_riemannian:
        # 标准 L2 损失（各向同性，用于对比）
        sq = diff.square()
        return sq.mean() if valid_mask is None else sq[valid_mask.expand_as(sq)].mean()
    
    B, C, H, W = diff.shape
    
    # ============================================
    # 方法：使用 M 的迹作为空间自适应权重
    # ============================================
    # 
    # 理论依据：
    # M = αI + βS，其中 S 是结构张量
    # trace(M) = 2α + β·trace(S) = 2α + β(S_xx + S_yy)
    # 
    # 在边缘区域：梯度大 -> S 特征值大 -> trace(M) 大
    # 在平滑区域：梯度小 -> S 特征值小 -> trace(M) 小
    #
    # 因此 trace(M) 可以作为结构显著性的标量测度
    
    # 提取 M 的对角元素
    M_00 = M[:, 0, 0, :, :]  # (B, H, W)
    M_11 = M[:, 1, 1, :, :]  # (B, H, W)
    
    # 计算 trace(M) 作为空间权重
    trace_M = M_00 + M_11  # (B, H, W)
    
    # 归一化到 [1, 10] 范围，确保稳定性
    # trace(M) 的理论最小值是 2α（在完全平滑区域）
    support = trace_M if valid_mask is None else trace_M[valid_mask[:, 0]]
    trace_min = support.min().detach()
    trace_max = support.max().detach()
    
    if trace_max > trace_min:
        # 归一化到 [1, max_weight] 范围
        max_weight = 5.0  # 边缘区域最大权重为 5 倍
        weight = 1.0 + (max_weight - 1.0) * (trace_M - trace_min) / (trace_max - trace_min + 1e-8)
    else:
        weight = torch.ones_like(trace_M)
    
    # 扩展到 (B, 1, H, W) 以便广播
    weight = weight.unsqueeze(1)  # (B, 1, H, W)
    
    # 计算加权 L2 损失
    # L_RFM = mean( weight * ||v_pred - u_target||^2 )
    weighted_diff_sq = weight * (diff ** 2)  # (B, C, H, W)
    
    # 对所有维度求均值
    loss = weighted_diff_sq.mean() if valid_mask is None else weighted_diff_sq[valid_mask.expand_as(weighted_diff_sq)].mean()
    
    return loss


def gradient_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    梯度一致性损失（Potential-Guided Drift Correction）
    
    鼓励预测的速度场与目标速度场在空间梯度上保持一致。
    """
    pred_grad = compute_spatial_gradient(pred)  # (B, C, 2, H, W)
    target_grad = compute_spatial_gradient(target)  # (B, C, 2, H, W)
    
    diff = pred_grad - target_grad
    loss = torch.mean(diff ** 2)
    
    return loss.clamp(max=100.0)


def angle_loss(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    光谱角损失 (Spectral Angle Loss / SAM)
    
    计算预测光谱与目标光谱之间的角度差异（弧度），用于约束光谱一致性。
    
    Args:
        out: 预测的光谱图像 (B, C, H, W)
        ref: 目标光谱图像 (B, C, H, W)
    
    Returns:
        loss: 平均光谱角损失（弧度）
    """
    # 确保输入是 4D: (B, C, H, W)
    if out.dim() == 5:
        out = out.squeeze(0)
        ref = ref.squeeze(0)
    
    B, C, H, W = out.shape
    
    # 重塑为 (B, C, H*W) - 每个像素是一个C维光谱向量
    out_flat = out.view(B, C, -1)  # (B, C, N) where N = H*W
    ref_flat = ref.view(B, C, -1)  # (B, C, N)
    
    # 计算每个像素位置的光谱角
    # 点积: sum over spectral dimension
    dot_product = torch.sum(out_flat * ref_flat, dim=1)  # (B, N)
    
    # 范数: L2 norm over spectral dimension
    norm_out = torch.norm(out_flat, dim=1)  # (B, N)
    norm_ref = torch.norm(ref_flat, dim=1)  # (B, N)
    
    # 分母
    denominator = norm_out * norm_ref  # (B, N)
    
    # 避免除零
    eps = 1e-8
    valid_mask = denominator > eps
    
    if not torch.any(valid_mask):
        return torch.tensor(0.0, dtype=out.dtype, device=out.device)
    
    # 计算余弦相似度
    cos_theta = dot_product[valid_mask] / denominator[valid_mask]
    
    # 限制在 [-1, 1] 范围
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # 计算角度（弧度）
    theta = torch.acos(cos_theta)
    
    # 返回平均光谱角
    loss = torch.mean(theta)
    
    return loss


def compute_optimal_transport_direction(
    d: torch.Tensor,
    M: torch.Tensor
) -> torch.Tensor:
    """
    计算结构感知最优单步传输方向 v†(u) (Eq. 13 in paper)
    
    v†(u) = argmin_v  v^T M(u) v   s.t.  <v, d(u)> = ||d(u)||_2
    
    闭式解：
    v†(u) = (M(u)^{-1} d(u)) / (d(u)^T M(u)^{-1} d(u)) * ||d(u)||_2
    
    几何解释：
    - d(u) = s_1(u) - s_0(u) 是欧氏位移
    - v†(u) 是在黎曼度量下的最小代价方向，同时保持朝向目标的全局方向
    - 在各向同性情况下 (M = I)，v† = d
    - 在各向异性情况下，v† 会偏向低代价结构方向
    
    Args:
        d: 欧氏位移向量 d(u) = x_1 - x_0  (B, C, H, W)
        M: 黎曼度量张量 M(u) = αI + βS(u)  (B, 2, 2, H, W)
    
    Returns:
        v_opt: 最优传输方向 v†(u)  (B, C, H, W)
    
    注意：
    这里我们对每个光谱通道独立计算最优方向。
    由于 M 是 2x2 空间度量，我们计算空间梯度的最优方向，
    然后将这种结构感知性传递到整个速度场。
    """
    B, C, H, W = d.shape
    
    # 提取 M 的分量 (B, H, W)
    M_00 = M[:, 0, 0, :, :]  # M_xx
    M_01 = M[:, 0, 1, :, :]  # M_xy
    M_11 = M[:, 1, 1, :, :]  # M_yy
    
    # 计算 M 的行列式 det(M) = M_00 * M_11 - M_01^2
    det_M = M_00 * M_11 - M_01 * M_01 + 1e-8  # 添加小量确保数值稳定
    
    # 计算 M^{-1}
    # M^{-1} = (1/det) * [[M_11, -M_01], [-M_01, M_00]]
    M_inv_00 = M_11 / det_M  # (B, H, W)
    M_inv_01 = -M_01 / det_M
    M_inv_11 = M_00 / det_M
    
    # 计算 ||d||_2 (欧氏范数，对光谱维度)
    d_norm = torch.norm(d, dim=1, keepdim=True)  # (B, 1, H, W)
    d_norm = d_norm.clamp(min=1e-8)  # 避免除零
    
    # 对于光谱向量，我们使用黎曼度量来调制空间方向
    # 计算空间梯度来获取方向信息
    # d_x = ∂d/∂x, d_y = ∂d/∂y
    d_x = torch.zeros_like(d)
    d_y = torch.zeros_like(d)
    d_x[..., :, 1:] = d[..., :, 1:] - d[..., :, :-1]  # x方向梯度
    d_y[..., 1:, :] = d[..., 1:, :] - d[..., :-1, :]  # y方向梯度
    
    # 计算每个像素的 2D 空间梯度向量 (在光谱通道上平均)
    grad_x = d_x.mean(dim=1, keepdim=False)  # (B, H, W)
    grad_y = d_y.mean(dim=1, keepdim=False)  # (B, H, W)
    
    # 计算 M^{-1} * grad
    # [M_inv_00  M_inv_01] [grad_x]   [M_inv_00*grad_x + M_inv_01*grad_y]
    # [M_inv_01  M_inv_11] [grad_y] = [M_inv_01*grad_x + M_inv_11*grad_y]
    Minv_grad_x = M_inv_00 * grad_x + M_inv_01 * grad_y  # (B, H, W)
    Minv_grad_y = M_inv_01 * grad_x + M_inv_11 * grad_y  # (B, H, W)
    
    # 计算 grad^T M^{-1} grad = grad_x * Minv_grad_x + grad_y * Minv_grad_y
    grad_Minv_grad = grad_x * Minv_grad_x + grad_y * Minv_grad_y  # (B, H, W)
    grad_Minv_grad = grad_Minv_grad.clamp(min=1e-8)  # 避免除零
    
    # 计算缩放因子: ||d||_2 / (d^T M^{-1} d)
    # 这里我们用梯度来近似空间方向的调制
    scale = 1.0 / (1.0 + grad_Minv_grad.unsqueeze(1))  # (B, 1, H, W)
    
    # 最优传输方向：在欧氏位移的基础上，根据黎曼度量进行调制
    # v† 保持整体方向朝向目标，同时偏好低代价路径
    # 简化实现：使用迹的倒数作为权重调制
    trace_M = M_00 + M_11  # (B, H, W)
    trace_M_inv = 1.0 / trace_M.clamp(min=1e-4)  # (B, H, W)
    
    # 归一化权重到合理范围
    weight = trace_M_inv / (trace_M_inv.mean() + 1e-8)  # (B, H, W)
    weight = weight.unsqueeze(1)  # (B, 1, H, W)
    
    # v† = d * weight (调制后的最优方向)
    # 在高结构区域（大 trace），weight 小，更保守
    # 在低结构区域（小 trace），weight 大，更自由
    v_opt = d * weight
    
    # 重新归一化以保持整体幅度
    v_opt = v_opt * (d_norm / (torch.norm(v_opt, dim=1, keepdim=True) + 1e-8))
    
    return v_opt


def optimal_transport_direction_loss(
    v_pred: torch.Tensor,
    d: torch.Tensor,
    M: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    最优传输方向正则化损失 (Eq. 14 in paper)
    
    L_OTD = E_u [ || v_θ(u) - v†(u) ||_2^2 ]
    
    鼓励预测的速度场接近结构感知的最优传输方向。
    
    Args:
        v_pred: 网络预测的速度场 (B, C, H, W)
        d: 欧氏位移 d(u) = x_1 - x_0 (B, C, H, W)
        M: 黎曼度量张量 (B, 2, 2, H, W)
    
    Returns:
        loss: L_OTD 损失值
    """
    # 计算最优传输方向
    v_opt = compute_optimal_transport_direction(d, M)
    
    # L2 损失
    sq = (v_pred - v_opt).square()
    loss = sq.mean() if valid_mask is None else sq[valid_mask.expand_as(sq)].mean()
    
    return loss


# ==========================================
# 5. Structure-Aware Flow Matching Trainer
# ==========================================

class SA_SB_Trainer:
    """
    Structure-Aware Flow Matching Trainer
    
    实现基于 Monge 最优传输的结构感知流匹配训练。
    
    训练流程：
    1. 定义端点：x_0 = Upsample(LR-HSI), x_1 = GT HR-HSI
    2. 线性插值：x_t = (1-t) * x_0 + t * x_1
    3. 目标速度：u_t = x_1 - x_0
    4. 黎曼度量加权损失：L_RFM = E[(v_θ - u)^T M(x) (v_θ - u)]
    5. 光谱角损失：L_angle = angle_loss(x_t + v_pred, x_1)
    
    Total Loss: L_total = L_RFM + λ_grad * L_gradient + λ_angle * L_angle + λ_otd * L_OTD
    """
    def __init__(self, model: nn.Module, optimizer, device, 
                 lambda_potential: float = 0.1,
                 lambda_angle: float = 0.1,
                 lambda_otd: float = 0.1,
                 use_riemannian: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_potential = lambda_potential
        self.lambda_angle = lambda_angle
        self.lambda_otd = lambda_otd
        self.use_riemannian = use_riemannian

    def train_step(self, hsi_lr: torch.Tensor, hsi_hr: torch.Tensor, 
                   msi_hr: torch.Tensor) -> Dict[str, float]:
        """
        单步训练 Structure-Aware Flow Matching。
        
        Args:
            hsi_lr: 低分辨率 HSI (B, C, H_lr, W_lr)
            hsi_hr: 高分辨率 HSI Ground Truth (B, C, H_hr, W_hr)
            msi_hr: 高分辨率 MSI (B, C_msi, H_hr, W_hr)
        
        Returns:
            损失字典:
            - "rfm": Riemannian Flow Matching Loss (L_RFM)
            - "grad": Gradient Consistency Loss (L_grad)
            - "angle": Spectral Angle Loss (L_angle)
            - "otd": Optimal Transport Direction Loss (L_OTD)
            - "total": L_RFM + λ_grad * L_grad + λ_angle * L_angle + λ_otd * L_OTD
        """
        B = hsi_lr.shape[0]

        # 1. 定义端点 (Optimal Transport endpoints)
        # x_0: 上采样的 LR-HSI（起点，可能存在几何失配）
        x_0 = F.interpolate(hsi_lr, size=hsi_hr.shape[-2:], mode='bicubic', align_corners=False)
        # x_1: Ground Truth HR-HSI（终点，对齐的目标）
        x_1 = hsi_hr

        # 2. 采样时间 t ~ Uniform[0, 1]
        t = torch.rand(B, device=self.device)
        t_reshaped = t.view(B, 1, 1, 1)

        # 3. 计算线性插值状态 x_t (Displacement Interpolation)
        # x_t = (1 - t) * x_0 + t * x_1
        x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1

        # 4. 计算目标速度场 u_t (条件向量场)
        # 对于线性路径，u_t = x_1 - x_0 (恒定位移)
        u_target = x_1 - x_0

        # 5. 模型预测速度场和黎曼度量
        outputs = self.model(x_t, t, hsi_lr, msi_hr)
        v_pred = outputs["pred"]
        M = outputs["metric_tensor"]

        # 6. 计算黎曼度量加权流匹配损失 (L_RFM)
        l_rfm = riemannian_flow_matching_loss(v_pred, u_target, M, 
                                               use_riemannian=self.use_riemannian)

        # 7. 计算梯度一致性损失 (L_grad)
        l_grad = gradient_consistency_loss(v_pred, u_target)

        # 8. 计算光谱角损失 (L_angle)
        # 预测的下一步状态: x_t + v_pred * dt (近似为 x_t + v_pred)
        # 与目标 x_1 之间的光谱一致性
        x_pred = x_t + v_pred  # 预测的融合结果
        l_angle = angle_loss(x_pred, x_1)
        
        # DEBUG: 打印一次调试信息
        if not hasattr(self, '_debug_printed'):
            print(f"[DEBUG] x_pred shape: {x_pred.shape}, x_1 shape: {x_1.shape}")
            print(f"[DEBUG] x_pred range: [{x_pred.min():.4f}, {x_pred.max():.4f}]")
            print(f"[DEBUG] x_1 range: [{x_1.min():.4f}, {x_1.max():.4f}]")
            print(f"[DEBUG] l_angle value: {l_angle.item():.6f}")
            self._debug_printed = True

        # 9. 计算最优传输方向损失 (L_OTD)
        l_otd = optimal_transport_direction_loss(v_pred, u_target, M)

        # 10. 总损失: L_total = L_RFM + λ_grad * L_grad + λ_angle * L_angle + λ_otd * L_OTD
        total_loss = l_rfm + self.lambda_otd * l_otd
        # 可选: + self.lambda_potential * l_grad + self.lambda_angle * l_angle

        # 11. 优化步骤
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "rfm": l_rfm.item(),
            "grad": l_grad.item(),
            "angle": l_angle.item(),
            "otd": l_otd.item()
        }


# ==========================================
# 6. ODE Solver for Inference
# ==========================================

@torch.no_grad()
def ode_solve(model: nn.Module, hsi_lr: torch.Tensor, msi_hr: torch.Tensor, 
              steps: int = 1) -> Tuple[torch.Tensor, list]:
    """
    通过 Euler 积分求解 ODE，从 x_0 传输到 x_1。
    
    ODE: dz_t/dt = v_θ(z_t, t)
    
    积分路径: t: 0 -> 1
    起点: x_0 = Upsample(hsi_lr)
    终点: x_1 = 融合后的 HR-HSI
    
    Args:
        model: 训练好的速度场网络
        hsi_lr: 低分辨率 HSI (B, C, H_lr, W_lr)
        msi_hr: 高分辨率 MSI (B, C_msi, H_hr, W_hr)
        steps: ODE 积分步数
    
    Returns:
        x: 最终融合结果 (B, C, H_hr, W_hr)
        traj: 轨迹列表
    """
    B, C, H_lr, W_lr = hsi_lr.shape
    H_hr, W_hr = msi_hr.shape[-2:]
    device = hsi_lr.device
    
    # 起点: 上采样的 LR-HSI
    x = F.interpolate(hsi_lr, size=(H_hr, W_hr), mode='bicubic', align_corners=False)
    
    dt = 1.0 / steps
    traj = [x.cpu()]
    
    for i in range(steps):
        t_val = i / steps
        t = torch.ones(B, device=device) * t_val
        
        # 预测速度场
        outputs = model(x, t, hsi_lr, msi_hr)
        v = outputs["pred"]
        
        # Euler 积分: x_{t+dt} = x_t + v_t * dt
        x = x + v * dt
        traj.append(x.cpu())
        
    return x, traj


# ==========================================
# 7. Testing & Analysis
# ==========================================

def analyze_module_flops(module, input_tensor, module_name):
    """分析模块的参数量和 FLOPs"""
    module_params = sum(p.numel() for p in module.parameters())
    try:
        module_flops_total = 0
        for sub_module in module.modules():
            if isinstance(sub_module, nn.Conv2d):
                h, w = input_tensor.shape[-2:] if hasattr(input_tensor, 'shape') else (64, 64)
                cin = sub_module.in_channels
                cout = sub_module.out_channels
                k = sub_module.kernel_size[0] if isinstance(sub_module.kernel_size, tuple) else sub_module.kernel_size
                groups = sub_module.groups
                module_flops_total += h * w * cin * cout * k * k // groups
            elif isinstance(sub_module, nn.Linear):
                in_features = sub_module.in_features
                out_features = sub_module.out_features
                module_flops_total += in_features * out_features
    except Exception as e:
        module_flops_total = 0
        print(f"Warning: Failed to estimate FLOPs for {module_name}: {e}")
    return module_params, module_flops_total


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*80)
    print("Testing Structure-Aware Flow Matching (SAFM) Model")
    print("="*80)

    # 测试配置
    hsi_channels = 200  # IndianP
    msi_channels = 5
    dim = 220

    # 创建模型
    model = ImprovedDiT(
        hsi_channels=hsi_channels,
        msi_channels=msi_channels,
        dim=dim,
        alpha=1.0,
        beta=10.0,
        sigma=1.0
    ).to(device)

    # 创建测试输入
    B = 1
    x_t = torch.randn(B, hsi_channels, 128, 128, device=device)
    t = torch.rand(B, device=device)
    hsi_lr = torch.randn(B, hsi_channels, 32, 32, device=device)
    msi = torch.randn(B, msi_channels, 128, 128, device=device)

    print(f"\nInput shapes:")
    print(f"  x_t (current HR HSI): {x_t.shape}")
    print(f"  t (time step): {t.shape}")
    print(f"  hsi_lr (LR HSI): {hsi_lr.shape}")
    print(f"  msi (HR MSI): {msi.shape}")

    # 前向传播测试
    model.eval()
    with torch.no_grad():
        output = model(x_t, t, hsi_lr, msi)

    print(f"\nOutput shapes:")
    print(f"  pred (velocity field): {output['pred'].shape}")
    print(f"  metric_tensor (Riemannian metric): {output['metric_tensor'].shape}")

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  Total Parameters: {total_params:,} ({total_params/1e6:.3f}M)")

    # 测试黎曼度量计算
    print(f"\n" + "-"*40)
    print("Testing Riemannian Metric Computation:")
    metric = output['metric_tensor']
    print(f"  Metric tensor shape: {metric.shape}")
    print(f"  Metric M_00 range: [{metric[:,0,0].min():.4f}, {metric[:,0,0].max():.4f}]")
    print(f"  Metric M_11 range: [{metric[:,1,1].min():.4f}, {metric[:,1,1].max():.4f}]")
    print(f"  Metric M_01 range: [{metric[:,0,1].min():.4f}, {metric[:,0,1].max():.4f}]")

    # 测试损失函数
    print(f"\n" + "-"*40)
    print("Testing Riemannian Flow Matching Loss:")
    u_target = torch.randn_like(output['pred'])
    loss = riemannian_flow_matching_loss(output['pred'], u_target, metric, use_riemannian=True)
    print(f"  Riemannian FM Loss: {loss.item():.4f}")
    
    loss_l2 = riemannian_flow_matching_loss(output['pred'], u_target, metric, use_riemannian=False)
    print(f"  Standard L2 Loss: {loss_l2.item():.4f}")

    # 测试训练器
    print(f"\n" + "-"*40)
    print("Testing SAFM Trainer:")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = SA_SB_Trainer(model, optimizer, device, lambda_potential=0.1, use_riemannian=True)
    
    model.train()
    hsi_hr = torch.randn(B, hsi_channels, 128, 128, device=device)
    loss_dict = trainer.train_step(hsi_lr, hsi_hr, msi)
    print(f"  Training losses: {loss_dict}")

    # 测试 ODE 求解器
    print(f"\n" + "-"*40)
    print("Testing ODE Solver:")
    model.eval()
    result, trajectory = ode_solve(model, hsi_lr, msi, steps=5)
    print(f"  Final result shape: {result.shape}")
    print(f"  Trajectory length: {len(trajectory)}")

    print("\n" + "="*80)
    print("✅ Structure-Aware Flow Matching (SAFM) Model Test Complete")
    print("="*80)
