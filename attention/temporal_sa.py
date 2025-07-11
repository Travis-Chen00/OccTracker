import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import einops
from timm.models.layers import to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np

# 先假设 LayerNormProxy 按普通LayerNorm写
class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DAttentionBaseline(nn.Module):

    def __init__(
            self, q_size, kv_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, use_pe, dwc_pe,
            no_off, fixed_pe, ksize, log_cpb
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                            q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                                   n_sample,
                                                                                                   2).unsqueeze(1)).mul(
                    0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True)  # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)


# 下面是测试代码

def load_image_tensor(path, size=(16,16)):
    img = Image.open(path).convert('RGB')
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # [1, 3, H, W]

def visualize_tensor(tensor, title="", channel=0):
    tensor = tensor.detach().cpu()
    if tensor.dim() == 4:  # B,C,H,W
        if tensor.size(1) == 3:
            # 彩色图像(3通道)，直接显示
            img = tensor[0].permute(1, 2, 0).clamp(0, 1).numpy()
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
            plt.show()
            return
        else:
            # 非3通道，展示单通道
            img = tensor[0, channel]
    elif tensor.dim() == 3:
        img = tensor[channel]
    else:
        raise ValueError("Unsupported tensor shape for visualization")
    plt.imshow(img, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_all_channels(tensor, title_prefix="", max_channels=16):
    """
    以网格形式展示tensor的所有通道（B=1），最多展示max_channels个通道
    tensor: shape (1, C, H, W)
    """
    tensor = tensor.detach().cpu()
    assert tensor.dim() == 4 and tensor.size(0) == 1, "输入需为 (1,C,H,W)"
    C = tensor.size(1)
    n_channels = min(C, max_channels)  # 最多展示多少个通道

    # 计算子图行列数（如展示16个，建议4x4）
    ncols = 4
    nrows = (n_channels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    axes = axes.flatten()

    for i in range(n_channels):
        ax = axes[i]
        img = tensor[0, i]
        ax.imshow(img, cmap='viridis')
        ax.set_title(f"{title_prefix} ch{i}")
        ax.axis('off')

    # 隐藏多余的子图
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def test_dattention_with_image(image_path):
    # 参数设置
    batch_size = 1
    n_heads = 4
    n_head_channels = 8
    n_groups = 2
    q_h, q_w = 16, 16
    stride = 2
    dim = n_heads * n_head_channels

    # 载入并预处理图像
    img_tensor = load_image_tensor(image_path, size=(q_h, q_w))  # [1, 3, 16, 16]

    # 映射层，3->dim
    mapping = nn.Conv2d(3, dim, kernel_size=3, padding=1)

    # 初始化模型
    model = DAttentionBaseline(
        q_size=(q_h, q_w),
        kv_size=(q_h // stride, q_w // stride),
        n_heads=n_heads,
        n_head_channels=n_head_channels,
        n_groups=n_groups,
        attn_drop=0.1,
        proj_drop=0.1,
        stride=stride,
        offset_range_factor=1.0,
        use_pe=True,
        dwc_pe=False,
        no_off=False,
        fixed_pe=True,
        ksize=3,
        log_cpb=False
    )

    model.eval()
    mapping.eval()

    with torch.no_grad():
        x_mapped = mapping(img_tensor)  # [1, dim, 16, 16]
        out, pos, ref = model(x_mapped)

    print("Input image tensor shape:", img_tensor.shape)
    print("Mapped feature tensor shape:", x_mapped.shape)
    print("Output tensor shape:", out.shape)
    print("Offset position shape:", pos.shape)
    print("Reference grid shape:", ref.shape)

    # 可视化输入图 (第0通道，接近RGB红通道)
    visualize_tensor(img_tensor, title="Input Image (channel 0)", channel=0)

    visualize_all_channels(x_mapped, title_prefix="Mapped Feature", max_channels=16)
    visualize_all_channels(out, title_prefix="DAT Output", max_channels=16)


if __name__ == "__main__":
    image_path = "/img/frame_0.jpg"  # 请把sample.jpg换成你本地的测试图片路径
    test_dattention_with_image(image_path)
