from architectures.patch import apply_mamba_patch
apply_mamba_patch()

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba  

class HaarDWT2D(nn.Module):
    """Mendekomposisi citra input menjadi 4 subband frekuensi ter-vektorisasi."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
        hl = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        filters = torch.stack([ll, lh, hl, hh], dim=0).repeat(in_channels, 1, 1)
        filters = filters.unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, x):
        B, C, H, W = x.shape
        out = F.conv2d(x, self.filters, stride=2, groups=C) # type: ignore
        out = out.view(B, C, 4, H // 2, W // 2).permute(0, 2, 1, 3, 4)

        sb_min = out.amin(dim=(-2, -1), keepdim=True)
        sb_max = out.amax(dim=(-2, -1), keepdim=True)
        out_norm = (out - sb_min) / (sb_max - sb_min + 1e-5)

        return out_norm  # Shape: [B, 4, C, H//2, W//2]

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        return self.fc(x).view(b, c, 1, 1)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)

class SpaSE_SSM(nn.Module):
    """Spatial-Aware SE State Space Model Block menggunakan mamba-ssm."""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, se_reduction=16):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2

        self.conv_branch = nn.Sequential(
            nn.Conv2d(self.half_dim, self.half_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.half_dim, self.half_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.half_dim, self.half_dim, 1)
        )

        self.norm = nn.LayerNorm(self.half_dim)
        
        self.row_mamba = Mamba(
            d_model=self.half_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.col_mamba = Mamba(
            d_model=self.half_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.se = SqueezeAndExcitation(self.half_dim, reduction=se_reduction)

    def forward(self, x):
        B, C, H, W = x.shape
        x_conv, x_ssm = torch.chunk(x, 2, dim=1)

        c_out = self.conv_branch(x_conv)

        x_norm = self.norm(x_ssm.permute(0, 2, 3, 1)) 

        x_row = x_norm.reshape(B * H, W, self.half_dim).contiguous()
        y_row = self.row_mamba(x_row)
        y_row = y_row.reshape(B, H, W, self.half_dim)

        x_col = y_row.permute(0, 2, 1, 3).reshape(B * W, H, self.half_dim).contiguous()
        y_col = self.col_mamba(x_col)

        y_col = y_col.reshape(B, W, H, self.half_dim).permute(0, 2, 1, 3).contiguous()
        y_ssm = y_col.permute(0, 3, 1, 2) 

        a_ssm = self.se(y_ssm)
        o_ssm = y_ssm * a_ssm

        y_cat = torch.cat([c_out, o_ssm], dim=1)
        y_shuff = channel_shuffle(y_cat, groups=2)
        return y_shuff + x

class MB_GSF(nn.Module):
    """Menggabungkan fitur 4 subband DWT secara ter-vektorisasi tanpa Python loop."""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim

        self.gate_fc = nn.Sequential(
            nn.Linear(dim, max(1, dim // reduction)),
            nn.GELU(),
            nn.Linear(max(1, dim // reduction), dim),
            nn.Sigmoid()
        )

        self.proj = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x_5d):
        # x_5d: [B, 4, C, H, W]
        B, N, C, H, W = x_5d.shape

        # Global Average Pooling paralel [B, 4, C]
        v = x_5d.mean(dim=(-2, -1))
        
        # Gated Channel Recalibration paralel [B, 4, C, 1, 1]
        g = self.gate_fc(v).view(B, N, C, 1, 1)
        recalibrated = x_5d * g

        # Concatenation & Channel Shuffle
        f_cat = recalibrated.view(B, N * C, H, W)
        f_shuff = channel_shuffle(f_cat, groups=4)
        f_proj = self.proj(f_shuff)

        # Residual Sum & LayerNorm
        f_sum = recalibrated.sum(dim=1)
        f_fused = f_proj + f_sum

        return self.ln(f_fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

class LatentEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class DWTMamba(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        embed_dim=256,
        depth=3,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        se_reduction=16,
        mb_gsf_reduction=4,
        latent_dim=128,
        proj_dim=256
    ):
        super().__init__()
        self.dwt = HaarDWT2D(in_channels)

        self.pe_branches = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
            for _ in range(4)
        ])

        self.spase_branches = nn.ModuleList([
            nn.Sequential(*[
                SpaSE_SSM(
                    dim=embed_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    se_reduction=se_reduction
                ) for _ in range(depth)
            ])
            for _ in range(4)
        ])

        self.pm_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim)
            ) for _ in range(4)
        ])

        self.mb_gsf = MB_GSF(dim=embed_dim, reduction=mb_gsf_reduction)
        self.latent_encoder = LatentEncoder(in_channels, latent_dim=latent_dim)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj_fused = nn.Linear(embed_dim, proj_dim)
        self.proj_latent = nn.Linear(latent_dim, proj_dim)
        self.classifier = nn.Linear(proj_dim * 2, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        subbands = self.dwt(x) # Output DWT: [B, 4, C, H//2, W//2]

        branch_outputs = []
        for i in range(4):
            f = self.pe_branches[i](subbands[:, i])
            f = self.spase_branches[i](f)
            f = self.pm_branches[i](f)
            branch_outputs.append(f)

        feat_5d = torch.stack(branch_outputs, dim=1)

        f_fused = self.mb_gsf(feat_5d)
        f_global = self.gap(f_fused).view(f_fused.size(0), -1)
        u_fused = self.proj_fused(f_global)

        z_latent = self.latent_encoder(x)
        l_latent = self.proj_latent(z_latent)

        h_final = torch.cat([u_fused, l_latent], dim=1)
        logits = self.classifier(h_final)
        return logits

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(2, 1, 224, 224, device=device)
    
    model = DWTMamba(
        in_channels=1,
        num_classes=3,
        embed_dim=256,
        depth=3,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        se_reduction=16,
        mb_gsf_reduction=4,
        latent_dim=128,
        proj_dim=256
    ).to(device)

    output = model(dummy_input)
    print("✅ Testing DWT-Mamba dengan Fleksibilitas Hyperparameter Selesai!")
    print(f"📦 Input Shape  : {dummy_input.shape}")
    print(f"🎯 Output Shape : {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 Total Parameter Model : {total_params:,}")
    print(f"🔥 Parameter yang Dilatih : {trainable_params:,}")