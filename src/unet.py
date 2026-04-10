import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, timesteps, output_dim):
        super().__init__()
        self.output_dim = output_dim
        position = torch.arange(timesteps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, output_dim, 2) * (-math.log(10000.) / output_dim))
        pe = torch.zeros(timesteps, output_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[x].reshape(x.shape[0], self.output_dim)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.ln = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.ln(x)

class TimeClassEmbedding(nn.Module):
    def __init__(self, timesteps, n_classes, hid_size):
        super().__init__()
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(timesteps=timesteps, output_dim=hid_size),
            nn.Linear(hid_size, hid_size * 4),
            nn.ReLU(),
            nn.Linear(hid_size * 4, hid_size * 4)
        )
        self.class_emb = nn.Embedding(n_classes, hid_size * 4)

    def forward(self, t, y):
      return self.time_mlp(t) + self.class_emb(y)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_class_emb_dim, is_residual=False):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.mlp = MLP(input_dim=time_class_emb_dim, output_dim=out_channels)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.is_residual = is_residual

    def forward(self, x, time_class_emb):
        h = F.relu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.mlp(time_class_emb).view(time_class_emb.size(0), -1, 1, 1)
        h = F.relu(self.norm2(h))
        h = self.conv2(h)

        return self.skip(x) + h if self.is_residual else h

class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, time_class_emb_dim, input_dim):
        super().__init__()
        self.n_h = n_heads
        assert time_class_emb_dim % n_heads == 0
        self.head_dim = time_class_emb_dim // n_heads

        self.to_k = nn.Linear(input_dim, time_class_emb_dim)
        self.to_q = nn.Linear(input_dim, time_class_emb_dim)
        self.to_v = nn.Linear(input_dim, time_class_emb_dim)
        self.proj = nn.Linear(time_class_emb_dim, input_dim)

        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x, t):
        b, c, h, w = x.shape
        seq_len = h * w
        x_flat = x.view(b, c, seq_len).permute(0, 2, 1)
        res = self.norm(x_flat)

        k = self.to_k(res).view(b, seq_len, self.n_h, self.head_dim).permute(0, 2, 1, 3)
        q = self.to_q(res).view(b, seq_len, self.n_h, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(res).view(b, seq_len, self.n_h, self.head_dim).permute(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).reshape(b, seq_len, c)

        out = self.proj(context)
        res = out + res
        res = self.mlp(res) + res

        return res.permute(0, 2, 1).view(b, c, h, w)

class SequenceWithTimeEmbedding(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.models = nn.ModuleList(blocks)

    def forward(self, x, time_class_emb):
        for model in self.models:
            x = model(x, time_class_emb)

        return x

class DownBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, time_class_emb=None):
        return self.pool(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upscale = nn.ConvTranspose2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=2,
                                          stride=2)

    def forward(self, x, time_class_emb=None):
        return self.upscale(x)

class UNet(nn.Module):
    def __init__(self,
                 timesteps,
                 n_classes,
                 in_channels=3,
                 out_channels=3,
                 steps=(1, 2, 4),
                 hid_size=128,
                 attn_step_indexes=[1, 2, 4],
                 n_resolution_blocks=2,
                 has_residuals=True,
                 ):
        super().__init__()

        time_class_emb_dim = hid_size * 4

        self.time_class_emb = TimeClassEmbedding(timesteps, n_classes, hid_size)

        self.first_conv = nn.Conv2d(in_channels, steps[0] * hid_size, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        prev_hid_size = steps[0] * hid_size

        for (i, step) in enumerate(steps):
            blocks = []

            for num_block in range(n_resolution_blocks):
                blocks.append(
                    ResnetBlock(
                        in_channels=prev_hid_size if num_block == 0 else step * hid_size,
                        out_channels=step * hid_size,
                        time_class_emb_dim=time_class_emb_dim,
                        is_residual=has_residuals
                    )
                )

                if step in attn_step_indexes:
                    blocks.append(
                        MultiheadAttention(
                            n_heads=4,
                            time_class_emb_dim=step * hid_size,
                            input_dim=step * hid_size
                        )
                    )

            self.down_blocks.append(SequenceWithTimeEmbedding(blocks))

            if i != len(steps) - 1:
                self.down_blocks.append(DownBlock())

            prev_hid_size = step * hid_size

        self.bottleneck = SequenceWithTimeEmbedding([
            ResnetBlock(prev_hid_size, prev_hid_size, time_class_emb_dim),
            MultiheadAttention(n_heads=4, time_class_emb_dim=prev_hid_size, input_dim=prev_hid_size),
            ResnetBlock(prev_hid_size, prev_hid_size, time_class_emb_dim)
        ])

        self.up_blocks = nn.ModuleList()
        reverse_steps = list(reversed(steps))
        for (i, step) in enumerate(reverse_steps):
            blocks = []

            for num_block in range(n_resolution_blocks):
                in_size = prev_hid_size * 2 if num_block == 0 else (reverse_steps[i+1] * hid_size if i < len(reverse_steps)-1 else step * hid_size)
                out_size = (reverse_steps[i+1] * hid_size if i < len(reverse_steps)-1 else step * hid_size)

                blocks.append(
                    ResnetBlock(
                        in_channels=in_size,
                        out_channels=out_size,
                        time_class_emb_dim=time_class_emb_dim,
                        is_residual=has_residuals
                    )
                )

                if len(reverse_steps)-i-1 in attn_step_indexes:
                    blocks.append(
                        MultiheadAttention(
                            n_heads=4,
                            time_class_emb_dim=out_size,
                            input_dim=out_size
                        )
                    )

            self.up_blocks.append(SequenceWithTimeEmbedding(blocks))

            if i < len(reverse_steps)-1:
                self.up_blocks.append(UpBlock(out_size, out_size))

            prev_hid_size = out_size

        self.out = nn.Sequential(*[
            nn.GroupNorm(8, steps[0] * hid_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=steps[0] * hid_size, out_channels=out_channels, kernel_size=3, padding=1)
        ])

    def forward(self, x, t, y):
        time_class_emb = self.time_class_emb(t, y)
        x = self.first_conv(x)
        skips = []
        for down_block in self.down_blocks:
            x = down_block(x, time_class_emb) if isinstance(down_block, SequenceWithTimeEmbedding) else down_block(x)
            if isinstance(down_block, SequenceWithTimeEmbedding):
                skips.append(x)

        x = self.bottleneck(x, time_class_emb)

        for up_block in self.up_blocks:
            if isinstance(up_block, SequenceWithTimeEmbedding):
                x = up_block(torch.cat([x, skips.pop()], 1), time_class_emb)
            else:
                x = up_block(x)

        x = self.out(x)

        return x