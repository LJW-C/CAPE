"""The released CAPE architecture: complex time/frequency encoders and CFDA."""

import math

import torch
from torch import nn
import torch.nn.functional as F


def iq_fft(x):
    spectrum = torch.fft.fft(torch.complex(x[:, 0], x[:, 1]), dim=-1, norm="ortho")
    return torch.stack([spectrum.real, spectrum.imag], dim=1)


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3):
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.imag = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        real, imag = x.chunk(2, dim=1)
        rr, ii = self.real(real), self.imag(imag)
        ri, ir = self.real(imag), self.imag(real)
        return torch.cat([rr - ii, ri + ir], dim=1)


class ComplexBasicBlock(nn.Module):
    def __init__(self, in_complex, out_complex, kernel_size=7):
        super().__init__()
        self.conv = ComplexConv1d(in_complex, out_complex, kernel_size, kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_complex * 2)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x)), inplace=True))


class DCPNEncoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        blocks, in_complex = [], 1
        for out_complex in channels:
            blocks.append(ComplexBasicBlock(in_complex, out_complex))
            in_complex = out_complex
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels[-1] * 2, latent_dim), nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        return self.head(self.blocks(x))


class CFDA(nn.Module):
    def __init__(self, latent_dim, semantic_dim, channel_dim, attention_dim=128,
                 num_heads=4, dropout=0.1, suppression_strength=1.0):
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        fused_dim = 2 * latent_dim
        self.time_proj = nn.Sequential(nn.Linear(latent_dim, attention_dim),
                                       nn.LayerNorm(attention_dim), nn.GELU())
        self.freq_proj = nn.Sequential(nn.Linear(latent_dim, attention_dim),
                                       nn.LayerNorm(attention_dim), nn.GELU())
        self.semantic_query = nn.Parameter(torch.empty(1, 1, attention_dim))
        self.channel_query = nn.Parameter(torch.empty(1, 1, attention_dim))
        self.semantic_query_from_fused = nn.Linear(fused_dim, attention_dim)
        self.channel_query_from_fused = nn.Linear(fused_dim, attention_dim)
        self.semantic_attention = nn.MultiheadAttention(
            attention_dim, num_heads, dropout=dropout, batch_first=True)
        self.channel_attention = nn.MultiheadAttention(
            attention_dim, num_heads, dropout=dropout, batch_first=True)
        self.channel_to_semantic = nn.Linear(attention_dim, attention_dim, bias=False)
        self.semantic_to_channel = nn.Linear(attention_dim, attention_dim, bias=False)
        self.cross_gate = nn.Sequential(nn.Linear(2 * attention_dim, attention_dim),
                                       nn.GELU(), nn.Linear(attention_dim, 2), nn.Sigmoid())
        self.semantic_norm = nn.LayerNorm(attention_dim)
        self.channel_norm = nn.LayerNorm(attention_dim)
        self.suppression_strength = float(suppression_strength)
        self.semantic_head = nn.Sequential(
            nn.Linear(attention_dim, semantic_dim), nn.LayerNorm(semantic_dim),
            nn.GELU(), nn.Linear(semantic_dim, semantic_dim))
        self.channel_head = nn.Sequential(
            nn.Linear(attention_dim, channel_dim), nn.LayerNorm(channel_dim),
            nn.GELU(), nn.Linear(channel_dim, channel_dim))
        nn.init.normal_(self.semantic_query, std=0.02)
        nn.init.normal_(self.channel_query, std=0.02)

    def forward(self, z_t, z_f):
        fused = torch.cat([z_t, z_f], dim=1)
        tokens = torch.stack([self.time_proj(z_t), self.freq_proj(z_f)], dim=1)
        q_s = self.semantic_query + self.semantic_query_from_fused(fused).unsqueeze(1)
        q_c = self.channel_query + self.channel_query_from_fused(fused).unsqueeze(1)
        a_s, _ = self.semantic_attention(q_s, tokens, tokens, need_weights=False)
        a_c, _ = self.channel_attention(q_c, tokens, tokens, need_weights=False)
        a_s, a_c = a_s.squeeze(1), a_c.squeeze(1)
        gates = self.cross_gate(torch.cat([a_s, a_c], dim=1))
        h_s = self.semantic_norm(a_s - self.suppression_strength * gates[:, :1]
                                * self.channel_to_semantic(a_c))
        h_c = self.channel_norm(a_c - self.suppression_strength * gates[:, 1:2]
                               * self.semantic_to_channel(a_s))
        return self.semantic_head(h_s), self.channel_head(h_c)


def simplex_etf(num_classes, dimension, seed=0):
    k, d = int(num_classes), int(dimension)
    if k < 2 or d < k - 1:
        raise ValueError("ETF requires K >= 2 and dimension >= K-1")
    generator = torch.Generator().manual_seed(int(seed))
    basis = torch.zeros(k, k - 1, dtype=torch.float64)
    for column in range(k - 1):
        scale = math.sqrt((column + 1) * (column + 2))
        basis[:column + 1, column] = 1 / scale
        basis[column + 1, column] = -(column + 1) / scale
    rotation, _ = torch.linalg.qr(torch.randn(d, k - 1, generator=generator, dtype=torch.float64))
    return (math.sqrt(k / (k - 1)) * basis @ rotation.T).float()


class FixedETFClassifier(nn.Module):
    def __init__(self, dimension, num_classes, temperature=0.2, seed=0):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Classifier temperature must be positive")
        self.temperature = float(temperature)
        self.weight = nn.Parameter(simplex_etf(num_classes, dimension, seed), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(num_classes), requires_grad=False)

    def forward(self, features):
        return F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1)) / self.temperature


class CAPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        latent = config.get("latent_dim", 128)
        semantic = config.get("semantic_dim", 128)
        self.time_encoder = DCPNEncoder(config["channels"], latent)
        self.freq_encoder = DCPNEncoder(config["channels"], latent)
        self.factor_attention = CFDA(
            latent, semantic, config.get("channel_dim", 64),
            config.get("attention_dim", 128), config.get("attention_heads", 4),
            config.get("attention_dropout", 0.1), config.get("suppression_strength", 1.0))
        self.classifier = FixedETFClassifier(
            semantic, len(config["known_classes"]), config.get("classifier_temperature", 0.2),
            config.get("etf_seed", 0))

    def factors(self, x):
        return self.factor_attention(self.time_encoder(x), self.freq_encoder(iq_fft(x)))

    def forward(self, x):
        semantic, channel = self.factors(x)
        return self.classifier(semantic), semantic, channel

    def embed(self, x):
        return F.normalize(self.factors(x)[0], dim=1)
