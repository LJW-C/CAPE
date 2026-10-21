"""The six frozen evaluation stress operators; training uses original inputs."""

import torch


def to_complex(x):
    return torch.complex(x[:, 0], x[:, 1])


def from_complex(x):
    return torch.stack([x.real, x.imag], dim=1)


def apply(x, name, level):
    if name == "awgn":
        signal = to_complex(x)
        power = signal.abs().pow(2).mean(dim=1, keepdim=True).clamp_min(1e-8)
        noise_std = (power / (10 ** (float(level) / 10.0)) / 2.0).sqrt()
        noise = torch.complex(torch.randn_like(signal.real), torch.randn_like(signal.real)) * noise_std
        return from_complex(signal + noise)
    if name == "cfo":
        t = torch.linspace(0, 1, steps=x.shape[-1], device=x.device, dtype=x.dtype)
        phase = 2.0 * torch.pi * float(level) * t
        return from_complex(to_complex(x) * torch.exp(1j * phase.unsqueeze(0)))
    if name == "phase":
        phase = torch.tensor(float(level) * torch.pi / 180.0, device=x.device, dtype=x.dtype)
        return from_complex(to_complex(x) * torch.exp(1j * phase))
    if name == "amplitude":
        return x * float(level)
    if name == "multipath":
        delay, attenuation = int(level[0]), float(level[1])
        if delay <= 0:
            return x
        signal = to_complex(x)
        delayed = torch.roll(signal, shifts=delay, dims=-1)
        delayed[..., :delay] = 0
        return from_complex(signal + attenuation * delayed)
    if name == "doppler":
        strength = float(level)
        if strength <= 0:
            return x
        signal = to_complex(x)
        t = torch.linspace(0, 1, steps=x.shape[-1], device=x.device, dtype=x.dtype)
        severity = max(strength / 0.05, 0.1)
        output = torch.zeros_like(signal)
        gains = [1.0, 0.45, 0.25]
        for delay, gain, shift, rate, fading_rate in zip(
                [0, 3, 7], gains, [0.030, -0.022, 0.015], [0.020, -0.014, 0.010], [1.0, 1.7, 2.3]):
            path = torch.roll(signal, shifts=delay, dims=-1)
            if delay > 0:
                path[..., :delay] = 0
            phase = 2.0 * torch.pi * (severity * shift * t + 0.5 * severity * rate * t**2)
            fading = 1.0 + 0.08 * severity * torch.sin(2.0 * torch.pi * fading_rate * t)
            output = output + float(gain) * path * fading.unsqueeze(0) * torch.exp(1j * phase.unsqueeze(0))
        return from_complex(output / sum(gains))
    raise ValueError(f"Unsupported evaluation perturbation: {name}")
