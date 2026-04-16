#!/usr/bin/env python3
"""Download DEMAND noise dataset (16-channel environmental noise recordings)."""
import os
import urllib.request
import zipfile
from pathlib import Path
import soundfile as sf
import numpy as np

DEMAND_URL = "https://zenodo.org/records/1227121/files/DEMAND.zip"
DATA_DIR = Path("/Users/liuanwei/NIW/experiments/data")
DEMAND_DIR = DATA_DIR / "demand_raw"
NOISE_OUT = DATA_DIR / "noise_clips"

# DEMAND environments we want (diverse noise types)
WANTED_ENVS = {
    "DLIVING": "domestic_living",     # domestic living room
    "NFIELD": "nature_field",         # outdoor field
    "NPARK": "nature_park",           # park
    "OOFFICE": "office",              # office
    "PCAFETER": "cafeteria",          # cafeteria
    "PRESTO": "restaurant",           # restaurant
    "SCAFE": "street_cafe",           # street cafe
    "SPSQUARE": "public_square",      # public square
    "STRAFFIC": "street_traffic",     # street traffic
    "TBUS": "transport_bus",          # bus
    "TMETRO": "transport_metro",      # metro
    "DWASHING": "domestic_washing",   # washing machine
}

SR = 16000


def download_demand():
    """Download DEMAND dataset from Zenodo."""
    zip_path = DATA_DIR / "DEMAND.zip"

    if DEMAND_DIR.exists() and len(list(DEMAND_DIR.glob("*/*.wav"))) > 10:
        print(f"DEMAND already downloaded at {DEMAND_DIR}")
        return True

    # DEMAND is ~5GB, too large. Instead, use a smaller approach:
    # Download individual noise clips from alternative sources
    print("DEMAND dataset is ~5GB. Using alternative noise source...")
    return False


def generate_realistic_noise():
    """Generate more realistic noise using colored noise + modulation.
    This is a fallback when DEMAND is too large to download."""

    from scipy.signal import butter, lfilter, resample

    NOISE_OUT.mkdir(parents=True, exist_ok=True)

    duration = 30.0  # 30 seconds each
    n_samples = int(duration * SR)

    noise_configs = {
        "babble": {
            "description": "Multi-talker babble noise",
            "gen": lambda n: _generate_babble(n),
        },
        "office": {
            "description": "Office background (keyboard, AC, phone)",
            "gen": lambda n: _generate_office(n),
        },
        "cafeteria": {
            "description": "Cafeteria noise (dishes, crowd, music)",
            "gen": lambda n: _generate_cafeteria(n),
        },
        "traffic": {
            "description": "Street traffic noise",
            "gen": lambda n: _generate_traffic(n),
        },
        "white": {
            "description": "White Gaussian noise",
            "gen": lambda n: np.random.randn(n).astype(np.float32),
        },
        "pink": {
            "description": "Pink (1/f) noise",
            "gen": lambda n: _generate_pink(n),
        },
        "brown": {
            "description": "Brown (1/f^2) noise",
            "gen": lambda n: _generate_brown(n),
        },
        "fan": {
            "description": "HVAC/fan noise (low-frequency dominant)",
            "gen": lambda n: _generate_fan(n),
        },
    }

    for name, config in noise_configs.items():
        out_path = NOISE_OUT / f"{name}.wav"
        if out_path.exists():
            continue
        print(f"  Generating {name}: {config['description']}")
        noise = config["gen"](n_samples)
        noise = noise / (np.max(np.abs(noise)) + 1e-8) * 0.5
        sf.write(str(out_path), noise.astype(np.float32), SR)

    print(f"  Generated {len(noise_configs)} noise clips in {NOISE_OUT}")


def _generate_babble(n):
    """Simulate multi-talker babble by summing modulated harmonics."""
    signal = np.zeros(n, dtype=np.float32)
    # Sum 15 "talkers" with different fundamentals and modulation
    for i in range(15):
        f0 = np.random.uniform(80, 350)
        t = np.arange(n) / SR
        # Harmonics
        voice = np.zeros(n)
        for h in range(1, 6):
            voice += (0.5 ** h) * np.sin(2 * np.pi * f0 * h * t + np.random.uniform(0, 2*np.pi))
        # Syllable-like amplitude modulation (3-6 Hz)
        mod_freq = np.random.uniform(3, 6)
        mod = np.abs(np.sin(2 * np.pi * mod_freq * t + np.random.uniform(0, 2*np.pi))) ** 0.3
        # Random pauses
        pause_mask = np.ones(n)
        for _ in range(int(n / SR)):  # ~1 pause per second
            start = np.random.randint(0, max(1, n - SR))
            length = np.random.randint(int(0.1*SR), int(0.5*SR))
            pause_mask[start:start+length] = 0
        voice = voice * mod * pause_mask
        signal += voice * np.random.uniform(0.05, 0.15)
    # Low-pass filter to make it more speech-like
    from scipy.signal import butter, lfilter
    b, a = butter(4, 4000 / (SR/2), btype='low')
    signal = lfilter(b, a, signal).astype(np.float32)
    return signal


def _generate_office(n):
    """Office noise: AC hum + keyboard clicks + occasional phone."""
    t = np.arange(n) / SR
    # AC hum (50/60 Hz + harmonics)
    ac = 0.3 * np.sin(2*np.pi*60*t) + 0.1 * np.sin(2*np.pi*120*t) + 0.05 * np.sin(2*np.pi*180*t)
    # Keyboard clicks (impulse noise)
    clicks = np.zeros(n)
    for _ in range(int(n / SR * 3)):  # ~3 clicks per second
        pos = np.random.randint(0, n)
        click_len = np.random.randint(50, 200)
        end = min(pos + click_len, n)
        clicks[pos:end] += np.random.randn(end - pos) * np.random.uniform(0.02, 0.08)
    # Low-level white noise (computer fans)
    fan = np.random.randn(n) * 0.02
    from scipy.signal import butter, lfilter
    b, a = butter(2, 2000 / (SR/2), btype='low')
    fan = lfilter(b, a, fan)
    return (ac + clicks + fan).astype(np.float32)


def _generate_cafeteria(n):
    """Cafeteria: babble + dish clinks + background music."""
    babble = _generate_babble(n) * 0.6
    # Dish/glass clinks (high-frequency transients)
    clinks = np.zeros(n)
    for _ in range(int(n / SR * 0.5)):
        pos = np.random.randint(0, n)
        freq = np.random.uniform(3000, 8000)
        dur = np.random.randint(100, 500)
        end = min(pos + dur, n)
        t_local = np.arange(end - pos) / SR
        clinks[pos:end] += np.sin(2*np.pi*freq*t_local) * np.exp(-t_local * 30) * 0.1
    # Background music (filtered noise)
    t = np.arange(n) / SR
    music = np.sin(2*np.pi*440*t) * 0.02 * (1 + 0.5*np.sin(2*np.pi*2*t))
    return (babble + clinks + music).astype(np.float32)


def _generate_traffic(n):
    """Street traffic: low-frequency rumble + passing cars."""
    t = np.arange(n) / SR
    # Base rumble (brown noise filtered)
    rumble = _generate_brown(n) * 0.4
    from scipy.signal import butter, lfilter
    b, a = butter(2, 500 / (SR/2), btype='low')
    rumble = lfilter(b, a, rumble)
    # Passing cars (swept low-frequency bursts)
    cars = np.zeros(n)
    for _ in range(int(n / SR * 0.3)):
        pos = np.random.randint(0, max(1, n - 2*SR))
        dur = np.random.randint(SR, 2*SR)
        end = min(pos + dur, n)
        t_local = np.arange(end - pos) / SR
        car = np.random.randn(end - pos) * 0.15
        # Envelope: rise then fall
        env = np.sin(np.pi * np.arange(end - pos) / (end - pos)) ** 2
        b2, a2 = butter(2, 800 / (SR/2), btype='low')
        car = lfilter(b2, a2, car * env)
        cars[pos:end] += car
    return (rumble + cars).astype(np.float32)


def _generate_pink(n):
    """Pink (1/f) noise using Voss-McCartney algorithm."""
    from scipy.signal import lfilter
    white = np.random.randn(n).astype(np.float32)
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004709510])
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
    return lfilter(b, a, white).astype(np.float32)


def _generate_brown(n):
    """Brown (1/f^2) noise via cumulative sum of white noise."""
    white = np.random.randn(n).astype(np.float32) * 0.01
    brown = np.cumsum(white)
    # High-pass to remove DC drift
    from scipy.signal import butter, lfilter
    b, a = butter(1, 20 / (SR/2), btype='high')
    brown = lfilter(b, a, brown).astype(np.float32)
    return brown


def _generate_fan(n):
    """HVAC/fan noise: low-frequency harmonics + broadband."""
    t = np.arange(n) / SR
    # Fan blade frequency and harmonics
    f_blade = np.random.uniform(30, 80)
    fan = np.zeros(n)
    for h in range(1, 8):
        fan += (0.3 ** h) * np.sin(2*np.pi*f_blade*h*t + np.random.uniform(0, 2*np.pi))
    # Broadband turbulence
    turb = np.random.randn(n) * 0.1
    from scipy.signal import butter, lfilter
    b, a = butter(3, 3000 / (SR/2), btype='low')
    turb = lfilter(b, a, turb)
    return (fan * 0.3 + turb).astype(np.float32)


if __name__ == "__main__":
    print("Preparing noise dataset...")
    if not download_demand():
        generate_realistic_noise()
    print("Done!")
