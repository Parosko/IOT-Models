#!/usr/bin/env python3
"""
Generator data sintetis profil MINIATUR (water level + sensor AJ-SR04M).

Pemakaian:
    python generate_miniatur.py --config config_miniatur.yaml --out data

Keluaran di folder --out:
    miniatur_train.csv, miniatur_val.csv, miniatur_test.csv
    contoh_rekaman.png, laporan_validasi.txt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# skenario -> apakah air diharapkan MELEWATI ambang banjir
SCENARIOS = {
    "pelan_terus": False,
    "cepat_terus": True,
    "pelan_lalu_cepat": True,
    "cepat_lalu_berhenti": False,
    "berhenti_berhenti": False,
    "naik_lalu_bocor": False,
    "diam": False,
}


# ----------------------------------------------------------------------------
# 1. Menyusun fase (urutan tuang / jeda / bocor)
# ----------------------------------------------------------------------------
def _rate(kind, k, rng, cfg):
    lo, hi = cfg["flow"]["phases"][kind]
    r = rng.uniform(lo, hi) * k
    return -r if kind == "bocor" else r


def _pour(kind, budget, k, rng, cfg):
    """Tuang dengan jenis `kind` sampai total kenaikan kira-kira `budget` cm."""
    dmin, dmax = cfg["flow"]["phase_duration_s"]
    out = []
    while budget > 0.05:
        r = _rate(kind, k, rng, cfg)
        d = max(3.0, min(rng.uniform(dmin, dmax), budget / r))
        out.append((kind, r, d))
        budget -= r * d
    return out


def _pause(rng, cfg, lo=None, hi=None):
    dmin, dmax = cfg["flow"]["phase_duration_s"]
    return ("jeda", 0.0, rng.uniform(lo or dmin, hi or dmax))


def build_phases(scenario, h0, thr, cap, k, T, rng, cfg):
    room = max(thr - h0, 1.0)          # ruang sampai ambang
    room_cap = max(cap - h0, 1.0)      # ruang sampai wadah penuh
    lead = [("jeda", 0.0, rng.uniform(*cfg["flow"]["lead_idle_s"]))]

    if scenario == "pelan_terus":
        ph = lead + _pour("pelan", rng.uniform(0.3, 0.85) * room, k, rng, cfg)
    elif scenario == "cepat_terus":
        ph = lead + _pour("cepat", 1.5 * room_cap, k, rng, cfg)
    elif scenario == "pelan_lalu_cepat":
        ph = (lead + _pour("pelan", rng.uniform(0.2, 0.6) * room, k, rng, cfg)
              + [_pause(rng, cfg, 3, 10)] + _pour("cepat", 1.5 * room_cap, k, rng, cfg))
    elif scenario == "cepat_lalu_berhenti":
        ph = lead + _pour("cepat", rng.uniform(0.3, 0.85) * room, k, rng, cfg)
    elif scenario == "berhenti_berhenti":
        remaining = rng.uniform(0.4, 0.9) * room
        ph = list(lead)
        while remaining > 0.1 and sum(p[2] for p in ph) < T:
            kind = rng.choice(["pelan", "sedang", "cepat"])
            chunk = min(remaining, rng.uniform(0.1, 0.3) * room)
            ph += _pour(kind, chunk, k, rng, cfg)
            ph.append(_pause(rng, cfg, 5, 30))
            remaining -= chunk
    elif scenario == "naik_lalu_bocor":
        kind = rng.choice(["pelan", "sedang"])
        ph = (lead + _pour(kind, rng.uniform(0.3, 0.8) * room, k, rng, cfg)
              + [_pause(rng, cfg, 3, 10)]
              + [("bocor", _rate("bocor", k, rng, cfg), rng.uniform(40, 120))])
    elif scenario == "diam":
        ph = [("jeda", 0.0, float(T))]
    else:
        raise ValueError(scenario)

    total = sum(p[2] for p in ph)
    if total < T:
        ph.append(("jeda", 0.0, T - total))
    return ph


def phases_to_arrays(phases, T):
    kinds, rates = [], []
    for kind, r, d in phases:
        n = max(1, int(round(d)))
        kinds += [kind] * n
        rates += [r] * n
    kinds, rates = kinds[:T], rates[:T]
    return np.array(kinds), np.array(rates, dtype=float)


# ----------------------------------------------------------------------------
# 2. Simulasi tinggi air sebenarnya
# ----------------------------------------------------------------------------
def simulate_level(kinds, target, h0, cap, H, rng, cfg):
    T = len(target)
    fl = cfg["flow"]
    a = np.exp(-1.0 / fl["wobble_tau_s"])
    w = 0.0
    tau = fl["response_tau_s"]
    taper_start = cap - 0.05 * H

    level = np.empty(T)
    rates = np.zeros(T)
    level[0] = h0
    r = 0.0
    for t in range(T):
        w = a * w + np.sqrt(1 - a * a) * rng.standard_normal()
        tgt = target[t] * (1 + fl["wobble"] * w)
        tgt = max(tgt, 0.0) if target[t] > 0 else (min(tgt, 0.0) if target[t] < 0 else 0.0)
        r += (tgt - r) / tau                       # respon bertahap
        rr = r
        if rr > 0 and level[t] > taper_start:      # melambat mendekati bibir wadah
            rr *= max(0.0, (cap - level[t]) / (cap - taper_start))
        rates[t] = rr
        if t + 1 < T:
            level[t + 1] = min(cap, max(0.0, level[t] + rr))
    return level, rates


# ----------------------------------------------------------------------------
# 3. Sensor AJ-SR04M (noise, spike, dropout) dan pembersihan
# ----------------------------------------------------------------------------
def simulate_sensor(level, rates, h_pasang, rng, cfg):
    s = cfg["sensor"]
    T = len(level)
    bias = rng.uniform(-s["calibration_bias_cm"], s["calibration_bias_cm"])
    h_true = h_pasang + bias                       # pemasangan asli sedikit berbeda dari yang diisi
    sigma_base = rng.uniform(*s["noise_sigma_cm"])
    sigma = sigma_base + s["pour_extra_sigma_cm"] * np.minimum(1.0, np.abs(rates) / 1.0)
    dist = h_true - level + rng.standard_normal(T) * sigma

    is_spike = rng.random(T) < s["spike_prob"]
    dist[is_spike] = rng.uniform(s["blind_zone_cm"], h_pasang * 1.2, is_spike.sum())
    is_dropout = rng.random(T) < s["dropout_prob"]
    dist[is_dropout] = 0.0
    is_spike &= ~is_dropout

    dist = np.round(dist / s["resolution_cm"]) * s["resolution_cm"]
    return dist, is_spike, is_dropout


def clean_distance(dist, cfg):
    """Pembersihan kausal (hanya memakai data masa lalu), bisa dipakai real-time."""
    s, c = cfg["sensor"], cfg["cleaning"]
    out = np.full(len(dist), np.nan)
    rejects = 0
    # nilai awal = median 5 pembacaan valid pertama (kebal terhadap spike di detik pertama)
    valid_first = [x for x in dist if s["blind_zone_cm"] < x < s["max_range_cm"]][:5]
    last = float(np.median(valid_first)) if valid_first else None
    for t, x in enumerate(dist):
        valid = s["blind_zone_cm"] < x < s["max_range_cm"]
        if not valid:
            out[t] = last if last is not None else np.nan
            continue
        if last is None or abs(x - last) <= c["max_jump_cm"]:
            last, rejects = x, 0
        else:
            rejects += 1
            if rejects >= c["max_consecutive_rejects"]:
                last, rejects = x, 0
        out[t] = last
    # isi bagian awal yang masih kosong dengan nilai valid pertama
    if np.isnan(out).any() and (~np.isnan(out)).any():
        out[np.isnan(out)] = out[~np.isnan(out)][0]
    return out


# ----------------------------------------------------------------------------
# 4. Satu rekaman utuh
# ----------------------------------------------------------------------------
def make_recording(rec_id, scenario, rng, cfg):
    cc = cfg["container"]
    H = rng.uniform(*cc["height_cm"])
    h_pasang = round(H + rng.uniform(*cc["sensor_above_rim_cm"]), 1)
    thr = round(H * rng.uniform(*cc["flood_fraction"]), 1)
    cap = cc["max_fill_fraction"] * H
    lo, hi = cfg["flow"]["speed_multiplier"]
    expect_cross = SCENARIOS[scenario]

    for attempt in range(40):
        h0 = rng.uniform(*cc["initial_level_fraction"]) * H
        k = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        T = int(rng.integers(cfg["length_s"][0], cfg["length_s"][1] + 1))
        phases = build_phases(scenario, h0, thr, cap, k, T, rng, cfg)
        kinds, target = phases_to_arrays(phases, T)
        level, rates = simulate_level(kinds, target, h0, cap, H, rng, cfg)

        # potong rekaman setelah wadah penuh (+10 detik), tapi tidak kurang dari min_length_s
        full = np.where(level >= cap - 0.3)[0]
        if len(full):
            cut = int(min(T, max(cfg["min_length_s"], full[0] + 10)))
            kinds, level, rates = kinds[:cut], level[:cut], rates[:cut]

        crossed = level.max() >= thr
        ok = crossed if expect_cross else (level.max() < thr - 0.5)
        if ok and len(level) >= cfg["min_length_s"]:
            break
    else:
        ok = False

    dist, is_spike, is_dropout = simulate_sensor(level, rates, h_pasang, rng, cfg)
    dist_clean = clean_distance(dist, cfg)
    water_level = h_pasang - dist_clean            # pakai H_pasang yang diisi pengguna

    df = pd.DataFrame({
        "recording_id": rec_id,
        "t": np.arange(len(level)),
        "scenario": scenario,
        "phase": kinds,
        "distance_cm": np.round(dist, 1),
        "water_level_true_cm": np.round(level, 2),
        "water_level_cm": np.round(water_level, 2),
        "is_spike": is_spike,
        "is_dropout": is_dropout,
        "container_height_cm": round(H, 1),
        "h_pasang_cm": h_pasang,
        "flood_threshold_cm": thr,
    })
    return df, ok


def make_split(name, n, seed, cfg):
    names = list(SCENARIOS)
    frames, failed = [], 0
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        df, ok = make_recording(f"{name}_{i:04d}", names[i % len(names)], rng, cfg)
        failed += (not ok)
        frames.append(df)
    return pd.concat(frames, ignore_index=True), failed


# ----------------------------------------------------------------------------
# 5. Laporan validasi dan grafik
# ----------------------------------------------------------------------------
def make_report(splits, failed):
    lines = []
    for name, df in splits.items():
        g = df.groupby("recording_id")
        lines.append(f"=== {name.upper()} ===")
        lines.append(f"rekaman: {g.ngroups}, baris: {len(df)}, "
                     f"rata-rata panjang: {len(df) / g.ngroups:.0f} detik, "
                     f"rekaman yang gagal memenuhi skenario: {failed[name]}")
        lines.append(f"spike: {df.is_spike.mean() * 100:.2f}%  dropout: {df.is_dropout.mean() * 100:.2f}%")
        err = (df.water_level_cm - df.water_level_true_cm).abs()
        lines.append(f"selisih data bersih vs asli: rata-rata {err.mean():.2f} cm, "
                     f"persentil 95 {err.quantile(.95):.2f} cm, maks {err.max():.2f} cm")
        over = (df.water_level_true_cm > df.container_height_cm + 1e-6).sum()
        lines.append(f"baris dengan air melewati bibir wadah: {over}")
        rec = g.agg(scenario=("scenario", "first"),
                    mx=("water_level_true_cm", "max"),
                    thr=("flood_threshold_cm", "first"),
                    n=("t", "size"))
        rec["cross"] = rec.mx >= rec.thr
        summ = rec.groupby("scenario").agg(rekaman=("n", "size"),
                                           panjang_rata2=("n", "mean"),
                                           persen_lewat_ambang=("cross", lambda s: 100 * s.mean()))
        lines.append(summ.round(1).to_string())
        lines.append("")
    return "\n".join(lines)


def make_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(13, 13))
    axes = axes.ravel()
    for ax, sc in zip(axes, SCENARIOS):
        rec = df[df.scenario == sc]
        rec = rec[rec.recording_id == rec.recording_id.iloc[0]]
        raw = rec.h_pasang_cm - rec.distance_cm
        bad = rec.is_spike | rec.is_dropout
        ax.plot(rec.t, raw.where(~bad), color="#bbbbbb", lw=0.8, label="pembacaan mentah")
        ax.scatter(rec.t[bad], raw[bad].clip(-5, rec.container_height_cm.iloc[0] + 25),
                   color="red", s=14, zorder=5, label="spike/dropout")
        ax.plot(rec.t, rec.water_level_cm, color="#1f77b4", lw=1.2, label="setelah dibersihkan (input model)")
        ax.plot(rec.t, rec.water_level_true_cm, color="black", lw=1, ls="--", label="tinggi asli")
        ax.axhline(rec.flood_threshold_cm.iloc[0], color="orange", ls=":", label="ambang banjir")
        ax.set_ylim(-5, rec.container_height_cm.iloc[0] + 8)
        ax.set_title(f"{sc}  (wadah {rec.container_height_cm.iloc[0]:.0f} cm)", fontsize=10)
        ax.set_xlabel("detik")
        ax.set_ylabel("tinggi air (cm)")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_miniatur.yaml")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits, failed = {}, {}
    for name, sp in cfg["splits"].items():
        df, f = make_split(name, sp["n"], sp["seed"], cfg)
        df.to_csv(out / f"{cfg['profile']}_{name}.csv", index=False)
        splits[name], failed[name] = df, f
        print(f"{name}: {df.recording_id.nunique()} rekaman, {len(df)} baris")

    report = make_report(splits, failed)
    (out / "laporan_validasi.txt").write_text(report)
    make_plot(splits["train"], out / "contoh_rekaman.png")
    print(report)


if __name__ == "__main__":
    main()
