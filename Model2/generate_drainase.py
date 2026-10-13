#!/usr/bin/env python3
"""
Generator data sintetis profil DRAINASE (tinggi air saluran + sensor AJ-SR04M, 1 titik/menit).

Pemakaian:
    python generate_drainase.py --config config_drainase.yaml --out data

Keluaran di folder --out:
    drainase_train.csv, drainase_val.csv, drainase_test.csv   (data utama, kolom mirip Miniatur)
    saluran_<split>.csv    parameter acak tiap saluran (termasuk catchment_gain hasil kalibrasi)
    kejadian_<split>.csv   daftar kejadian hujan tiap rekaman (untuk evaluasi per kejadian)
    contoh_rekaman.png, laporan_validasi.txt

Alur fisik per rekaman:
    hujan (segitiga + noise lognormal) -> koefisien limpasan naik seiring kelembapan tanah
    -> waduk cepat seri (k1,k2) + jalur lambat paralel -> waktu tempuh
    -> tinggi air = aliran dasar + (gain * debit)^p, dibatasi kedalaman saluran
    -> sensor (noise, turbulensi, riak, geseran suhu, spike, dropout, burst)
    -> pembersihan kausal (bisa dipakai real-time di backend).

Catatan kejujuran: "realistis" di sini = masuk akal secara fisik, BUKAN tervalidasi
terhadap data saluran sungguhan.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import lfilter


# ----------------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------------
def loguni(rng, lohi):
    lo, hi = lohi
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def uni(rng, lohi):
    return float(rng.uniform(*lohi))


def lowpass(x, k):
    """Waduk linear dengan konstanta waktu k menit (gain DC = 1)."""
    a = 1.0 - np.exp(-1.0 / k)
    return lfilter([a], [1.0, -(1.0 - a)], x)


def ar1(rng, n, corr_min):
    """Noise AR(1) terstandarisasi (rata-rata 0, simpangan baku ~1)."""
    rho = np.exp(-1.0 / max(corr_min, 1e-6))
    return lfilter([np.sqrt(1 - rho ** 2)], [1.0, -rho], rng.standard_normal(n))


# ----------------------------------------------------------------------------
# 1. Hujan
# ----------------------------------------------------------------------------
def layout_events(sc, rng, cfg):
    """Menentukan kejadian hujan (mulai, durasi, puncak, posisi puncak) dan panjang rekaman."""
    rc = cfg["rain"]
    n_lo, n_hi = sc["n_events"]
    if n_hi == 0:
        T = int(round(uni(rng, rc["dry_length_min"])))
        return [], T

    cursor = uni(rng, rc["lead_idle_min"])
    events = []

    def add(n, dur, peak, pos, gap_fn, role):
        nonlocal cursor
        for _ in range(n):
            d = uni(rng, dur)
            events.append({"start": cursor, "dur": d, "peak": loguni(rng, peak),
                           "pos": uni(rng, pos), "role": role})
            cursor += d + gap_fn()

    if sc.get("sequential"):
        gap = lambda: uni(rng, sc["gap_min"])
    else:
        gap = lambda: sc["min_gap_min"] * rng.uniform(1.0, 2.0)

    add(int(rng.integers(n_lo, n_hi + 1)), sc["duration_min"], sc["peak_mm_h"], sc["peak_pos"], gap, "utama")
    if "extra" in sc:
        ex = sc["extra"]
        add(int(rng.integers(ex["n_events"][0], ex["n_events"][1] + 1)),
            ex["duration_min"], ex["peak_mm_h"], sc["peak_pos"], gap, "tambahan")

    end_last = max(e["start"] + e["dur"] for e in events)
    T = int(np.ceil(end_last + uni(rng, rc["tail_min"])))
    return events, T


def build_rain(events, T, sc, rng, cfg):
    """Intensitas hujan (mm/jam) per menit. Mengembalikan array dan tabel kejadian."""
    if cfg["rain"].get("source", "synthetic") != "synthetic":
        raise NotImplementedError(
            "rain.source = medan_real belum diimplementasikan. Dibutuhkan fungsi yang memecah "
            "data hujan per jam (Open-Meteo/BMKG) menjadi pola per menit. Pakai 'synthetic' dulu.")
    rain = np.zeros(T)
    sigma = sc.get("log_noise_sigma", 0.4)
    corr = cfg["rain"]["noise_corr_min"]
    rows = []
    for i, e in enumerate(events):
        s = int(round(e["start"]))
        n = max(2, int(round(e["dur"])))
        x = (np.arange(n) + 0.5) / n
        pp = e["pos"]
        shape = np.where(x < pp, x / pp, (1 - x) / (1 - pp))
        noise = np.exp(sigma * ar1(rng, n, corr) - 0.5 * sigma ** 2)
        r = e["peak"] * shape * noise
        a, b = min(s, T), min(s + n, T)
        rain[a:b] += r[: b - a]
        rows.append({"event_idx": i, "role": e["role"], "start_min": a, "end_min": b,
                     "peak_mm_h": round(float(r[: max(b - a, 1)].max()) if b > a else 0.0, 2),
                     "total_mm": round(float(r[: b - a].sum() / 60.0), 2)})
    return rain, rows


# ----------------------------------------------------------------------------
# 2. Hujan -> tinggi air
# ----------------------------------------------------------------------------
def draw_hydrology(rng, cfg):
    h = cfg["hydrology"]
    return {
        "k1": loguni(rng, h["k1_min"]), "k2": loguni(rng, h["k2_min"]),
        "k_slow": loguni(rng, h["k_slow_min"]), "w_slow": uni(rng, h["slow_weight"]),
        "delay": int(rng.integers(h["delay_min"][0], h["delay_min"][1] + 1)),
        "p": uni(rng, h["rating_exponent"]),
        "c_dry": uni(rng, h["runoff_coeff_dry"]), "c_wet": uni(rng, h["runoff_coeff_wet"]),
        "wet_cap": uni(rng, h["wetness_capacity_mm"]), "wet_decay": loguni(rng, h["wetness_decay_min"]),
        "wet0": uni(rng, h["initial_wetness_fraction"]),
    }


def route_flow(rain, hy):
    """Debit relatif (belum dikali gain) dari hujan."""
    T = len(rain)
    d = np.exp(-1.0 / hy["wet_decay"])
    wet = lfilter([1.0], [1.0, -d], rain / 60.0)
    wet = wet + hy["wet0"] * hy["wet_cap"] * d ** (np.arange(T) + 1)
    coeff = hy["c_dry"] + (hy["c_wet"] - hy["c_dry"]) * np.minimum(1.0, wet / hy["wet_cap"])
    runoff = rain * coeff
    fast = lowpass(lowpass(runoff, hy["k1"]), hy["k2"])
    slow = lowpass(runoff, hy["k_slow"])
    q = (1 - hy["w_slow"]) * fast + hy["w_slow"] * slow
    if hy["delay"] > 0:
        q = np.concatenate([np.zeros(hy["delay"]), q])[:T]
    return q


def solve_gain(q, base_t, p, target):
    """Cari gain sehingga puncak (base + (gain*q)^p) = target. Monoton, jadi bisection di skala log."""
    lo, hi = -25.0, 25.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        peak = (base_t + (np.exp(mid) * q) ** p).max()
        if peak < target:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


# ----------------------------------------------------------------------------
# 3. Sensor AJ-SR04M dan pembersihan
# ----------------------------------------------------------------------------
def simulate_temperature(rain, rng, cfg):
    s = cfg["sensor"]
    T = len(rain)
    amp = uni(rng, s["temp_daily_amp_c"])
    tod0 = rng.uniform(0, 1440)
    tod = (tod0 + np.arange(T)) % 1440
    daily = amp * np.sin(2 * np.pi * (tod - 540) / 1440)        # puncak sekitar pukul 15:00
    rain_ema = lowpass(rain, 30.0)
    drop = s["temp_rain_drop_c"] * np.minimum(1.0, rain_ema / 5.0)
    return s["temp_mean_c"] + daily - drop + 0.3 * ar1(rng, T, 20)


def simulate_sensor(level, temp, h_pasang, rng, cfg):
    s = cfg["sensor"]
    T = len(level)
    bias = rng.uniform(-s["calibration_bias_cm"], s["calibration_bias_cm"])
    h_true = h_pasang + bias                                       # pemasangan asli sedikit berbeda dari yang diisi
    t_ref = s["temp_mean_c"] + s["temp_install_offset_c"]          # suhu saat sensor dipasang/dikalibrasi
    k = s["sound_speed_pct_per_c"] / 100.0
    temp_factor = 1.0 - k * (temp - t_ref)                         # perangkat tidak mengoreksi suhu

    rate = np.abs(np.gradient(level))                              # cm/menit
    sigma = rng.uniform(*s["noise_sigma_cm"]) + s["turbulence_sigma_cm"] * np.minimum(1.0, rate / 1.0)
    rf = s["ripple_fraction"]
    noise = sigma * (rng.standard_normal(T) + rf * ar1(rng, T, s["ripple_corr_min"]))
    dist = (h_true - level) * temp_factor + noise

    is_spike = np.zeros(T, bool)
    is_dropout = np.zeros(T, bool)

    # burst: gangguan beruntun (serangga, busa, tersumbat)
    starts = np.where(rng.random(T) < s["burst_prob"])[0]
    for st in starts:
        L = int(rng.integers(s["burst_len_min"][0], s["burst_len_min"][1] + 1))
        sl = slice(st, min(T, st + L))
        if rng.random() < s["burst_invalid_share"]:
            is_dropout[sl] = True
        else:
            n = len(range(*sl.indices(T)))
            dist[sl] = rng.uniform(s["blind_zone_cm"], h_pasang * 1.2, n)
            is_spike[sl] = True

    sp = rng.random(T) < s["spike_prob"]
    dist[sp] = rng.uniform(s["blind_zone_cm"], h_pasang * 1.2, sp.sum())
    is_spike |= sp
    is_dropout |= rng.random(T) < s["dropout_prob"]

    dist = np.round(dist / s["resolution_cm"]) * s["resolution_cm"]
    dist[is_dropout] = 0.0
    is_spike &= ~is_dropout
    return dist, is_spike, is_dropout


def clean_distance(dist, cfg):
    """Pembersihan kausal (hanya memakai data masa lalu), bisa dipakai real-time."""
    s, c = cfg["sensor"], cfg["cleaning"]
    out = np.full(len(dist), np.nan)
    rejects = invalid_run = 0
    valid_first = [x for x in dist if s["blind_zone_cm"] < x < s["max_range_cm"]][:5]
    last = float(np.median(valid_first)) if valid_first else None
    for t, x in enumerate(dist):
        valid = s["blind_zone_cm"] < x < s["max_range_cm"]
        if not valid:
            invalid_run += 1
            out[t] = last if last is not None else np.nan
            continue
        if last is None or invalid_run >= c["accept_after_invalid_run"] or abs(x - last) <= c["max_jump_cm"]:
            last, rejects = x, 0
        else:
            rejects += 1
            if rejects >= c["max_consecutive_rejects"]:
                last, rejects = x, 0
        invalid_run = 0
        out[t] = last
    if np.isnan(out).any() and (~np.isnan(out)).any():
        out[np.isnan(out)] = out[~np.isnan(out)][0]
    return out


# ----------------------------------------------------------------------------
# 4. Satu rekaman utuh
# ----------------------------------------------------------------------------
def make_recording(rec_id, scenario, rng, cfg):
    sc = cfg["rain"]["scenarios"][scenario]
    expect_cross = sc["expect_cross"]
    ch_cfg = cfg["channel"]
    max_len = cfg["max_length_min"]

    for attempt in range(60):
        depth = uni(rng, ch_cfg["depth_cm"])
        h_pasang = round(depth + uni(rng, ch_cfg["sensor_above_rim_cm"]), 1)
        thr = round(depth * uni(rng, ch_cfg["flood_fraction"]), 1)
        hy = draw_hydrology(rng, cfg)

        events, T = layout_events(sc, rng, cfg)
        T = min(T, max_len)
        rain, ev_rows = build_rain(events, T, sc, rng, cfg)
        q = route_flow(rain, hy)

        a = np.exp(-1.0 / ch_cfg["wander_tau_min"])
        wander = uni(rng, ch_cfg["wander_fraction"]) * depth * lfilter(
            [np.sqrt(1 - a * a)], [1.0, -a], rng.standard_normal(T))
        base_t = np.maximum(0.5, uni(rng, ch_cfg["base_level_fraction"]) * depth + wander)

        if q.max() > 0:
            tf = uni(rng, sc["target_frac"])
            target = max(tf * thr, base_t.max() + 0.03 * depth)   # puncak harus di atas aliran dasar
            gain = solve_gain(q, base_t, hy["p"], target)
        else:
            tf, target, gain = 0.0, base_t.max(), 0.0
        level = np.minimum(depth, base_t + (gain * q) ** hy["p"])

        crossed = level.max() >= thr
        ok = crossed if expect_cross else (level.max() < thr - 0.5)
        if ok:
            break

    temp = simulate_temperature(rain, rng, cfg)
    dist, is_spike, is_dropout = simulate_sensor(level, temp, h_pasang, rng, cfg)
    dist_clean = clean_distance(dist, cfg)
    water_level = h_pasang - dist_clean

    d5 = np.zeros(T)
    d5[5:] = (level[5:] - level[:-5]) / 5.0
    phase = np.where(d5 > 0.05, "naik", np.where(d5 < -0.03, "surut", "stabil"))

    df = pd.DataFrame({
        "recording_id": rec_id,
        "t": np.arange(T),
        "scenario": scenario,
        "phase": phase,
        "distance_cm": np.round(dist, 1),
        "water_level_true_cm": np.round(level, 2),
        "water_level_cm": np.round(water_level, 2),
        "is_spike": is_spike,
        "is_dropout": is_dropout,
        "container_height_cm": round(depth, 1),       # = kedalaman saluran (nama dipertahankan agar pipeline sama)
        "h_pasang_cm": h_pasang,
        "flood_threshold_cm": thr,
        "rain_mm_h": np.round(rain, 2),               # kebenaran lapangan, BUKAN input model
        "temp_c": np.round(temp, 1),
    })
    params = {
        "recording_id": rec_id, "scenario": scenario, "length_min": T, "ok": ok,
        "depth_cm": round(depth, 1), "h_pasang_cm": h_pasang, "flood_threshold_cm": thr,
        "target_frac": round(tf, 3), "true_peak_cm": round(float(level.max()), 2),
        "catchment_gain": gain, **{k: round(v, 4) if isinstance(v, float) else v for k, v in hy.items()},
    }
    for r in ev_rows:
        r["recording_id"] = rec_id
    return df, params, ev_rows, ok


def make_split(name, n, seed, cfg):
    names = list(cfg["rain"]["scenarios"])
    frames, params, events, failed = [], [], [], 0
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        df, p, ev, ok = make_recording(f"{name}_{i:04d}", names[i % len(names)], rng, cfg)
        failed += (not ok)
        frames.append(df)
        params.append(p)
        events += ev
    return pd.concat(frames, ignore_index=True), pd.DataFrame(params), pd.DataFrame(events), failed


# ----------------------------------------------------------------------------
# 5. Laporan validasi dan grafik
# ----------------------------------------------------------------------------
def make_report(splits, params, failed):
    lines = []
    for name, df in splits.items():
        g = df.groupby("recording_id")
        pr = params[name]
        lines.append(f"=== {name.upper()} ===")
        lines.append(f"rekaman: {g.ngroups}, baris: {len(df)}, "
                     f"rata-rata panjang: {len(df) / g.ngroups:.0f} menit "
                     f"(min {g.size().min()}, maks {g.size().max()}), "
                     f"rekaman yang gagal memenuhi skenario: {failed[name]}")
        lines.append(f"spike: {df.is_spike.mean() * 100:.2f}%  dropout: {df.is_dropout.mean() * 100:.2f}%")
        err = (df.water_level_cm - df.water_level_true_cm).abs()
        lines.append(f"selisih data bersih vs asli: rata-rata {err.mean():.2f} cm, "
                     f"persentil 95 {err.quantile(.95):.2f} cm, maks {err.max():.2f} cm")
        ph = df.phase.value_counts(normalize=True) * 100
        lines.append("porsi baris per fase: " + ", ".join(f"{k} {v:.1f}%" for k, v in ph.items()))
        rate = df.groupby("recording_id").water_level_true_cm.diff().abs()
        lines.append(f"laju perubahan tinggi asli: persentil 99 {rate.quantile(.99):.2f} cm/menit, "
                     f"maks {rate.max():.2f} cm/menit (batas filter {15} cm/menit)")
        lines.append(f"kedalaman saluran: {pr.depth_cm.min():.0f}-{pr.depth_cm.max():.0f} cm; "
                     f"ambang banjir: {pr.flood_threshold_cm.min():.0f}-{pr.flood_threshold_cm.max():.0f} cm")
        lines.append(f"catchment_gain (kalibrasi): median {pr.catchment_gain[pr.catchment_gain > 0].median():.3g}")
        rec = g.agg(scenario=("scenario", "first"), mx=("water_level_true_cm", "max"),
                    thr=("flood_threshold_cm", "first"), n=("t", "size"))
        rec["cross"] = rec.mx >= rec.thr
        rec["puncak_per_ambang"] = rec.mx / rec.thr
        summ = rec.groupby("scenario", sort=False).agg(
            rekaman=("n", "size"), panjang_rata2=("n", "mean"),
            persen_lewat_ambang=("cross", lambda s: 100 * s.mean()),
            puncak_per_ambang=("puncak_per_ambang", "mean"))
        lines.append(summ.round(2).to_string())
        lines.append("")
    return "\n".join(lines)


def make_plot(df, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(cfg["rain"]["scenarios"])
    fig, axes = plt.subplots(4, 2, figsize=(14, 15))
    axes = axes.ravel()
    for ax, sc in zip(axes, names):
        rec = df[df.scenario == sc]
        rec = rec[rec.recording_id == rec.recording_id.iloc[0]]
        raw = rec.h_pasang_cm - rec.distance_cm
        bad = rec.is_spike | rec.is_dropout
        depth = rec.container_height_cm.iloc[0]
        ax.plot(rec.t, raw.where(~bad), color="#bbbbbb", lw=0.7, label="pembacaan mentah")
        ax.scatter(rec.t[bad], raw[bad].clip(-5, depth + 25), color="red", s=10, zorder=5, label="spike/dropout")
        ax.plot(rec.t, rec.water_level_cm, color="#1f77b4", lw=1.1, label="setelah dibersihkan (input model)")
        ax.plot(rec.t, rec.water_level_true_cm, color="black", lw=1, ls="--", label="tinggi asli")
        ax.axhline(rec.flood_threshold_cm.iloc[0], color="orange", ls=":", label="ambang banjir")
        ax.set_ylim(-5, depth + 8)
        ax.set_title(f"{sc}  (saluran {depth:.0f} cm)", fontsize=10)
        ax.set_xlabel("menit"); ax.set_ylabel("tinggi air (cm)")
        ax2 = ax.twinx()
        ax2.bar(rec.t, rec.rain_mm_h, width=1.0, color="#4c9be8", alpha=0.35, label="hujan (mm/jam)")
        ax2.set_ylim(rec.rain_mm_h.max() * 4 + 1, 0)       # sumbu terbalik: hujan "turun" dari atas
        ax2.set_ylabel("hujan (mm/jam)")
    axes[0].legend(fontsize=7, loc="upper left")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_drainase.yaml")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_sc = len(cfg["rain"]["scenarios"])
    splits, params, failed = {}, {}, {}
    for name, sp in cfg["splits"].items():
        if sp["n"] % n_sc:
            print(f"PERINGATAN: n={sp['n']} di split {name} bukan kelipatan {n_sc}, skenario tidak seimbang")
        df, pr, ev, f = make_split(name, sp["n"], sp["seed"], cfg)
        df.to_csv(out / f"{cfg['profile']}_{name}.csv", index=False)
        pr.to_csv(out / f"saluran_{name}.csv", index=False)
        ev.to_csv(out / f"kejadian_{name}.csv", index=False)
        splits[name], params[name], failed[name] = df, pr, f
        print(f"{name}: {df.recording_id.nunique()} rekaman, {len(df)} baris")

    report = make_report(splits, params, failed)
    (out / "laporan_validasi.txt").write_text(report)
    make_plot(splits["train"], cfg, out / "contoh_rekaman.png")
    print(report)


if __name__ == "__main__":
    main()
