"""C3.0 spike: compare Route A (FOA -> ears) and Route B (direct HRIR) ear cues."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from secondear.binaural import encode_foa, foa_to_binaural, load_hrirs, render_direct
from secondear.binaural.cues import ild_bands_db, itd_us, spectral_diff_db

ROOT = Path(__file__).resolve().parents[1]
SOFA = ROOT / "data" / "hrtf" / "hutubs" / "pp1_HRIRs_measured.sofa"
FIG = ROOT / "docs" / "research" / "c3.0-spike.png"
FS = 24000
AZS = list(range(-180, 166, 15))
KEYS = ("broadband", "low", "mid", "high")


def measure(hrirs):
    noise = np.random.default_rng(0).standard_normal(FS)
    rows = {}
    wav = {}
    for az in AZS:
        for route, out in (
            ("A", foa_to_binaural(encode_foa(noise, az, 0), hrirs)),
            ("B", render_direct(noise, az, 0, hrirs)),
        ):
            wav[route, az] = out
            rows[route, az] = {
                "itd": itd_us(out, FS),
                "itd_lf": itd_us(out, FS, fmax=1500.0),
                **ild_bands_db(out, FS),
            }
    return rows, wav


def main() -> int:
    if not SOFA.exists():
        print(f"missing {SOFA}")
        return 1
    hrirs = load_hrirs(SOFA, FS)
    rows, wav = measure(hrirs)

    print(
        "az | ITD_us broadband A B | ITD_us <1.5k A B | ILD_dB broadband A B | low A B | mid A B | high A B"
    )
    for az in AZS:
        a, b = rows["A", az], rows["B", az]
        print(
            f"{az:4d} | {a['itd']:7.1f} {b['itd']:7.1f} | {a['itd_lf']:7.1f} {b['itd_lf']:7.1f} | "
            + " | ".join(f"{a[k]:6.2f} {b[k]:6.2f}" for k in KEYS)
        )

    print("\ncue sanity (ILD = left - right; ITD > 0 means left lags)")
    for az, sign in ((90, 1), (-90, -1)):
        for route in "AB":
            r = rows[route, az]
            ok = r["broadband"] * sign > 0 and r["itd"] * sign < 0
            ok_lf = r["broadband"] * sign > 0 and r["itd_lf"] * sign < 0
            print(
                f"az {az:+d} route {route}: ILD {r['broadband']:.2f} dB, ITD broadband {r['itd']:.1f} us"
                f" -> {'PASS' if ok else 'FAIL'}; ITD <1.5k {r['itd_lf']:.1f} us -> {'PASS' if ok_lf else 'FAIL'}"
            )

    print("\nblur A vs B (RMS over 24 azimuths)")
    d_itd = np.sqrt(np.mean([(rows["A", z]["itd"] - rows["B", z]["itd"]) ** 2 for z in AZS]))
    print(f"ITD broadband RMS diff: {d_itd:.1f} us")
    d_lf = np.sqrt(np.mean([(rows["A", z]["itd_lf"] - rows["B", z]["itd_lf"]) ** 2 for z in AZS]))
    print(f"ITD <1.5k RMS diff: {d_lf:.1f} us")
    for k in KEYS:
        d = np.sqrt(np.mean([(rows["A", z][k] - rows["B", z][k]) ** 2 for z in AZS]))
        print(f"ILD {k} RMS diff: {d:.2f} dB")

    print("\nfront/back: left-ear 1/3-octave spectral diff (dB RMS) between theta and 180-theta")
    fb = {"A": [], "B": []}
    for th in range(-90, 91, 15):
        other = 180 - th
        other = other - 360 if other >= 180 else other
        if th in (-90, 90):
            continue
        line = f"{th:4d} vs {other:4d}:"
        for route in "AB":
            v = spectral_diff_db(wav[route, th][0], wav[route, other][0], FS)
            fb[route].append(v)
            line += f" {route}={v:5.2f}"
        print(line)
    for route in "AB":
        print(f"mean front/back diff route {route}: {np.mean(fb[route]):.2f} dB")

    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for route, style in (("A", "o-"), ("B", "s--")):
        ax[0].plot(AZS, [rows[route, z]["itd"] for z in AZS], style, label=f"Route {route}")
        ax[1].plot(AZS, [rows[route, z]["itd_lf"] for z in AZS], style, label=f"Route {route}")
        ax[2].plot(AZS, [rows[route, z]["broadband"] for z in AZS], style, label=f"Route {route}")
    ax[0].set_ylabel("ITD broadband (us), + = left lags")
    ax[1].set_ylabel("ITD <1.5 kHz (us)")
    ax[2].set_ylabel("ILD (dB), left - right")
    ax[2].set_xlabel("azimuth (deg, + = left)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG, dpi=110)
    print(f"\nwrote {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
