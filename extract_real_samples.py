"""
Extracts 3 real DEAP samples (low / mild / high stress) and saves them
to test_samples/ as JSON, and prints the arrays needed for index.html.
Run from: c:\Stress
"""
import os, sys, json
import numpy as np

sys.path.insert(0, r'c:\Stress')
from deap_pipeline import DEAPPipeline

DATA_DIR = r"c:\Users\sambh\Downloads\archive (9)\deap-dataset\data_preprocessed_python"
OUT_DIR  = r"c:\Stress\test_samples"

pipeline = DEAPPipeline(
    data_dir=DATA_DIR,
    window_size_sec=30.0,
    overlap_sec=15.0,
    remove_baseline=True,
    extract_eeg_features=True
)

collected = {}   # "low", "mild", "high"

for sid in range(1, 33):
    if len(collected) == 3:
        break
    fpath = os.path.join(DATA_DIR, f"s{sid:02d}.dat")
    if not os.path.exists(fpath):
        continue

    data, orig_labels = pipeline.load_participant_data(fpath)
    stress_labels = pipeline.extract_stress_labels(orig_labels)
    _, y, _, features = pipeline.create_segments(data, stress_labels)

    if features.shape[0] == 0:
        continue

    hi_mask  = (y == 1)
    lo_mask  = (y == 0)

    # HIGH: stressed label, pick window with strongest beta
    if "high" not in collected and np.any(hi_mask):
        hi_feats  = features[hi_mask]
        beta_mean = hi_feats[:, 64:96].mean(axis=1)
        idx       = np.argmax(beta_mean)
        collected["high"] = {
            "student": f"s{sid:02d}",
            "features": [round(float(v), 6) for v in hi_feats[idx]],
            "hr":   round(float(hi_feats[idx][97]), 1),
            "sdnn": round(float(hi_feats[idx][98]) * 1000, 1),  # convert to ms
            "lf_hf": round(float(hi_feats[idx][100]), 2),
        }
        print(f"[s{sid:02d}] HIGH: beta={beta_mean[idx]:.4f}, HR={collected['high']['hr']}")

    # LOW: non-stressed, pick window with strongest alpha
    if "low" not in collected and np.any(lo_mask):
        lo_feats   = features[lo_mask]
        alpha_mean = lo_feats[:, 32:64].mean(axis=1)
        idx        = np.argmax(alpha_mean)
        collected["low"] = {
            "student": f"s{sid:02d}",
            "features": [round(float(v), 6) for v in lo_feats[idx]],
            "hr":   round(float(lo_feats[idx][97]), 1),
            "sdnn": round(float(lo_feats[idx][98]) * 1000, 1),
            "lf_hf": round(float(lo_feats[idx][100]), 2),
        }
        print(f"[s{sid:02d}] LOW: alpha={alpha_mean[idx]:.4f}, HR={collected['low']['hr']}")

    # MILD: non-stressed but moderate beta (50th–75th pct of beta)
    if "mild" not in collected and np.any(lo_mask):
        lo_feats  = features[lo_mask]
        beta_mean = lo_feats[:, 64:96].mean(axis=1)
        p50 = np.percentile(beta_mean, 50)
        p75 = np.percentile(beta_mean, 75)
        mid_mask  = (beta_mean >= p50) & (beta_mean <= p75)
        if np.any(mid_mask):
            mid_feats = lo_feats[mid_mask]
            idx = len(mid_feats) // 2
            collected["mild"] = {
                "student": f"s{sid:02d}",
                "features": [round(float(v), 6) for v in mid_feats[idx]],
                "hr":   round(float(mid_feats[idx][97]), 1),
                "sdnn": round(float(mid_feats[idx][98]) * 1000, 1),
                "lf_hf": round(float(mid_feats[idx][100]), 2),
            }
            print(f"[s{sid:02d}] MILD: beta={beta_mean[mid_mask][idx]:.4f}, HR={collected['mild']['hr']}")

print("\n=== Summary ===")
for level in ["low", "mild", "high"]:
    if level not in collected:
        print(f"{level}: NOT FOUND"); continue
    c = collected[level]
    arr = np.array(c["features"])
    print(f"{level.upper()} ({c['student']}): theta={arr[:32].mean():.3f} alpha={arr[32:64].mean():.3f} beta={arr[64:96].mean():.3f} FAA={arr[96]:.3f} HR={c['hr']} SDNN={c['sdnn']}ms LF/HF={c['lf_hf']}")

# Save JSON files to test_samples/
os.makedirs(OUT_DIR, exist_ok=True)
for level, info in collected.items():
    out_path = os.path.join(OUT_DIR, f"{level}_stress_sample.json")
    with open(out_path, "w") as f:
        json.dump({"label": f"{level.capitalize()} Stress (Student {info['student']})",
                   "features": info["features"],
                   "meta": {"hr": info["hr"], "sdnn_ms": info["sdnn"], "lf_hf": info["lf_hf"]}}, f, indent=2)
    print(f"Saved: {out_path}")

# Also dump the full collected dict for embedding
with open(os.path.join(OUT_DIR, "_real_samples_for_html.json"), "w") as f:
    json.dump(collected, f, indent=2)
print("Done. Check test_samples/_real_samples_for_html.json")
