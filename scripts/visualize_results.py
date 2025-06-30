import os, glob, json
import matplotlib.pyplot as plt

# 절대 경로로 변경: scripts/../checkpoints/.../evaluation_results
RESULTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..",
    "checkpoints", "resnetsan01-ssi-silog",
    "from_worker6", "evaluation_results"
))

def load_epoch_results(results_dir):
    pattern = os.path.join(results_dir, "epoch_*_results.json")
    files = sorted(glob.glob(pattern),
                   key=lambda p: int(os.path.basename(p).split("_")[1]))
    if not files:
        print(f"[Error] No result files found in {results_dir}")
        return [], {}
    epochs, metrics = [], {}
    for path in files:
        epoch = int(os.path.basename(path).split("_")[1])
        epochs.append(epoch)
        with open(path, "r") as f:
            data = json.load(f)
        for full_key, val in data.items():
            name = full_key.split("-")[-1]
            metrics.setdefault(name, []).append(val)
    return epochs, metrics

def plot_and_save(epochs, metrics):
    # 데이터 없으면 무시
    if not epochs or not metrics:
        print("[Error] No data to plot. Check RESULTS_DIR and JSON files.")
        return

    n = len(metrics)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols*4, rows*3),
                             squeeze=False)

    for ax, (name, vals) in zip(axes.flat, metrics.items()):
        ax.plot(epochs, vals, marker="o")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.set_xticks(epochs)

    # 남는 subplot 숨기기
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    # evaluation_results 폴더에 저장
    out_png = os.path.join(RESULTS_DIR, "evaluation_metrics.png")
    fig.savefig(out_png, dpi=150)
    print(f"Saved all metrics plot to {out_png}")

if __name__ == "__main__":
    epochs, metrics = load_epoch_results(RESULTS_DIR)
    plot_and_save(epochs, metrics)