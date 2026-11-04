import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up clean academic aesthetic
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)

folds = [
    {"name": "Fold 1", "train": (2011, 2017), "val": (2017, 2018), "test": (2018, 2020), "y": 3.2},
    {"name": "Fold 2", "train": (2013, 2019), "val": (2019, 2020), "test": (2020, 2022), "y": 2.2},
    {"name": "Fold 3", "train": (2015, 2021), "val": (2021, 2022), "test": (2022, 2024), "y": 1.2},
    {"name": "Fold 4", "train": (2017, 2023), "val": (2023, 2024), "test": (2024, 2026), "y": 0.2},
]

c_train = "#4C72B0"  # Steel Blue
c_val = "#E1812C"    # Muted Amber / Coral
c_test = "#55A868"   # Forest Green
height = 0.55

for f in folds:
    y = f["y"]
    # Train bar
    w_train = f["train"][1] - f["train"][0]
    rect_tr = patches.FancyBboxPatch((f["train"][0], y), w_train, height, boxstyle="round,pad=0.02,rounding_size=0.08",
                                     edgecolor="black", facecolor=c_train, linewidth=0.8)
    ax.add_patch(rect_tr)
    ax.text(f["train"][0] + w_train / 2, y + height / 2, f"Training ({w_train}y)\n{f['train'][0]}–{f['train'][1]-1}",
            ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    # Val bar
    w_val = f["val"][1] - f["val"][0]
    rect_val = patches.FancyBboxPatch((f["val"][0], y), w_val, height, boxstyle="round,pad=0.02,rounding_size=0.08",
                                      edgecolor="black", facecolor=c_val, linewidth=0.8)
    ax.add_patch(rect_val)
    ax.text(f["val"][0] + w_val / 2, y + height / 2, f"Val ({w_val}y)\n{f['val'][0]}",
            ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    # Test bar
    w_test = f["test"][1] - f["test"][0]
    rect_te = patches.FancyBboxPatch((f["test"][0], y), w_test, height, boxstyle="round,pad=0.02,rounding_size=0.08",
                                     edgecolor="black", facecolor=c_test, linewidth=0.8)
    ax.add_patch(rect_te)
    ax.text(f["test"][0] + w_test / 2, y + height / 2, f"Testing ({w_test}y)\n{f['test'][0]}–{f['test'][1]-1}",
            ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    # Fold label
    ax.text(2010.5, y + height / 2, f["name"], ha="right", va="center", fontsize=9.5, fontweight="bold", color="#333333")

# Forward shift indicators between successive folds
for i in range(len(folds) - 1):
    y_from = folds[i]["y"]
    y_to = folds[i+1]["y"] + height
    ax.annotate("", xy=(folds[i+1]["train"][0], y_to), xytext=(folds[i]["train"][0], y_from),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2, ls="--"))
    ax.text(folds[i]["train"][0] + 0.8, (y_from + y_to)/2, "+2y shift", fontsize=7.5, color="#555555", style="italic")

# Continuous out-of-sample test coverage span bracket at bottom
ax.annotate("", xy=(2026, -0.25), xytext=(2018, -0.25),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.4))
ax.text(2022, -0.5, "Contiguous 8-Year Out-of-Sample Test Horizon (2018–2025)",
        ha="center", va="center", fontsize=9, fontweight="bold", color="#222222")

ax.set_xlim(2009.5, 2026.5)
ax.set_ylim(-0.7, 4.2)
ax.set_xticks(range(2011, 2027))
ax.set_xticklabels([str(y) for y in range(2011, 2027)], fontsize=8.5)
ax.set_yticks([])

# Style spines
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#666666")
ax.spines["bottom"].set_linewidth(1.0)
ax.grid(axis="x", linestyle=":", alpha=0.4, color="#888888")

# Legend at top
train_patch = patches.Patch(facecolor=c_train, edgecolor="black", label="Training Window (6 years, model parameter updates)")
val_patch = patches.Patch(facecolor=c_val, edgecolor="black", label="Validation Window (1 year, checkpoint Sharpe selection)")
test_patch = patches.Patch(facecolor=c_test, edgecolor="black", label="Held-Out Test Window (2 years, frozen evaluation)")
ax.legend(handles=[train_patch, val_patch, test_patch], loc="upper center", bbox_to_anchor=(0.5, 1.15),
          ncol=3, frameon=False, fontsize=8.5)

plt.tight_layout()
plt.savefig(r"c:\RTM\IIT KGP\sem 7\BTP1_repo\thesis\figures\walk_forward_schematic.png", dpi=300, bbox_inches="tight")
print("Saved walk_forward_schematic.png successfully.")
