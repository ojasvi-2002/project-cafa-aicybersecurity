import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 📂 Locate latest results dynamically
outputs_folder = "outputs"
latest_date_folder = sorted(os.listdir(outputs_folder))[-1]
latest_run_folder = sorted(os.listdir(os.path.join(outputs_folder, latest_date_folder)))[-1]
eval_path = os.path.join(outputs_folder, latest_date_folder, latest_run_folder, "evaluations.json")

# 📖 Load JSON data
with open(eval_path, "r") as f:
    evaluations = json.load(f)

# 📌 Define metrics to plot
metrics = [
    "is_misclassified_rate",
    "is_comp_rate",
    "is_mis_and_comp_rate",
    "l0_costs_mean",
    "l0_costs_on_mis_mean",
    "l0_costs_on_mis_and_comp_mean",
    "stand_linf_costs_mean",
    "stand_linf_costs_on_mis_mean",
    "stand_linfcosts_costs_on_mis_and_comp_mean"
]

# 📊 Set up positions and labels
stages = ["before-attack", "after-cafa", "after-cafa-projection"]
x = np.arange(len(metrics))  # label locations
width = 0.25  # bar width

# 📈 Pull values for each stage
before_values = [evaluations["before-attack"][m] for m in metrics]
after_values = [evaluations["after-cafa"][m] for m in metrics]
after_proj_values = [evaluations["after-cafa-projection"][m] for m in metrics]

# 📊 Create plot
fig, ax = plt.subplots(figsize=(15, 7))

# 📦 Plot bars
ax.bar(x - width, before_values, width, label="Before Attack")
ax.bar(x, after_values, width, label="After CaFA")
ax.bar(x + width, after_proj_values, width, label="After CaFA Projection")

# 🎨 Labeling
ax.set_ylabel("Values")
ax.set_title("Metrics Comparison Across Attack Stages")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig("plots/full_metrics_comparison.png")
plt.show()