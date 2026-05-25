import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

log_file = "outputs/sft-1.5b/checkpoint-8987/trainer_state.json"

with open(log_file) as f:
    state = json.load(f)

train_loss_steps = []
train_loss_vals = []
eval_loss_steps = []
eval_loss_vals = []

for entry in state["log_history"]:
    if "loss" in entry:
        train_loss_steps.append(entry["step"])
        train_loss_vals.append(float(entry["loss"]))
    if "eval_loss" in entry:
        eval_loss_steps.append(entry["step"])
        eval_loss_vals.append(float(entry["eval_loss"]))

plt.figure(figsize=(10, 5))
plt.plot(train_loss_steps, train_loss_vals, label="train loss", alpha=0.6)
plt.plot(eval_loss_steps, eval_loss_vals, label="eval loss", marker="o")
plt.xlabel("step")
plt.ylabel("loss")
plt.title("SFT Training Loss (Qwen2.5-3B)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/loss_curve.png", dpi=150)
print("saved to outputs/loss_curve.png")
