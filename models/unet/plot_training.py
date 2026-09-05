import json
import matplotlib.pyplot as plt

with open("training_history.json", "r") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("U-Net Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("training_loss.png", dpi=300)
plt.show()