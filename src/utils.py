# src/utils.py
import os
import matplotlib.pyplot as plt

def save_plot(filename: str, folder: str = "../results/plots") -> None:
    """
    Save the current matplotlib plot to the specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)
    plt.savefig(full_path, bbox_inches='tight')
    print(f"✅ Saved: {full_path}")
