import matplotlib.pyplot as plt
import json
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(SCRIPT_DIR, "history.json")

plt.ion()
fig, ax1 = plt.subplots(figsize=(10, 6))

ax2 = ax1.twinx()

try:
    while True:
        if not os.path.exists(HISTORY_FILE):
            print("Keine history.json gefunden.")
            time.sleep(2)
            continue

        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                time.sleep(1)
                continue

        if not isinstance(data, list) or not data:
            time.sleep(2)
            continue

        steps = [d.get("steps", 0) for d in data]
        births = [d.get("births", 0) for d in data]
        x = list(range(1, len(data) + 1))

        ax1.clear()
        ax2.clear()

        ax1.plot(x, steps, color='tab:blue', label="Schritte")
        ax2.plot(x, births, color='tab:orange', label="Geburten")

        ax1.set_xlabel("Simulation")
        ax1.set_ylabel("Schritte (Überlebenszeit)", color='tab:blue')
        ax2.set_ylabel("Geburten", color='tab:orange')

        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        ax1.set_title("KI-Entwicklung über Simulationen")
        ax1.grid(True)
        fig.tight_layout()

        plt.pause(2)

except KeyboardInterrupt:
    print("Live-Graph wurde beendet.")
