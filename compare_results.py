import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# LOAD FILES
# -----------------------------
baseline = pd.read_csv("baseline_results.csv")
proposed = pd.read_csv("proposed_results.csv")

# Ensure same length
min_len = min(len(baseline), len(proposed))
baseline = baseline.iloc[:min_len]
proposed = proposed.iloc[:min_len]

# -----------------------------
# 1. ACCURACY COMPARISON
# -----------------------------
plt.figure()
plt.plot(baseline["accuracy"], label="Baseline")
plt.plot(proposed["accuracy"], label="Energy-Aware")
plt.title("Accuracy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("comparison_accuracy.png")
plt.show()

# -----------------------------
# 2. TOTAL ENERGY COMPARISON
# -----------------------------
plt.figure()
plt.plot(baseline["total_energy"], label="Baseline")
plt.plot(proposed["total_energy"], label="Energy-Aware")
plt.title("Total Energy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.savefig("comparison_total_energy.png")
plt.show()

# -----------------------------
# 3. COMPUTATION ENERGY COMPARISON
# -----------------------------
plt.figure()
plt.plot(baseline["compute_energy"], label="Baseline")
plt.plot(proposed["compute_energy"], label="Energy-Aware")
plt.title("Computation Energy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.savefig("comparison_compute_energy.png")
plt.show()

# -----------------------------
# 4. COMMUNICATION ENERGY COMPARISON
# -----------------------------
plt.figure()
plt.plot(baseline["communication_energy"], label="Baseline")
plt.plot(proposed["communication_energy"], label="Energy-Aware")
plt.title("Communication Energy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.savefig("comparison_communication_energy.png")
plt.show()

# -----------------------------
# 5. SUMMARY (LAST 10 ROUNDS)
# -----------------------------
def last_avg(series):
    return series.tail(10).mean()

print("\n===== FINAL COMPARISON =====")

print("\nAccuracy:")
print("Baseline:", last_avg(baseline["accuracy"]))
print("Proposed:", last_avg(proposed["accuracy"]))

print("\nComputation Energy:")
print("Baseline:", last_avg(baseline["compute_energy"]))
print("Proposed:", last_avg(proposed["compute_energy"]))

print("\nCommunication Energy:")
print("Baseline:", last_avg(baseline["communication_energy"]))
print("Proposed:", last_avg(proposed["communication_energy"]))

print("\nTotal Energy:")
print("Baseline:", last_avg(baseline["total_energy"]))
print("Proposed:", last_avg(proposed["total_energy"]))