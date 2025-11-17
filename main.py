# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- Prepare outputs folder ---
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1) Define universes ---
ambient = ctrl.Antecedent(np.arange(0, 41, 1), "ambient")
desired = ctrl.Antecedent(np.arange(0, 41, 1), "desired")
power = ctrl.Consequent(np.arange(0, 101, 1), "power")

# --- 2) Membership functions ---
ambient["cold"] = fuzz.trimf(ambient.universe, [0, 0, 18])
ambient["comfortable"] = fuzz.trimf(ambient.universe, [15, 22, 28])
ambient["hot"] = fuzz.trimf(ambient.universe, [25, 40, 40])

desired["low"] = fuzz.trimf(desired.universe, [0, 0, 18])
desired["medium"] = fuzz.trimf(desired.universe, [16, 22, 28])
desired["high"] = fuzz.trimf(desired.universe, [25, 40, 40])

power["low"] = fuzz.trimf(power.universe, [0, 0, 30])
power["medium"] = fuzz.trimf(power.universe, [20, 50, 80])
power["high"] = fuzz.trimf(power.universe, [60, 100, 100])

# --- 3) Fuzzy rules ---
rules = [
    ctrl.Rule(ambient["cold"] & desired["high"], power["high"]),
    ctrl.Rule(ambient["cold"] & desired["medium"], power["medium"]),
    ctrl.Rule(ambient["cold"] & desired["low"], power["medium"]),
    ctrl.Rule(ambient["comfortable"] & desired["high"], power["high"]),
    ctrl.Rule(ambient["comfortable"] & desired["medium"], power["medium"]),
    ctrl.Rule(ambient["comfortable"] & desired["low"], power["low"]),
    ctrl.Rule(ambient["hot"] & desired["high"], power["medium"]),
    ctrl.Rule(ambient["hot"] & desired["medium"], power["low"]),
    ctrl.Rule(ambient["hot"] & desired["low"], power["low"]),
]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

# --- Evaluate fuzzy system ---
def evaluate(amb_val, des_val):
    sim.input["ambient"] = float(amb_val)
    sim.input["desired"] = float(des_val)
    sim.compute()
    return sim.output["power"]

# --- Membership degree calculations ---
def membership_degrees(amb_val, des_val):
    amb_vals = [
        fuzz.interp_membership(ambient.universe, ambient["cold"].mf, amb_val),
        fuzz.interp_membership(ambient.universe, ambient["comfortable"].mf, amb_val),
        fuzz.interp_membership(ambient.universe, ambient["hot"].mf, amb_val),
    ]
    des_vals = [
        fuzz.interp_membership(desired.universe, desired["low"].mf, des_val),
        fuzz.interp_membership(desired.universe, desired["medium"].mf, des_val),
        fuzz.interp_membership(desired.universe, desired["high"].mf, des_val),
    ]
    return amb_vals, des_vals

# --- Rule firing strengths (same order as rules) ---
def compute_rule_firings(amb_val, des_val):
    amb_cold, amb_comf, amb_hot = membership_degrees(amb_val, des_val)[0]
    des_low, des_med, des_high = membership_degrees(amb_val, des_val)[1]
    return [
        min(amb_cold, des_high),
        min(amb_cold, des_med),
        min(amb_cold, des_low),
        min(amb_comf, des_high),
        min(amb_comf, des_med),
        min(amb_comf, des_low),
        min(amb_hot, des_high),
        min(amb_hot, des_med),
        min(amb_hot, des_low),
    ]

# --- Decide HEATER/COOLER behavior ---
def describe_behavior(amb, des, power_out):
    if abs(des - amb) < 0.1:
        return "NO ACTION (temperature stable)"
    if des > amb:
        mode = "HEATER"
        val = power_out
    else:
        mode = "COOLER"
        val = 100 - power_out
    if val >= 70:
        return f"{mode}: HIGH power (aggressive)"
    elif val >= 40:
        return f"{mode}: MEDIUM power (moderate)"
    elif val >= 10:
        return f"{mode}: LOW power (minor adjustment)"
    else:
        return f"{mode}: OFF / MINIMAL power"

# --- Improved MF plot: clear, compact, with boxed numeric table ---
def plot_mf_clear(amb_val, des_val, savepath=None):
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)  # single compact axis for MF (ambient+desired shown separately)
    x = ambient.universe

    # Plot ambient MFs (thicker lines)
    ax.plot(x, ambient["cold"].mf, label="Ambient - cold", linewidth=2.0, color="#1f77b4")
    ax.plot(x, ambient["comfortable"].mf, label="Ambient - comfortable", linewidth=2.0, color="#2ca02c")
    ax.plot(x, ambient["hot"].mf, label="Ambient - hot", linewidth=2.0, color="#d62728")

    # Also plot desired MFs (dashed, thinner) for comparison on same axis
    ax.plot(x, desired["low"].mf, linestyle="--", label="Desired - low", linewidth=1.2, color="#1f77b4")
    ax.plot(x, desired["medium"].mf, linestyle="--", label="Desired - medium", linewidth=1.2, color="#2ca02c")
    ax.plot(x, desired["high"].mf, linestyle="--", label="Desired - high", linewidth=1.2, color="#d62728")

    # Mark the input positions and their degrees
    amb_degs, des_degs = membership_degrees(amb_val, des_val)
    # ambient marker
    ax.plot(amb_val, amb_degs[0], "o", color="#1f77b4", markersize=6, clip_on=False)
    ax.plot(amb_val, amb_degs[1], "o", color="#2ca02c", markersize=6, clip_on=False)
    ax.plot(amb_val, amb_degs[2], "o", color="#d62728", markersize=6, clip_on=False)
    # desired marker (slightly shifted vertically so visible)
    ax.plot(des_val, des_degs[0], "s", color="#1f77b4", markersize=6, clip_on=False)
    ax.plot(des_val, des_degs[1], "s", color="#2ca02c", markersize=6, clip_on=False)
    ax.plot(des_val, des_degs[2], "s", color="#d62728", markersize=6, clip_on=False)

    # Vertical lines for ambient and desired
    ax.axvline(amb_val, color="#555555", linestyle=":", linewidth=1.0)
    ax.axvline(des_val, color="#888888", linestyle="--", linewidth=1.0)

    # Titles and labels
    ax.set_xlim(-1, 41)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Temperature (°C)", fontsize=9)
    ax.set_ylabel("Membership degree (0..1)", fontsize=9)
    ax.set_title("Membership Functions — ambient (solid) & desired (dashed)", fontsize=10)

    # Build a tidy boxed table with memberships (top-right)
    table_text = (
        f"Ambient = {amb_val}°C\n"
        f"  cold: {amb_degs[0]:.2f}\n"
        f"  comfortable: {amb_degs[1]:.2f}\n"
        f"  hot: {amb_degs[2]:.2f}\n\n"
        f"Desired = {des_val}°C\n"
        f"  low: {des_degs[0]:.2f}\n"
        f"  medium: {des_degs[1]:.2f}\n"
        f"  high: {des_degs[2]:.2f}"
    )
    # Place the table inside an anchored box
    bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="#333333", alpha=0.95)
    ax.text(0.985, 0.98, table_text, transform=ax.transAxes, fontsize=8,
            va="top", ha="right", bbox=bbox_props)

    # Legend (compact)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig

# --- Rule Firing plot (unchanged) ---
def plot_rule_firings_compact(amb_val, des_val, savepath=None):
    firings = compute_rule_firings(amb_val, des_val)
    fig = plt.figure(figsize=(6, 3), dpi=120)
    plt.bar(range(1, len(firings) + 1), firings, color="skyblue", edgecolor="black")
    plt.xlabel("Rule Index", fontsize=9)
    plt.ylabel("Firing Strength", fontsize=9)
    plt.title("Rule Firing Strengths", fontsize=10)
    plt.ylim(0, 1.05)
    for i, f in enumerate(firings):
        plt.text(i + 1, f + 0.03, f"{f:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig

# --- MAIN ---
if __name__ == "__main__":
    amb = float(input("Enter Ambient Temperature (°C): ").strip())
    des = float(input("Enter Desired Temperature (°C): ").strip())

    power_out = evaluate(amb, des)
    behavior = describe_behavior(amb, des, power_out)

    print(f"\nAmbient={amb}°C, Desired={des}°C → Power={power_out:.2f}%")
    print(f"Behavior: {behavior}\n")

    amb_degs, des_degs = membership_degrees(amb, des)
    print("Ambient memberships: cold={:.2f}, comfortable={:.2f}, hot={:.2f}".format(*amb_degs))
    print("Desired memberships: low={:.2f}, medium={:.2f}, high={:.2f}".format(*des_degs))

    mfs_path = os.path.join(OUT_DIR, "membership_functions_clear.png")
    rf_path = os.path.join(OUT_DIR, "rule_firings_compact.png")

    plot_mf_clear(amb, des, savepath=mfs_path)
    plot_rule_firings_compact(amb, des, savepath=rf_path)

    print(f"\nSaved: {mfs_path}")
    print(f"Saved: {rf_path}")

    plt.show()
