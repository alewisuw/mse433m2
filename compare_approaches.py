"""
Compare greedy vs optimize approaches across a range of
(max_rooms, starting_occupancy) parameters and plot the results.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from hospital_simulation import HospitalSimulation, optimize_staffing


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare greedy vs MIP optimizer across a range of "
            "(max_rooms, starting_occupancy) parameters and plot the results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--budget", type=int, default=15000,
        help="Hourly budget cap on controllable spending (nurses + diversions).",
    )
    parser.add_argument(
        "--hours", type=int, default=24,
        help="Number of simulation hours.",
    )
    parser.add_argument(
        "--greedy-runs", type=int, default=5,
        help="Number of greedy simulation runs to average (greedy is stochastic).",
    )
    parser.add_argument(
        "--nurse-cost", type=float, default=150,
        help="Cost per nurse per hour.",
    )
    parser.add_argument(
        "--diversion-cost", type=float, default=1000,
        help="Cost per diverted ambulance patient.",
    )
    parser.add_argument(
        "--death-cost", type=float, default=10_000_000,
        help="Penalty per expected patient death (not charged against budget).",
    )
    parser.add_argument(
        "--death-rate", type=float, default=0.1,
        help="Per-hour death probability for waiting (unassigned) patients.",
    )
    parser.add_argument(
        "--rooms", type=int, nargs="+",
        default=[50, 60, 70, 80, 88, 100, 120], metavar="R",
        help="Space-separated list of max-room values to sweep over.",
    )
    parser.add_argument(
        "--occupancy", type=int, nargs="+",
        default=[20, 30, 40, 50, 55, 60, 70], metavar="OCC",
        help="Space-separated list of starting-occupancy values to sweep over.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save plots to disk without displaying them (headless/CI mode).",
    )
    return parser.parse_args()


args = parse_args()

BUDGET = args.budget
HOURS = args.hours
GREEDY_RUNS = args.greedy_runs        # average multiple greedy runs (stochastic)
NURSE_COST = args.nurse_cost
DIVERSION_COST = args.diversion_cost
DEATH_COST = args.death_cost
DEATH_RATE = args.death_rate
SHOW_PLOTS = not args.no_show

# Parameter ranges
rooms_range = args.rooms
occupancy_range = args.occupancy


def run_greedy(initial_occ, max_rooms, budget=BUDGET, nurse_cost=NURSE_COST,
               diversion_cost=DIVERSION_COST, death_cost=DEATH_COST, death_rate=DEATH_RATE):
    """Run greedy simulation and return total cost (average of multiple runs)."""
    costs = []
    for _ in range(GREEDY_RUNS):
        sim = HospitalSimulation(
            initial_occupancy=initial_occ,
            initial_nurses=initial_occ,
            budget=budget,
            max_rooms=max_rooms,
            nurse_cost=nurse_cost,
            diversion_cost=diversion_cost,
            death_cost=death_cost,
            death_rate=death_rate,
        )
        for hour in range(HOURS):
            record = sim.simulate_hour(hour)
            if record.get("status") == "INFEASIBLE":
                return np.nan
        costs.append(sim.cumulative_cost)
    return np.mean(costs)


def run_greedy_hourly(initial_occ, max_rooms, budget=BUDGET):
    """Run greedy simulation and return mean hourly waiting queue across multiple runs."""
    all_waiting = []
    for _ in range(GREEDY_RUNS):
        sim = HospitalSimulation(
            initial_occupancy=initial_occ,
            initial_nurses=initial_occ,
            budget=budget,
            max_rooms=max_rooms,
        )
        run_waiting = []
        feasible = True
        for hour in range(HOURS):
            record = sim.simulate_hour(hour)
            if record.get("status") == "INFEASIBLE":
                feasible = False
                break
            run_waiting.append(record["waiting_queue"])
        if feasible:
            all_waiting.append(run_waiting)
    if not all_waiting:
        return [np.nan] * HOURS
    return list(np.mean(all_waiting, axis=0))


def run_optimize(initial_occ, max_rooms, budget=BUDGET, nurse_cost=NURSE_COST,
                 diversion_cost=DIVERSION_COST, death_cost=DEATH_COST, death_rate=DEATH_RATE):
    """Run MIP optimizer and return total expected cost."""
    sim = HospitalSimulation(
        initial_occupancy=initial_occ,
        initial_nurses=initial_occ,
        budget=budget,
        max_rooms=max_rooms,
        nurse_cost=nurse_cost,
        diversion_cost=diversion_cost,
        death_cost=death_cost,
        death_rate=death_rate,
    )
    if sim.budget is not None and sim.current_nurses * sim.nurse_cost > sim.budget:
        return None
    arrivals = []
    ambulances = []
    for h in range(HOURS):
        tot, amb = sim.get_arrivals(h)
        arrivals.append(tot)
        ambulances.append(amb)

    result = optimize_staffing(
        hours=HOURS,
        initial_occupancy=initial_occ,
        initial_nurses=initial_occ,
        arrivals_per_hour=arrivals,
        ambulance_per_hour=ambulances,
        expected_departures=np.mean(sim.departure_options),
        max_rooms=max_rooms,
        nurse_cost=nurse_cost,
        diversion_cost=diversion_cost,
        death_cost=death_cost,
        death_rate=death_rate,
        budget=budget,
    )
    if result["status"] != "Optimal":
        return None
    return result["total_cost"]


def run_optimize_hourly(initial_occ, max_rooms, budget=BUDGET):
    """Run MIP optimizer and return hourly waiting queue list."""
    sim = HospitalSimulation(
        initial_occupancy=initial_occ,
        initial_nurses=initial_occ,
        budget=budget,
        max_rooms=max_rooms,
    )
    if sim.budget is not None and sim.current_nurses * sim.nurse_cost > sim.budget:
        return [np.nan] * HOURS
    arrivals = []
    ambulances = []
    for h in range(HOURS):
        tot, amb = sim.get_arrivals(h)
        arrivals.append(tot)
        ambulances.append(amb)

    result = optimize_staffing(
        hours=HOURS,
        initial_occupancy=initial_occ,
        initial_nurses=initial_occ,
        arrivals_per_hour=arrivals,
        ambulance_per_hour=ambulances,
        expected_departures=np.mean(sim.departure_options),
        max_rooms=max_rooms,
        nurse_cost=sim.nurse_cost,
        diversion_cost=sim.diversion_cost,
        death_cost=sim.death_cost,
        budget=budget,
    )
    if result["status"] != "Optimal":
        return [np.nan] * HOURS
    return [r["waiting_queue"] for r in result["hourly_plan"]]


# --- Collect results ---
print("Running comparisons...")
print(f"Budget: ${BUDGET:,}/hr  |  Hours: {HOURS}")
print(f"Rooms range: {rooms_range}")
print(f"Occupancy range: {occupancy_range}")
print()

# 1) Vary rooms (fixed starting occupancy = 55)
fixed_occ = 55
greedy_by_rooms = []
opt_by_rooms = []
for r in rooms_range:
    if fixed_occ > r:
        greedy_by_rooms.append(np.nan)
        opt_by_rooms.append(np.nan)
        print(f"  rooms={r:3d}, occ={fixed_occ} => SKIPPED (occupancy > rooms)")
        continue
    g = run_greedy(fixed_occ, r)
    o = run_optimize(fixed_occ, r)
    greedy_by_rooms.append(g)
    opt_by_rooms.append(o)
    o_str = f"${o:,.0f}" if o is not None else "INFEASIBLE"
    print(f"  rooms={r:3d}, occ={fixed_occ} => greedy=${g:,.0f}  optimize={o_str}")

# 2) Vary starting occupancy (fixed rooms = 88)
fixed_rooms = 88
greedy_by_occ = []
opt_by_occ = []
for c in occupancy_range:
    g = run_greedy(c, fixed_rooms)
    o = run_optimize(c, fixed_rooms)
    greedy_by_occ.append(g)
    opt_by_occ.append(o)
    o_str = f"${o:,.0f}" if o is not None else "INFEASIBLE"
    print(f"  rooms={fixed_rooms}, occ={c:3d} => greedy=${g:,.0f}  optimize={o_str}")

# 3) Heatmap grid: vary both
greedy_grid = np.zeros((len(occupancy_range), len(rooms_range)))
opt_grid = np.zeros((len(occupancy_range), len(rooms_range)))
for i, c in enumerate(occupancy_range):
    for j, r in enumerate(rooms_range):
        # skip if starting occupancy > rooms (infeasible start)
        if c > r:
            greedy_grid[i, j] = np.nan
            opt_grid[i, j] = np.nan
            continue
        g = run_greedy(c, r)
        o = run_optimize(c, r)
        greedy_grid[i, j] = g
        opt_grid[i, j] = o if o is not None else np.nan
        o_str = f"${o:,.0f}" if o is not None else "INFEASIBLE"
        print(f"  rooms={r:3d}, occ={c:3d} => greedy=${g:,.0f}  optimize={o_str}")

print("\nDone. Generating plots...")

# --- Plots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Greedy vs Optimized Staffing — Budget ${BUDGET:,}/hr", fontsize=14, fontweight="bold")

# Plot 1: Cost vs Rooms
ax = axes[0, 0]
ax.plot(rooms_range, [x / 1e6 for x in greedy_by_rooms], "o-", label="Greedy", color="steelblue")
ax.plot(rooms_range, [x / 1e6 if x else np.nan for x in opt_by_rooms], "s--", label="Optimize", color="darkorange")
ax.set_xlabel("Max Rooms")
ax.set_ylabel("Total Cost ($M)")
ax.set_title(f"Varying Rooms (Starting Occupancy = {fixed_occ})")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Cost vs Starting Occupancy
ax = axes[0, 1]
ax.plot(occupancy_range, [x / 1e6 for x in greedy_by_occ], "o-", label="Greedy", color="steelblue")
ax.plot(occupancy_range, [x / 1e6 if x else np.nan for x in opt_by_occ], "s--", label="Optimize", color="darkorange")
ax.set_xlabel("Starting Occupancy (Patients = Nurses)")
ax.set_ylabel("Total Cost ($M)")
ax.set_title(f"Varying Starting Occupancy (Rooms = {fixed_rooms})")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Savings heatmap (greedy - optimize)
ax = axes[1, 0]
savings = greedy_grid - opt_grid
im = ax.imshow(savings / 1e6, aspect="auto", cmap="RdYlGn", origin="lower")
ax.set_xticks(range(len(rooms_range)))
ax.set_xticklabels(rooms_range)
ax.set_yticks(range(len(occupancy_range)))
ax.set_yticklabels(occupancy_range)
ax.set_xlabel("Max Rooms")
ax.set_ylabel("Starting Occupancy")
ax.set_title("Cost Savings from Optimizer ($M)\n(Greedy − Optimize, green = optimizer saves more)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Savings ($M)")

# Plot 4: Side-by-side bar for a few key combos
ax = axes[1, 1]
combos = [(55, 70), (55, 88), (40, 88), (60, 88), (55, 100)]
labels = []
g_vals = []
o_vals = []
for occ, rm in combos:
    oi = occupancy_range.index(occ) if occ in occupancy_range else None
    ri = rooms_range.index(rm) if rm in rooms_range else None
    if oi is not None and ri is not None and not np.isnan(greedy_grid[oi, ri]):
        labels.append(f"O={occ}\nR={rm}")
        g_vals.append(greedy_grid[oi, ri] / 1e6)
        o_vals.append(opt_grid[oi, ri] / 1e6)

x_pos = np.arange(len(labels))
width = 0.35
ax.bar(x_pos - width / 2, g_vals, width, label="Greedy", color="steelblue")
ax.bar(x_pos + width / 2, o_vals, width, label="Optimize", color="darkorange")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Total Cost ($M)")
ax.set_title("Selected Scenarios")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("comparison_results.png", dpi=150, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
print("Saved comparison_results.png")

# =============================================================================
# FIGURE 2: Hourly Waiting Queue Across Parameter Combinations
# =============================================================================
print("\nCollecting hourly waiting queue data...")

hours_axis = list(range(HOURS))

# Scenarios: (label, initial_occ, max_rooms)
hourly_scenarios_occ = [
    ("Occ=20, R=88", 20, 88),
    ("Occ=40, R=88", 40, 88),
    ("Occ=55, R=88", 55, 88),
    ("Occ=70, R=88", 70, 88),
]
hourly_scenarios_rooms = [
    ("Occ=55, R=60", 55, 60),
    ("Occ=55, R=70", 55, 70),
    ("Occ=55, R=88", 55, 88),
    ("Occ=55, R=120", 55, 120),
]

colors_occ = ["#1b7837", "#4393c3", "#d6604d", "#762a83"]
colors_rooms = ["#1b7837", "#4393c3", "#d6604d", "#762a83"]


def _safe_peak(seq):
    """Return formatted peak of a sequence, handling all-NaN / empty cases."""
    vals = [x for x in seq if not np.isnan(x)]
    return f"{max(vals):.1f}" if vals else "N/A"


greedy_wait_occ, opt_wait_occ = [], []
for label, occ, rooms in hourly_scenarios_occ:
    gw = run_greedy_hourly(occ, rooms)
    ow = run_optimize_hourly(occ, rooms)
    greedy_wait_occ.append(gw)
    opt_wait_occ.append(ow)
    print(f"  Hourly waiting [{label}]: greedy_peak={_safe_peak(gw)}  opt_peak={_safe_peak(ow)}")

greedy_wait_rooms, opt_wait_rooms = [], []
for label, occ, rooms in hourly_scenarios_rooms:
    gw = run_greedy_hourly(occ, rooms)
    ow = run_optimize_hourly(occ, rooms)
    greedy_wait_rooms.append(gw)
    opt_wait_rooms.append(ow)
    print(f"  Hourly waiting [{label}]: greedy_peak={_safe_peak(gw)}  opt_peak={_safe_peak(ow)}")

fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
fig2.suptitle(
    f"Hourly Waiting Queue — Greedy vs Optimizer (Budget ${BUDGET:,}/hr)",
    fontsize=14, fontweight="bold"
)

# Panel 1: Greedy — vary occupancy
ax = axes2[0, 0]
for (label, occ, rooms), gw, color in zip(hourly_scenarios_occ, greedy_wait_occ, colors_occ):
    ax.plot(hours_axis, gw, "o-", markersize=3, label=label, color=color)
ax.set_xlabel("Hour")
ax.set_ylabel("Patients Waiting (no nurse)")
ax.set_title("Greedy — Varying Starting Occupancy (Rooms=88)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Optimize — vary occupancy
ax = axes2[0, 1]
for (label, occ, rooms), ow, color in zip(hourly_scenarios_occ, opt_wait_occ, colors_occ):
    ax.plot(hours_axis, ow, "s--", markersize=3, label=label, color=color)
ax.set_xlabel("Hour")
ax.set_ylabel("Patients Waiting (expected)")
ax.set_title("Optimizer — Varying Starting Occupancy (Rooms=88)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Greedy — vary rooms
ax = axes2[1, 0]
for (label, occ, rooms), gw, color in zip(hourly_scenarios_rooms, greedy_wait_rooms, colors_rooms):
    ax.plot(hours_axis, gw, "o-", markersize=3, label=label, color=color)
ax.set_xlabel("Hour")
ax.set_ylabel("Patients Waiting (no nurse)")
ax.set_title("Greedy — Varying Max Rooms (Occupancy=55)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: Optimize — vary rooms
ax = axes2[1, 1]
for (label, occ, rooms), ow, color in zip(hourly_scenarios_rooms, opt_wait_rooms, colors_rooms):
    ax.plot(hours_axis, ow, "s--", markersize=3, label=label, color=color)
ax.set_xlabel("Hour")
ax.set_ylabel("Patients Waiting (expected)")
ax.set_title("Optimizer — Varying Max Rooms (Occupancy=55)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("waiting_queue_results.png", dpi=150, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
print("Saved waiting_queue_results.png")

# =============================================================================
# FIGURE 3: Sensitivity Analysis — how cost reacts to parameter changes
# =============================================================================
print("\nRunning sensitivity analysis (baseline: occ=55, rooms=88)...")

BASE_OCC = 55
BASE_ROOMS = 88

# Sensitivity sweep values (one parameter at a time; others at baseline)
sensitivity_params = {
    "Budget ($/hr)":       ("budget",        [6000, 9000, 12000, 15000, 18000, 21000, 25000]),
    "Nurse Cost ($/hr)":   ("nurse_cost",    [75,   100,  125,   150,   175,   200,   225]),
    "Death Rate":          ("death_rate",    [0.02, 0.05, 0.07,  0.10,  0.13,  0.15,  0.20]),
    "Diversion Cost ($)":  ("diversion_cost",[250,  500,  750,   1000,  1250,  1500,  2000]),
}

sens_results = {}
for param_label, (param_key, values) in sensitivity_params.items():
    g_costs, o_costs = [], []
    for val in values:
        kwargs = dict(nurse_cost=NURSE_COST, diversion_cost=DIVERSION_COST, death_cost=DEATH_COST,
                      death_rate=DEATH_RATE, budget=BUDGET)
        kwargs[param_key] = val
        # For budget sensitivity, also propagate to run functions
        b = kwargs.pop("budget")
        g = run_greedy(BASE_OCC, BASE_ROOMS, budget=b, **kwargs)
        o = run_optimize(BASE_OCC, BASE_ROOMS, budget=b, **kwargs)
        g_costs.append(g)
        o_costs.append(o if o is not None else np.nan)
        o_str = f"${o:,.0f}" if o is not None else "INFEASIBLE"
        print(f"  {param_label}={val} => greedy=${g:,.0f}  optimize={o_str}")
    sens_results[param_label] = (values, g_costs, o_costs)

# --- Tornado chart (% change from baseline) ---
baseline_g = run_greedy(BASE_OCC, BASE_ROOMS)
baseline_o = run_optimize(BASE_OCC, BASE_ROOMS)
print(f"\n  Baseline: greedy=${baseline_g:,.0f}  optimize=${baseline_o:,.0f}")

# Build tornado data: for each param, find the range of % cost change
tornado_labels, tornado_g_lo, tornado_g_hi, tornado_o_lo, tornado_o_hi = [], [], [], [], []
for param_label, (values, g_costs, o_costs) in sens_results.items():
    valid_g = [x for x in g_costs if not np.isnan(x)]
    valid_o = [x for x in o_costs if not np.isnan(x)]
    if valid_g and baseline_g:
        g_lo = (min(valid_g) - baseline_g) / baseline_g * 100
        g_hi = (max(valid_g) - baseline_g) / baseline_g * 100
    else:
        g_lo, g_hi = 0, 0
    if valid_o and baseline_o:
        o_lo = (min(valid_o) - baseline_o) / baseline_o * 100
        o_hi = (max(valid_o) - baseline_o) / baseline_o * 100
    else:
        o_lo, o_hi = 0, 0
    tornado_labels.append(param_label)
    tornado_g_lo.append(g_lo)
    tornado_g_hi.append(g_hi)
    tornado_o_lo.append(o_lo)
    tornado_o_hi.append(o_hi)

# Sort by total greedy range (largest swing on top)
ranges = [abs(h - l) for l, h in zip(tornado_g_lo, tornado_g_hi)]
order = sorted(range(len(tornado_labels)), key=lambda i: ranges[i])
tornado_labels = [tornado_labels[i] for i in order]
tornado_g_lo    = [tornado_g_lo[i]   for i in order]
tornado_g_hi    = [tornado_g_hi[i]   for i in order]
tornado_o_lo    = [tornado_o_lo[i]   for i in order]
tornado_o_hi    = [tornado_o_hi[i]   for i in order]

fig3, axes3 = plt.subplots(2, 2, figsize=(15, 11))
fig3.suptitle(
    f"Sensitivity Analysis — Baseline: Occ={BASE_OCC}, Rooms={BASE_ROOMS}, Budget=${BUDGET:,}/hr",
    fontsize=13, fontweight="bold"
)

param_list = list(sensitivity_params.items())

for idx, (param_label, (param_key, values)) in enumerate(param_list):
    ax = axes3[idx // 2, idx % 2]
    gv, ov = sens_results[param_label][1], sens_results[param_label][2]

    ax.plot(values, [x / 1e6 if not np.isnan(x) else np.nan for x in gv],
            "o-", label="Greedy", color="steelblue")
    ax.plot(values, [x / 1e6 if not np.isnan(x) else np.nan for x in ov],
            "s--", label="Optimizer", color="darkorange")

    # Mark the baseline parameter value
    base_val = dict(budget=BUDGET, nurse_cost=NURSE_COST, death_rate=DEATH_RATE, diversion_cost=DIVERSION_COST)[param_key]
    ax.axvline(base_val, color="gray", linestyle=":", linewidth=1.2, label=f"Baseline ({base_val})")

    ax.set_xlabel(param_label)
    ax.set_ylabel("Total Cost ($M)")
    ax.set_title(f"Sensitivity to {param_label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sensitivity_line.png", dpi=150, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
print("Saved sensitivity_line.png")

# --- Tornado chart ---
fig4, ax4 = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(tornado_labels))
bar_h = 0.35

for i, (label, g_lo, g_hi, o_lo, o_hi) in enumerate(
        zip(tornado_labels, tornado_g_lo, tornado_g_hi, tornado_o_lo, tornado_o_hi)):
    ax4.barh(i + bar_h / 2, g_hi - g_lo, left=g_lo, height=bar_h,
             color="steelblue", alpha=0.8, label="Greedy" if i == 0 else "")
    ax4.barh(i - bar_h / 2, o_hi - o_lo, left=o_lo, height=bar_h,
             color="darkorange", alpha=0.8, label="Optimizer" if i == 0 else "")

ax4.axvline(0, color="black", linewidth=0.8)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(tornado_labels)
ax4.set_xlabel("% Change in Total Cost from Baseline")
ax4.set_title(
    f"Tornado Chart — Cost Sensitivity (Occ={BASE_OCC}, Rooms={BASE_ROOMS})\n"
    "Bar width = range of % cost change across each parameter's sweep"
)
ax4.legend()
ax4.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("sensitivity_tornado.png", dpi=150, bbox_inches="tight")
if SHOW_PLOTS:
    plt.show()
print("Saved sensitivity_tornado.png")
print("\nAll analysis complete.")
