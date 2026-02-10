"""
Compare greedy vs optimize approaches across a range of
(max_rooms, starting_occupancy) parameters and plot the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from hospital_simulation import HospitalSimulation

BUDGET = 15000
HOURS = 24
GREEDY_RUNS = 5  # average multiple greedy runs (stochastic)

# Parameter ranges
rooms_range = [50, 60, 70, 80, 88, 100, 120]
occupancy_range = [20, 30, 40, 50, 55, 60, 70]


def run_greedy(initial_occ, max_rooms):
    """Run greedy simulation and return total cost (average of multiple runs)."""
    costs = []
    for _ in range(GREEDY_RUNS):
        sim = HospitalSimulation(
            initial_occupancy=initial_occ,
            initial_nurses=initial_occ,
            budget=BUDGET,
            max_rooms=max_rooms,
        )
        for hour in range(HOURS):
            record = sim.simulate_hour(hour)
            if record.get("status") == "INFEASIBLE":
                return np.nan
        costs.append(sim.cumulative_cost)
    return np.mean(costs)


def run_optimize(initial_occ, max_rooms):
    """Run MIP optimizer and return total expected cost."""
    sim = HospitalSimulation(
        initial_occupancy=initial_occ,
        initial_nurses=initial_occ,
        budget=BUDGET,
        max_rooms=max_rooms,
    )
    if sim.budget is not None and sim.current_nurses * sim.nurse_cost > sim.budget:
        return None
    arrivals = []
    ambulances = []
    for h in range(HOURS):
        tot, amb = sim.get_arrivals(h)
        arrivals.append(tot)
        ambulances.append(amb)

    from hospital_simulation import optimize_staffing

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
        budget=BUDGET,
    )
    if result["status"] != "Optimal":
        return None
    return result["total_cost"]


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
plt.show()
print("Saved comparison_results.png")
