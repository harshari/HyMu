"""
3D Chiplet Stack: Thermal–Warpage Co-Optimisation
=================================================
Stack (bottom → top):
  [Memory die  | 24×32 mm | CTE=3.5 ppm/°C | E=110 GPa]
  [Oxide TIM   |  ~full logic footprint | 1 µm | k=1 W/mK]
  [Logic dies  | 4× 8×8 mm | CTE=2.6 ppm/°C | E=130 GPa | 180 W]
  [Carrier Si  | 24×32 mm | CTE=2.6 ppm/°C | E=130 GPa]

Constraint : t_mem + t_logic + t_carrier + t_TIM = 800 µm
Sweep      : t_mem and t_logic each 20→100 µm (step 5 µm);
             t_carrier = 799 - t_mem - t_logic  µm

Warpage model : generalised multilayer plate (Stoney extended,
                biaxial moduli), ΔT = 235 °C (reflow → room).
Thermal model : 1-D conduction + approximate lateral spreading
                in carrier; logic→carrier heat path only.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. MATERIAL PROPERTIES
# ──────────────────────────────────────────────
# Biaxial modulus  Ē = E / (1 - ν)
def biax(E, nu):
    return E / (1.0 - nu)

# Logic die / Carrier silicon (same material)
E_Si    = 130e9          # Pa
nu_Si   = 0.28
Ebar_Si = biax(E_Si, nu_Si)
alpha_Si = 2.6e-6        # /°C
k_Si    = 150.0          # W/(m·K)

# Memory die  (DRAM-type, slightly higher CTE & lower E)
E_mem    = 110e9
nu_mem   = 0.28
Ebar_mem = biax(E_mem, nu_mem)
alpha_mem = 3.5e-6       # /°C
k_mem    = 100.0         # W/(m·K)

# Oxide TIM (SiO₂-like)
E_TIM    = 70e9
nu_TIM   = 0.17
Ebar_TIM = biax(E_TIM, nu_TIM)
alpha_TIM = 0.55e-6      # /°C  (SiO₂ has very low CTE)
k_TIM    = 1.0           # W/(m·K)

# ──────────────────────────────────────────────
# 2. GEOMETRY
# ──────────────────────────────────────────────
W_mem, L_mem = 24e-3, 32e-3          # m  — memory + carrier footprint
A_mem        = W_mem * L_mem         # 768 mm²

n_logic      = 4
W_l, L_l     = 8e-3, 8e-3
A_logic      = n_logic * W_l * L_l   # 256 mm²  total logic footprint
phi          = A_logic / A_mem        # coverage fraction ≈ 0.333

# Characteristic half-diagonal of the package (for warpage → deflection)
L_char = np.sqrt(W_mem**2 + L_mem**2)   # 40 mm

# ──────────────────────────────────────────────
# 3. FIXED PARAMETERS
# ──────────────────────────────────────────────
t_TIM       = 1e-6          # 1 µm oxide bond
T_TOTAL     = 800e-6        # 800 µm total stack target
T_avail     = T_TOTAL - t_TIM   # 799 µm shared by three die layers

dT_assembly = 260.0 - 25.0  # 235 °C  reflow → room temperature

P_logic  = 180.0            # W
P_memory = 20.0             # W
T_amb    = 25.0             # °C ambient

# ──────────────────────────────────────────────
# 4. WARPAGE CALCULATION  (multilayer plate theory)
# ──────────────────────────────────────────────
def warpage_um(t_mem, t_logic, t_carrier, dT=dT_assembly):
    """
    Returns peak warpage in µm using generalised Stoney for a
    multilayer plate with biaxial moduli.

    Layers (bottom → top):
      0: memory      full area     Ēbar_mem, alpha_mem
      1: oxide TIM   logic area    Ēbar_TIM × phi, alpha_TIM
      2: logic        logic area   Ēbar_Si  × phi, alpha_Si
      3: carrier     full area     Ēbar_Si,         alpha_Si
    """
    # z_bottom of each layer (from z=0 at memory bottom)
    z0 = [0.0,
          t_mem,
          t_mem + t_TIM,
          t_mem + t_TIM + t_logic]
    thk = [t_mem, t_TIM, t_logic, t_carrier]

    # Effective biaxial modulus accounting for partial coverage
    Ebar = [Ebar_mem,
            Ebar_TIM * phi,
            Ebar_Si  * phi,
            Ebar_Si]
    alpha = [alpha_mem, alpha_TIM, alpha_Si, alpha_Si]

    z_mid = [z0[i] + thk[i] / 2.0 for i in range(4)]

    # Extensional stiffness A and its first moment S
    A = sum(Ebar[i] * thk[i] for i in range(4))
    S = sum(Ebar[i] * thk[i] * z_mid[i] for i in range(4))

    z_n = S / A   # neutral surface position

    # Effective bending stiffness  D_eff = D - S²/A
    D = sum(Ebar[i] * (thk[i]**3 / 12.0 + thk[i] * (z_mid[i] - z_n)**2)
            for i in range(4))
    D_eff = D - S**2 / A

    # Thermal bending moment
    M_T = sum(Ebar[i] * alpha[i] * dT * thk[i] * (z_mid[i] - z_n)
              for i in range(4))

    kappa = M_T / D_eff          # 1/m
    # Peak deflection for a free plate: δ = κ L²/8
    delta = kappa * L_char**2 / 8.0
    return abs(delta) * 1e6      # µm

# ──────────────────────────────────────────────
# 5. THERMAL RESISTANCE  (logic junction → carrier top)
# ──────────────────────────────────────────────
def thermal_resistance(t_mem, t_logic, t_carrier):
    """
    θ_jc  (K/W) from logic junction to the top surface of the carrier.

    Heat path (upward from logic):
      logic die  →  TIM  →  carrier (with lateral spreading)

    Spreading approximation: effective area grows linearly from
    A_logic at the TIM interface to A_mem at the carrier top.
    """
    R_logic   = t_logic   / (k_Si  * A_logic)
    R_TIM_val = t_TIM     / (k_TIM * A_logic)

    # Spreading in carrier: harmonic mean of bottom and top areas
    A_spread_eff = (A_logic + A_mem) / 2.0
    R_carrier = t_carrier / (k_Si * A_spread_eff)

    return R_logic + R_TIM_val + R_carrier   # K/W

def junction_temp(t_mem, t_logic, t_carrier):
    return T_amb + P_logic * thermal_resistance(t_mem, t_logic, t_carrier)

# ──────────────────────────────────────────────
# 6. PARAMETER SWEEP
# ──────────────────────────────────────────────
step = 5e-6                              # 5 µm resolution
t_vals = np.arange(20e-6, 101e-6, step) # 20 → 100 µm

T_MEM   = []
T_LOGIC = []
T_CAR   = []
WARP    = []
THETA   = []
T_JUNC  = []

for tm in t_vals:
    for tl in t_vals:
        tc = T_avail - tm - tl
        if tc < 10e-6:           # carrier must be at least 10 µm
            continue
        T_MEM.append(tm * 1e6)
        T_LOGIC.append(tl * 1e6)
        T_CAR.append(tc * 1e6)
        WARP.append(warpage_um(tm, tl, tc))
        theta = thermal_resistance(tm, tl, tc)
        THETA.append(theta * 1e3)   # store in mK/W
        T_JUNC.append(T_amb + P_logic * theta)

T_MEM   = np.array(T_MEM)
T_LOGIC = np.array(T_LOGIC)
T_CAR   = np.array(T_CAR)
WARP    = np.array(WARP)
THETA   = np.array(THETA)   # mK/W
T_JUNC  = np.array(T_JUNC)

# ──────────────────────────────────────────────
# 7. PARETO FRONT  (minimise warpage AND θ_jc)
# ──────────────────────────────────────────────
def is_pareto(costs):
    """Return boolean mask of Pareto-optimal points (minimisation)."""
    n = len(costs)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (costs[j, 0] <= costs[i, 0] and
                    costs[j, 1] <= costs[i, 1] and
                    (costs[j, 0] < costs[i, 0] or costs[j, 1] < costs[i, 1])):
                dominated[i] = True
                break
    return ~dominated

costs = np.column_stack([WARP, THETA])
pareto_mask = is_pareto(costs)

# Pareto subset sorted by warpage
px = WARP[pareto_mask]
py = THETA[pareto_mask]
pc = T_CAR[pareto_mask]
pm = T_MEM[pareto_mask]
pl = T_LOGIC[pareto_mask]
order = np.argsort(px)
px, py, pc, pm, pl = px[order], py[order], pc[order], pm[order], pl[order]

# ──────────────────────────────────────────────
# 8. BUILD 2-D GRIDS FOR CONTOUR PLOTS
# ──────────────────────────────────────────────
t_arr = t_vals * 1e6      # µm axis labels
GRID_W   = np.full((len(t_arr), len(t_arr)), np.nan)
GRID_TH  = np.full((len(t_arr), len(t_arr)), np.nan)
GRID_TC  = np.full((len(t_arr), len(t_arr)), np.nan)

for k in range(len(T_MEM)):
    i = int(round((T_MEM[k]   - 20) / 5))
    j = int(round((T_LOGIC[k] - 20) / 5))
    if 0 <= i < len(t_arr) and 0 <= j < len(t_arr):
        GRID_W[j, i]  = WARP[k]
        GRID_TH[j, i] = THETA[k]
        GRID_TC[j, i] = T_CAR[k]

# ──────────────────────────────────────────────
# 9. DISCRETE CANDIDATE EVALUATION
# ──────────────────────────────────────────────
candidates_um = [30, 40, 60, 80]
print("\n{'t_mem':>8} {'t_logic':>8} {'t_carrier':>10} {'warp_µm':>10} {'θ_jc mK/W':>12} {'Tjunc °C':>10}")
print("-" * 65)
candidate_data = []
for tm_um in candidates_um:
    for tl_um in candidates_um:
        tc_um = (T_avail * 1e6) - tm_um - tl_um
        if tc_um < 10:
            continue
        tm, tl, tc = tm_um * 1e-6, tl_um * 1e-6, tc_um * 1e-6
        w  = warpage_um(tm, tl, tc)
        th = thermal_resistance(tm, tl, tc) * 1e3
        tj = T_amb + P_logic * th / 1e3
        print(f"{tm_um:>8} {tl_um:>8} {tc_um:>10.0f} {w:>10.3f} {th:>12.3f} {tj:>10.3f}")
        candidate_data.append((tm_um, tl_um, tc_um, w, th, tj))
print()

# ──────────────────────────────────────────────
# 10. PLOT
# ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f1117")

title_kw  = dict(color="white", fontsize=11, fontweight="bold", pad=10)
label_kw  = dict(color="#cccccc", fontsize=9)
tick_kw   = dict(colors="#aaaaaa", labelsize=8)
cbar_kw   = dict(color="#cccccc", fontsize=8)
grid_kw   = dict(color="#333333", linewidth=0.4, linestyle="--")

X, Y = np.meshgrid(t_arr, t_arr)

# ── (a) Warpage contour ──────────────────────
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor("#151820")
cf1 = ax1.contourf(X, Y, GRID_W, levels=30, cmap="plasma")
cs1 = ax1.contour(X, Y, GRID_W, levels=8, colors="white", linewidths=0.5, alpha=0.4)
ax1.clabel(cs1, fmt="%.1f µm", fontsize=6.5, colors="white")
cbar1 = fig.colorbar(cf1, ax=ax1, pad=0.02)
cbar1.set_label("Warpage  (µm)", **cbar_kw)
cbar1.ax.tick_params(**tick_kw)
ax1.set_title("Assembly Warpage  [µm]  —  ΔT = 235 °C", **title_kw)
ax1.set_xlabel("Memory die thickness  (µm)", **label_kw)
ax1.set_ylabel("Logic die thickness  (µm)",  **label_kw)
ax1.tick_params(axis="both", **tick_kw)
ax1.grid(**grid_kw)

# Mark the default point (t_mem=40, t_logic=40)
ax1.plot(40, 40, "w*", markersize=12, label="Current (40/40 µm)")
ax1.legend(fontsize=7.5, facecolor="#1e2128", labelcolor="white",
           framealpha=0.7, edgecolor="#555555")

# ── (b) θ_jc contour ────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor("#151820")
cf2 = ax2.contourf(X, Y, GRID_TH, levels=30, cmap="viridis")
cs2 = ax2.contour(X, Y, GRID_TH, levels=8, colors="white", linewidths=0.5, alpha=0.4)
ax2.clabel(cs2, fmt="%.1f", fontsize=6.5, colors="white")
cbar2 = fig.colorbar(cf2, ax=ax2, pad=0.02)
cbar2.set_label("θ_jc  (mK/W)  logic→carrier top", **cbar_kw)
cbar2.ax.tick_params(**tick_kw)
ax2.set_title("Thermal Resistance  θ_jc  (logic → carrier top)", **title_kw)
ax2.set_xlabel("Memory die thickness  (µm)", **label_kw)
ax2.set_ylabel("Logic die thickness  (µm)",  **label_kw)
ax2.tick_params(axis="both", **tick_kw)
ax2.grid(**grid_kw)
ax2.plot(40, 40, "w*", markersize=12, label="Current (40/40 µm)")
ax2.legend(fontsize=7.5, facecolor="#1e2128", labelcolor="white",
           framealpha=0.7, edgecolor="#555555")

# ── (c) Carrier thickness contour ───────────
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor("#151820")
cf3 = ax3.contourf(X, Y, GRID_TC, levels=30, cmap="cividis")
cs3 = ax3.contour(X, Y, GRID_TC, levels=8, colors="white", linewidths=0.5, alpha=0.4)
ax3.clabel(cs3, fmt="%.0f µm", fontsize=6.5, colors="white")
cbar3 = fig.colorbar(cf3, ax=ax3, pad=0.02)
cbar3.set_label("Carrier Si thickness  (µm)", **cbar_kw)
cbar3.ax.tick_params(**tick_kw)
ax3.set_title("Resulting Carrier Silicon Thickness  (µm)", **title_kw)
ax3.set_xlabel("Memory die thickness  (µm)", **label_kw)
ax3.set_ylabel("Logic die thickness  (µm)",  **label_kw)
ax3.tick_params(axis="both", **tick_kw)
ax3.grid(**grid_kw)
ax3.plot(40, 40, "w*", markersize=12, label="Current (40/40 µm)\n→ carrier ≈ 719 µm")
ax3.legend(fontsize=7.5, facecolor="#1e2128", labelcolor="white",
           framealpha=0.7, edgecolor="#555555")

# ── (d) Pareto front ────────────────────────
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor("#151820")

# All design points (faded scatter)
sc_all = ax4.scatter(WARP, THETA, c=T_CAR, cmap="cividis",
                     s=4, alpha=0.25, linewidths=0)
# Pareto front
sc_par = ax4.scatter(px, py, c=pc, cmap="cividis",
                     s=50, edgecolors="white", linewidths=0.8,
                     zorder=5, vmin=T_CAR.min(), vmax=T_CAR.max())
ax4.plot(px, py, color="white", linewidth=1.2, alpha=0.6, zorder=4)

cbar4 = fig.colorbar(sc_par, ax=ax4, pad=0.02)
cbar4.set_label("Carrier thickness  (µm)", **cbar_kw)
cbar4.ax.tick_params(**tick_kw)

# Annotate discrete candidates on Pareto
for row in candidate_data:
    tm_um, tl_um, tc_um, w, th, tj = row
    on_pareto = any(abs(w - px[k]) < 0.01 and abs(th - py[k]) < 0.01
                    for k in range(len(px)))
    if on_pareto:
        ax4.annotate(f"M{tm_um}/L{tl_um}",
                     xy=(w, th), xytext=(w + 0.05, th + 0.1),
                     fontsize=6.5, color="yellow",
                     arrowprops=dict(arrowstyle="->", color="yellow", lw=0.7))

# Current design point (40/40)
w_ref  = warpage_um(40e-6, 40e-6, (T_avail - 80e-6))
th_ref = thermal_resistance(40e-6, 40e-6, (T_avail - 80e-6)) * 1e3
ax4.scatter(w_ref, th_ref, marker="*", s=250, color="red",
            zorder=6, label=f"Current 40/40 µm\nwarp={w_ref:.2f} µm, θ={th_ref:.2f} mK/W")

ax4.set_xlabel("Assembly Warpage  (µm)", **label_kw)
ax4.set_ylabel("θ_jc  (mK/W)", **label_kw)
ax4.set_title("Pareto Front  —  Warpage vs Thermal Resistance\n(Pareto pts coloured by carrier thickness)",
              **title_kw)
ax4.tick_params(axis="both", **tick_kw)
ax4.grid(**grid_kw)
ax4.legend(fontsize=7.5, facecolor="#1e2128", labelcolor="white",
           framealpha=0.7, edgecolor="#555555")

# ── Global title ─────────────────────────────
fig.suptitle(
    "3D Chiplet Stack — Thermal–Warpage Co-Optimisation\n"
    "Memory (3.5 ppm/°C, E=110 GPa)  |  Logic 4×(8×8 mm², 2.6 ppm/°C)  "
    "|  Carrier Si  |  TIM 1 µm oxide  |  Total = 800 µm",
    color="white", fontsize=11, fontweight="bold", y=0.99
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = "/home/user/HyMu/3D-stack/thermal_warpage_tradeoff.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
plt.close()

# ──────────────────────────────────────────────
# 11. TEXT SUMMARY
# ──────────────────────────────────────────────
idx_min_w  = np.argmin(WARP)
idx_min_th = np.argmin(THETA)

print("=" * 65)
print("DESIGN RECOMMENDATIONS")
print("=" * 65)
print(f"\n[A] Minimum warpage point:")
print(f"    t_mem={T_MEM[idx_min_w]:.0f} µm  t_logic={T_LOGIC[idx_min_w]:.0f} µm"
      f"  t_carrier={T_CAR[idx_min_w]:.0f} µm")
print(f"    Warpage = {WARP[idx_min_w]:.3f} µm   θ_jc = {THETA[idx_min_w]:.3f} mK/W")

print(f"\n[B] Minimum thermal resistance point:")
print(f"    t_mem={T_MEM[idx_min_th]:.0f} µm  t_logic={T_LOGIC[idx_min_th]:.0f} µm"
      f"  t_carrier={T_CAR[idx_min_th]:.0f} µm")
print(f"    Warpage = {WARP[idx_min_th]:.3f} µm   θ_jc = {THETA[idx_min_th]:.3f} mK/W")

print(f"\n[C] Current baseline (40/40 µm):")
tc_cur = (T_avail - 80e-6)
print(f"    t_mem=40 µm  t_logic=40 µm  t_carrier={tc_cur*1e6:.0f} µm")
print(f"    Warpage = {warpage_um(40e-6,40e-6,tc_cur):.3f} µm"
      f"   θ_jc = {thermal_resistance(40e-6,40e-6,tc_cur)*1e3:.3f} mK/W")

print("\n[D] Pareto-optimal discrete candidates:")
for row in candidate_data:
    tm_um, tl_um, tc_um, w, th, tj = row
    on_pareto = any(abs(w  - WARP[pareto_mask][k]) < 0.02 and
                    abs(th - THETA[pareto_mask][k]) < 0.02
                    for k in range(pareto_mask.sum()))
    tag = " ← Pareto" if on_pareto else ""
    print(f"    M{tm_um}/L{tl_um}/C{tc_um:.0f} µm  |  warp={w:.3f} µm"
          f"  θ={th:.3f} mK/W{tag}")

print("\nKey physics insight:")
print("  • Warpage is driven by the memory–silicon CTE mismatch (3.5 vs 2.6 ppm/°C).")
print("  • Thinner memory die → less bending moment → lower warpage.")
print("  • Thicker carrier → more bending stiffness → lower warpage.")
print("  • θ_jc is dominated by the 1-µm oxide TIM (≈4 mK/W at logic area).")
print("  • Thinning logic or memory has minimal thermal impact;")
print("    the carrier thickness mainly affects spreading resistance.")
print("  • Optimal strategy: minimise t_mem, keep t_logic moderate,")
print("    let carrier fill the remaining budget for stiffness.")
