"""
4-panel 3-D visualisation of the chiplet stack:
  Fig 1 — Si carrier stack (exploded, with CTE labels)
  Fig 2 — SiC carrier stack (exploded, with CTE labels)
  Fig 3 — Warpage shape: Si carrier (concave-up memory side)
  Fig 4 — Warpage shape: SiC carrier (convex-up / reversed bow)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings("ignore")

# ── Geometry (mm) ─────────────────────────────────────────────────────────
W, L      = 24.0, 32.0          # memory / carrier footprint
wl, ll    = 8.0,  8.0           # single logic die
gap       = 1.5                 # exploded-view gap between layers (mm)

# Thicknesses in mm (scaled ×5 for visibility, true µm shown in labels)
SC = 5.0          # scale factor µm → mm for display

T_MEM  = 40e-3  * SC   # 40 µm → 0.2 mm display
T_TIM  = 1e-3   * SC   # 1 µm  → 0.005 mm  (draw thicker for visibility)
T_LOG  = 40e-3  * SC
T_CAR  = 719e-3 * SC   # 719 µm → 3.595 mm

T_TIM_draw = 0.08       # forced min thickness for TIM visibility

# Logic die positions (2×2 on the 24×32 carrier, centred)
# Margins: (24-2*8)/2 = 4 mm sides, (32-2*8)/2 = 8 mm top/bottom
logic_origins = [(4.0, 8.0), (12.0, 8.0), (4.0, 16.0), (12.0, 16.0)]

# ── Colour palette ────────────────────────────────────────────────────────
C_MEM  = '#4a90d9'   # blue
C_TIM  = '#aaaaaa'   # grey
C_LOG  = '#e8a838'   # amber
C_CAR_SI  = '#5dade2'  # light blue
C_CAR_SIC = '#e67e22'  # orange

ALPHA_FACE = 0.82
ALPHA_EDGE = 1.0

# ── Helper: draw a solid box ───────────────────────────────────────────────
def box_faces(x0, x1, y0, y1, z0, z1):
    """Return list of 6 face vertex arrays for a rectangular box."""
    verts = [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],  # bottom
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],  # top
        [(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],  # left
        [(x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0)],  # right
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],  # front
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],  # back
    ]
    return verts

def add_box(ax, x0, x1, y0, y1, z0, z1, color, alpha=ALPHA_FACE, lw=0.4):
    faces = box_faces(x0, x1, y0, y1, z0, z1)
    poly = Poly3DCollection(faces, alpha=alpha, linewidths=lw,
                            edgecolors='#222222')
    poly.set_facecolor(color)
    ax.add_collection3d(poly)

def arrow3d(ax, x, y, z0, z1, color='white', lw=1.2):
    ax.plot([x, x], [y, y], [z0, z1], color=color, lw=lw)
    ax.quiver(x, y, z1, 0, 0, 0.05, color=color, length=0.0, arrow_length_ratio=0)

def dim_arrow(ax, x, y, z0, z1, label, color='white', side='right'):
    ax.annotate3D = None   # no-op placeholder
    mid = (z0 + z1) / 2
    ax.plot([x,x],[y,y],[z0,z1], color=color, lw=0.8, ls='--')
    ax.text(x, y, mid, f'  {label}', color=color, fontsize=6.5, va='center')

# ── Draw stack (exploded) ──────────────────────────────────────────────────
def draw_stack(ax, carrier_color, carrier_label, cte_labels, title):
    ax.set_facecolor('#0e1117')

    z = 0.0

    # Memory die
    z_mem0 = z
    add_box(ax, 0, W, 0, L, z_mem0, z_mem0+T_MEM, C_MEM)
    ax.text(W/2, -2, z_mem0+T_MEM/2,
            f"Memory die\n24×32 mm\nCTE={cte_labels['mem']} ppm/°C\n40 µm",
            color='#aaddff', fontsize=6, ha='center', va='center')
    z += T_MEM + gap

    # TIM
    z_tim0 = z
    for ox, oy in logic_origins:
        add_box(ax, ox, ox+wl, oy, oy+ll, z_tim0, z_tim0+T_TIM_draw,
                C_TIM, alpha=0.6)
    ax.text(W/2, -2, z_tim0+T_TIM_draw/2,
            'Oxide TIM  1 µm', color='#cccccc', fontsize=6, ha='center')
    z += T_TIM_draw + gap

    # Logic dies
    z_log0 = z
    for k, (ox, oy) in enumerate(logic_origins):
        add_box(ax, ox, ox+wl, oy, oy+ll, z_log0, z_log0+T_LOG, C_LOG)
    ax.text(W/2, -2, z_log0+T_LOG/2,
            f"Logic dies (×4)\n8×8 mm each\nCTE={cte_labels['logic']} ppm/°C\n40 µm",
            color='#ffd080', fontsize=6, ha='center', va='center')
    z += T_LOG + gap

    # Carrier
    z_car0 = z
    add_box(ax, 0, W, 0, L, z_car0, z_car0+T_CAR, carrier_color)
    ax.text(W/2, -2, z_car0+T_CAR/2,
            f"{carrier_label}\n24×32 mm\nCTE={cte_labels['car']} ppm/°C\n719 µm",
            color='white', fontsize=6, ha='center', va='center')

    z_top = z_car0 + T_CAR

    # CTE gradient indicator (right side)
    cte_vals = [float(cte_labels['mem']), float(cte_labels['logic']),
                float(cte_labels['car'])]
    layer_z   = [z_mem0+T_MEM/2, z_log0+T_LOG/2, z_car0+T_CAR/2]
    cmap_cte  = plt.cm.RdYlGn
    cte_min, cte_max = 2.0, 4.5
    for cv, lz in zip(cte_vals, layer_z):
        norm_c = (cv - cte_min) / (cte_max - cte_min)
        col = cmap_cte(norm_c)
        ax.scatter([W+2], [L+1], [lz], color=col, s=60, zorder=5)
        ax.text(W+3, L+1, lz, f'{cv}', color=col, fontsize=7, va='center')

    ax.text(W+2, L+1, z_top+0.3, 'CTE\nppm/°C', color='white',
            fontsize=6.5, ha='center')

    ax.set_xlim(-1, W+5); ax.set_ylim(-4, L+3)
    ax.set_zlim(-0.5, z_top+0.8)
    ax.set_xlabel('X (mm)', color='#888', fontsize=7, labelpad=2)
    ax.set_ylabel('Y (mm)', color='#888', fontsize=7, labelpad=2)
    ax.set_zlabel('Z (stack)', color='#888', fontsize=7, labelpad=2)
    ax.tick_params(colors='#666', labelsize=6)
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=6)
    ax.view_init(elev=22, azim=-50)
    ax.grid(False)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

# ── Warpage surface ────────────────────────────────────────────────────────
def draw_warpage(ax, sign, carrier_label, kappa_um_per_mm2, title):
    """
    sign  = +1  → concave-up at memory (Si case)
    sign  = -1  → convex-up  at memory (SiC case)
    kappa_um_per_mm2: curvature magnitude
    """
    ax.set_facecolor('#0e1117')

    Nx, Ny = 40, 40
    xs = np.linspace(0, W, Nx)
    ys = np.linspace(0, L, Ny)
    Xg, Yg = np.meshgrid(xs, ys)

    # Centre of plate
    xc, yc = W/2, L/2
    # Deflection: δ = κ/2 * ((x-xc)²+(y-yc)²) — bowl or dome shape
    r2 = (Xg - xc)**2 + (Yg - yc)**2
    Z  = sign * kappa_um_per_mm2 * r2    # µm deflection

    # Scale Z to mm for display (×0.05 so shape is visible vs footprint)
    Zd = Z * 0.05

    # Colour by deflection
    norm = plt.Normalize(Z.min(), Z.max())
    cmap = plt.cm.RdBu_r if sign > 0 else plt.cm.RdBu
    colors = cmap(norm(Z))

    ax.plot_surface(Xg, Yg, Zd, facecolors=colors,
                    rstride=1, cstride=1, antialiased=True,
                    linewidth=0, alpha=0.92)

    # Flat reference plane
    ax.plot_surface(Xg, Yg, np.zeros_like(Zd),
                    color='#334455', alpha=0.15,
                    linewidth=0, rstride=4, cstride=4)

    # Arrows showing bow direction at corners and centre
    pts = [(0,0),(W,0),(0,L),(W,L),(xc,yc)]
    for px, py in pts:
        r2p = (px-xc)**2 + (py-yc)**2
        zp  = sign * kappa_um_per_mm2 * r2p * 0.05
        dz  = 0.08 * sign if (px==xc and py==yc) else -0.08*sign
        ax.quiver(px, py, zp, 0, 0, dz,
                  color='#ffff88', arrow_length_ratio=0.5, lw=1.5)

    delta_max = kappa_um_per_mm2 * ((W/2)**2 + (L/2)**2)
    bow_dir   = 'Concave↑ at memory (sad-face)' if sign > 0 else 'Convex↑ at memory (smile-face)'
    ax.text2D(0.5, 0.01,
              f'Max edge deflection ≈ {delta_max:.1f} µm\n{bow_dir}',
              transform=ax.transAxes, ha='center', color='white', fontsize=7.5)

    # Colourbar-like side strip
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, pad=0.08, fraction=0.03, shrink=0.6)
    cb.set_label('Deflection (µm)', color='#cccccc', fontsize=7)
    cb.ax.tick_params(colors='#888888', labelsize=6.5)

    ax.set_xlabel('X (mm)', color='#888', fontsize=7)
    ax.set_ylabel('Y (mm)', color='#888', fontsize=7)
    ax.set_zlabel('Deflection (scaled)', color='#888', fontsize=7)
    ax.tick_params(colors='#666', labelsize=6)
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=6)
    ax.view_init(elev=28, azim=-55)
    ax.grid(False)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

# ── Compute curvature for each carrier ────────────────────────────────────
# Simplified Stoney-like curvature (normalised so we can show shape)
# κ_Si  ~ 14.6 µm over 40mm diagonal → κ/r² ≈ 14.6/(20²+20²) mm units
# κ_SiC ~ 8.0 µm over same → use proportional values
# Using r² in mm², δ in µm → κ in µm/mm²
def kappa_from_delta(delta_um, W_mm, L_mm):
    r2_max = (W_mm/2)**2 + (L_mm/2)**2
    return delta_um / r2_max

k_Si  = kappa_from_delta(14.6, W, L)
k_SiC = kappa_from_delta(8.0,  W, L)

# ── Build figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0e1117')

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax4 = fig.add_subplot(2, 2, 4, projection='3d')

draw_stack(ax1, C_CAR_SI,
           'Silicon Carrier',
           {'mem': '3.5', 'logic': '2.6', 'car': '2.6'},
           'Stack — Silicon Carrier\n(CTE gradient: 3.5 → 2.6 ppm/°C, top same as logic)')

draw_stack(ax2, C_CAR_SIC,
           '4H-SiC Carrier',
           {'mem': '3.5', 'logic': '2.6', 'car': '4.2'},
           'Stack — 4H-SiC Carrier\n(CTE gradient: 3.5 → 4.2 ppm/°C, carrier > memory)')

draw_warpage(ax3, sign=+1, carrier_label='Si',
             kappa_um_per_mm2=k_Si,
             title='Warpage — Silicon Carrier\n'
                   'Memory (3.5) > Carrier (2.6): memory contracts more on cooling\n'
                   '→ Concave at memory face (edges lift, centre drops)')

draw_warpage(ax4, sign=-1, carrier_label='SiC',
             kappa_um_per_mm2=k_SiC,
             title='Warpage — 4H-SiC Carrier\n'
                   'Carrier (4.2) > Memory (3.5): SiC contracts more on cooling\n'
                   '→ Convex at memory face (dome up, edges drop) — bow REVERSED')

fig.suptitle(
    '3D Chiplet Stack — Layer Structure & Warpage Visualisation\n'
    'Memory 24×32 mm (40 µm)  |  Logic 4×(8×8 mm, 40 µm)  '
    '|  Carrier 719 µm  |  ΔT = 235 °C (reflow → room)',
    color='white', fontsize=10.5, fontweight='bold', y=1.001
)

# Legend patches
patches = [
    mpatches.Patch(color=C_MEM,     label='Memory die  (CTE=3.5 ppm/°C, E=110 GPa)'),
    mpatches.Patch(color=C_TIM,     label='Oxide TIM   (CTE=0.55 ppm/°C, 1 µm)'),
    mpatches.Patch(color=C_LOG,     label='Logic dies  (CTE=2.6 ppm/°C, E=130 GPa)'),
    mpatches.Patch(color=C_CAR_SI,  label='Si Carrier  (CTE=2.6 ppm/°C, E=130 GPa, k=150 W/mK)'),
    mpatches.Patch(color=C_CAR_SIC, label='SiC Carrier (CTE=4.2 ppm/°C, E=450 GPa, k=400 W/mK)'),
]
fig.legend(handles=patches, loc='lower center', ncol=3,
           facecolor='#1a1d24', edgecolor='#444',
           labelcolor='white', fontsize=8, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = '/home/user/HyMu/3D-stack/stack_3d_viz.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved → {out}')
plt.close()
