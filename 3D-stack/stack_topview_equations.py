"""
4-panel explainer figure:
  1. Top view — die layout (memory + 4 logic dies, dimensions)
  2. Side cross-section — how warpage δ is measured
  3. Warpage equations (ABD plate theory)
  4. Thermal resistance + frequency uplift equations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0e1117')

# ─── Layout ───────────────────────────────────────────────────────────────
gs = plt.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                  left=0.06, right=0.97, top=0.93, bottom=0.04)

AX_TOP   = fig.add_subplot(gs[0, 0])   # top view
AX_XSEC  = fig.add_subplot(gs[0, 1])   # cross-section warpage
AX_EQ1   = fig.add_subplot(gs[1, 0])   # warpage equations
AX_EQ2   = fig.add_subplot(gs[1, 1])   # thermal + freq equations

BG = '#141720'
for ax in [AX_TOP, AX_XSEC, AX_EQ1, AX_EQ2]:
    ax.set_facecolor(BG)
    ax.tick_params(colors='#666', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#333')

# ══════════════════════════════════════════════════════════════════════════
# PANEL 1 — TOP VIEW
# ══════════════════════════════════════════════════════════════════════════
ax = AX_TOP
ax.set_xlim(-2, 30);  ax.set_ylim(-3, 38)
ax.set_aspect('equal')
ax.set_title('Top View — Die Layout  (all dims in mm)', color='white',
             fontsize=11, fontweight='bold')
ax.set_xlabel('X (mm)', color='#aaa', fontsize=8)
ax.set_ylabel('Y (mm)', color='#aaa', fontsize=8)

# Reticle boundary (26×33 mm)
ret = patches.Rectangle((-1, -0.5), 26, 33, lw=1.2,
                         edgecolor='#888888', facecolor='none',
                         linestyle='--')
ax.add_patch(ret)
ax.text(-1+13, -0.5-1.2, '26 mm (reticle)', color='#888', fontsize=7,
        ha='center')
ax.text(-1-1.2, -0.5+16.5, '33 mm\n(reticle)', color='#888', fontsize=7,
        ha='right', va='center', rotation=90)

# Memory die (24×32 mm), origin at (0,0)
mem = patches.Rectangle((0, 0), 24, 32, lw=1.8,
                          edgecolor='#5599ee', facecolor='#1a3a6a', alpha=0.85)
ax.add_patch(mem)
ax.text(12, 16, 'Memory die\n24 × 32 mm\nCTE = 3.5 ppm/°C\nE = 110 GPa',
        color='#88bbff', fontsize=7.5, ha='center', va='center',
        fontweight='bold')

# 4 Logic dies: 8×8mm, 2×2 grid
# Width margins: (24-16)/2=4mm; Height margins: (32-16)/2=8mm
logic_orig = [(4, 8), (12, 8), (4, 16), (12, 16)]
for k, (ox, oy) in enumerate(logic_orig):
    lg = patches.Rectangle((ox, oy), 8, 8, lw=1.5,
                             edgecolor='#ffaa33', facecolor='#5a3500', alpha=0.9)
    ax.add_patch(lg)
    label = f'Logic {k+1}\n8×8 mm\nCTE=2.6\nE=130 GPa'
    ax.text(ox+4, oy+4, label, color='#ffd080', fontsize=6,
            ha='center', va='center')

# Dimension arrows — width (24mm)
ax.annotate('', xy=(24, -2), xytext=(0, -2),
            arrowprops=dict(arrowstyle='<->', color='white', lw=1))
ax.text(12, -2.7, '24 mm (usable)', color='white', ha='center', fontsize=7.5)

# Dimension arrows — height (32mm)
ax.annotate('', xy=(25.5, 32), xytext=(25.5, 0),
            arrowprops=dict(arrowstyle='<->', color='white', lw=1))
ax.text(26.5, 16, '32 mm\n(usable)', color='white', ha='left', va='center',
        fontsize=7.5)

# Margins
ax.annotate('', xy=(4, 6.5), xytext=(0, 6.5),
            arrowprops=dict(arrowstyle='<->', color='#aaaaaa', lw=0.8))
ax.text(2, 5.8, '4mm', color='#aaa', ha='center', fontsize=6.5)

ax.annotate('', xy=(12, 6.5), xytext=(4, 6.5),
            arrowprops=dict(arrowstyle='<->', color='#ffaa33', lw=0.8))
ax.text(8, 5.8, '8mm', color='#ffaa33', ha='center', fontsize=6.5)

ax.annotate('', xy=(3, 8), xytext=(3, 0),
            arrowprops=dict(arrowstyle='<->', color='#aaaaaa', lw=0.8))
ax.text(2.2, 4, '8mm\nmargin', color='#aaa', ha='right', fontsize=6.5)

# Gap between dies
ax.annotate('', xy=(12, 12.5), xytext=(12, 16),
            arrowprops=dict(arrowstyle='<->', color='#888', lw=0.8))
ax.text(12.3, 14.2, 'gap\n0 mm\n(butted)', color='#888', fontsize=6, va='center')

# Coverage annotation
ax.text(12, 33.5,
        'Logic coverage: 4×(8×8) = 256 mm²  /  24×32 = 768 mm²  → φ = 1/3',
        color='#cccccc', fontsize=7.5, ha='center')

# Legend
leg = [Line2D([0],[0], color='#5599ee', lw=2, label='Memory die (blue)'),
       patches.Patch(facecolor='#5a3500', edgecolor='#ffaa33',
                     label='Logic dies × 4 (amber)'),
       Line2D([0],[0], color='#888', lw=1.2, ls='--',
              label='Reticle boundary 26×33 mm')]
ax.legend(handles=leg, fontsize=7, facecolor='#1e2130',
          edgecolor='#444', labelcolor='white', loc='lower right')
ax.grid(color='#222233', lw=0.4, ls=':')

# ══════════════════════════════════════════════════════════════════════════
# PANEL 2 — CROSS-SECTION: LAYERS + WARPAGE MEASUREMENT
# ══════════════════════════════════════════════════════════════════════════
ax = AX_XSEC
ax.set_xlim(-1.5, 26);  ax.set_ylim(-2.5, 12)
ax.set_aspect('equal')
ax.set_title('Cross-Section (Y=16 slice) + Warpage Measurement',
             color='white', fontsize=11, fontweight='bold')
ax.set_xlabel('X (mm)', color='#aaa', fontsize=8)
ax.set_ylabel('Z — thickness (scaled ×12)', color='#aaa', fontsize=8)

# True thicknesses in µm; display in mm scaled ×12
SC = 12 / 1000   # 1 µm → 0.012 mm (scale for display)
T_MEM_d = 40  * SC       # 0.48 mm
T_TIM_d = max(1*SC, 0.08) # min thickness for visibility
T_LOG_d = 40  * SC
T_CAR_d = 719 * SC       # 8.628 mm  → clip to 6 for visual space
T_CAR_d = min(T_CAR_d, 5.5)

z0 = 0
# Memory
rect = patches.Rectangle((0, z0), 24, T_MEM_d,
                           fc='#1a3a6a', ec='#5599ee', lw=1.3)
ax.add_patch(rect)
ax.text(-0.2, z0+T_MEM_d/2, 'Memory\n40 µm\nCTE 3.5', color='#88bbff',
        fontsize=6.5, ha='right', va='center')
z0 += T_MEM_d

# TIM under the two middle dies (x=4-20 in 2×2 grid, slice at y=16)
for ox in [4, 12]:
    tim = patches.Rectangle((ox, z0), 8, T_TIM_d,
                              fc='#555555', ec='#aaaaaa', lw=0.8)
    ax.add_patch(tim)
ax.text(-0.2, z0+T_TIM_d/2, 'TIM 1µm', color='#aaa', fontsize=6,
        ha='right', va='center')
z0 += T_TIM_d

# Logic dies
for ox in [4, 12]:
    lg = patches.Rectangle((ox, z0), 8, T_LOG_d,
                             fc='#5a3500', ec='#ffaa33', lw=1.3)
    ax.add_patch(lg)
ax.text(-0.2, z0+T_LOG_d/2, 'Logic\n40 µm\nCTE 2.6', color='#ffd080',
        fontsize=6.5, ha='right', va='center')

# Gap regions (no logic die)
for ox, ow in [(0,4),(8,4),(20,4)]:
    gap_r = patches.Rectangle((ox, z0), ow, T_LOG_d,
                                fc='#0e1117', ec='#333344', lw=0.5,
                                linestyle=':')
    ax.add_patch(gap_r)
ax.text(2, z0+T_LOG_d/2, 'air/\nunderfill', color='#555', fontsize=5.5,
        ha='center', va='center')
z0 += T_LOG_d

z_car0 = z0
# Si carrier (shown as flat reference first)
car_si  = patches.Rectangle((0, z_car0), 24, T_CAR_d,
                              fc='#1a2a4a', ec='#5588ff', lw=1.2, alpha=0.5)
ax.add_patch(car_si)
ax.text(-0.2, z_car0+T_CAR_d/2, 'Carrier\n719 µm', color='#88aaff',
        fontsize=6.5, ha='right', va='center')

z_top = z_car0 + T_CAR_d

# ── Warpage geometry ────────────────────────────────────────────────────
# Show the bent neutral surface for Si case (concave up at memory → convex at carrier top)
# Sign convention: Si → memory contracts more → bottom bows up
# In cross-section: edges of the bottom surface go DOWN, centre stays

# For Si carrier: δ ~ 14.6 µm → in display units:
delta_si_d  = 14.6 * SC   # display mm
delta_sic_d = 8.0  * SC

x_arr = np.linspace(0, 24, 200)
xc = 12   # centre

def bow(x, delta, sign):
    # parabolic bow: z(x) = sign * delta * (1 - (x-xc)²/(xc²)) normalized
    # At x=0 and x=24: z = 0; At x=12: z = sign*delta
    return sign * delta * (1 - ((x - xc)/xc)**2)

# Bent bottom surface of full package (memory bottom)
z_bot_si  = bow(x_arr, delta_si_d,  +1)  # concave up at memory = centre raised
z_bot_sic = bow(x_arr, delta_sic_d, -1)  # convex = centre down

z_base_mem = 0

# Draw bent bottom lines
ax.plot(x_arr, z_base_mem + z_bot_si,
        color='#5588ff', lw=2, label='Si bow (memory bottom)')
ax.plot(x_arr, z_base_mem + z_bot_sic,
        color='#ff8833', lw=2, ls='--', label='SiC bow (memory bottom)')

# Reference flat line
ax.axhline(z_base_mem, color='white', lw=0.8, ls=':', alpha=0.5)
ax.text(24.2, z_base_mem, 'flat\nref', color='white', fontsize=6, va='center')

# Measurement arrows for Si
ax.annotate('', xy=(12, z_base_mem + delta_si_d),
            xytext=(12, z_base_mem),
            arrowprops=dict(arrowstyle='<->', color='#88aaff', lw=1.2))
ax.text(13, z_base_mem + delta_si_d/2,
        f'δ_Si ≈ {14.6:.1f} µm\n(centre rise)', color='#88aaff',
        fontsize=6.5, va='center')

# Measurement arrows for SiC
ax.annotate('', xy=(12, z_base_mem + z_bot_sic[100]),
            xytext=(12, z_base_mem),
            arrowprops=dict(arrowstyle='<->', color='#ffaa66', lw=1.2))
ax.text(13, z_base_mem + z_bot_sic[100]/2 - 0.3,
        f'δ_SiC ≈ {8.0:.1f} µm\n(centre drop)', color='#ffaa66',
        fontsize=6.5, va='top')

# Diagonal L_char annotation
ax.annotate('', xy=(24, z_top+0.3), xytext=(0, z_top+0.3),
            arrowprops=dict(arrowstyle='<->', color='#aaaaaa', lw=1))
ax.text(12, z_top+0.5, 'W = 24 mm  (L_char = √(24²+32²) = 40 mm diagonal)',
        color='#aaa', fontsize=6.5, ha='center')

ax.legend(fontsize=7, facecolor='#1e2130', edgecolor='#444',
          labelcolor='white', loc='upper right')
ax.grid(color='#222233', lw=0.3, ls=':')

# ══════════════════════════════════════════════════════════════════════════
# PANEL 3 — WARPAGE EQUATIONS
# ══════════════════════════════════════════════════════════════════════════
ax = AX_EQ1
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.axis('off')
ax.set_title('Warpage Model — ABD Plate Theory', color='white',
             fontsize=11, fontweight='bold')

EQ_COLOR   = '#e8f4ff'
HEAD_COLOR = '#ffcc66'
SUB_COLOR  = '#aaddff'
ANNO_COLOR = '#aaaaaa'

def eqbox(ax, x, y, text, color=EQ_COLOR, fs=8.5, ha='left'):
    ax.text(x, y, text, color=color, fontsize=fs, ha=ha, va='top',
            transform=ax.transAxes,
            fontfamily='monospace')

def head(ax, x, y, text):
    ax.text(x, y, text, color=HEAD_COLOR, fontsize=9.5, fontweight='bold',
            va='top', transform=ax.transAxes)

def anno(ax, x, y, text):
    ax.text(x, y, text, color=ANNO_COLOR, fontsize=7.5, va='top',
            transform=ax.transAxes, style='italic')

head(ax, 0.03, 0.97, '① Layer definitions (bottom z=0 → top)')
eqbox(ax, 0.03, 0.90,
      'For each layer i:  Ē_i = E_i / (1 − ν_i)   [biaxial modulus]')
eqbox(ax, 0.03, 0.84,
      'Area fraction:  φ = A_logic / A_carrier = 256/768 ≈ 1/3')
eqbox(ax, 0.03, 0.78,
      '  Layer       Ē_eff             α (ppm/°C)   t (µm)\n'
      '  Memory      Ē_mem             3.5          t_mem\n'
      '  TIM         Ē_TIM × φ         0.55         1\n'
      '  Logic       Ē_Si  × φ         2.6          t_logic\n'
      '  Carrier     Ē_car             2.6 or 4.2   t_carrier',
      fs=7.5)

head(ax, 0.03, 0.55, '② ABD plate stiffness matrix  (about z = 0)')
eqbox(ax, 0.03, 0.49,
      'A = Σ Ē_i · t_i                          [N/m]\n'
      'B = Σ Ē_i · t_i · z_mid_i               [N]\n'
      'D = Σ Ē_i · (t_i³/12 + t_i · z_mid_i²)  [N·m]')

head(ax, 0.03, 0.36, '③ Thermal force & moment resultants')
eqbox(ax, 0.03, 0.30,
      'N_T = ΔT · Σ Ē_i · α_i · t_i\n'
      'M_T = ΔT · Σ Ē_i · α_i · t_i · z_mid_i\n'
      '\n'
      'ΔT = 235 °C  (T_reflow=260°C → T_room=25°C)')

head(ax, 0.03, 0.17, '④ Curvature & warpage')
eqbox(ax, 0.03, 0.11,
      'κ = (A · M_T  −  B · N_T) / (A · D  −  B²)   [1/m]')
eqbox(ax, 0.03, 0.06,
      'δ = κ · L_char² / 8      [m]   L_char = 40 mm diagonal')
anno(ax, 0.03, 0.01,
     'Positive κ → concave-up at memory face (Si carrier); '
     'Negative κ → convex-up (SiC carrier)')

# Divider box
for spine in ['left','right','top','bottom']:
    ax.spines[spine].set_color('#334455')
ax.set_facecolor('#0c1018')

# ══════════════════════════════════════════════════════════════════════════
# PANEL 4 — THERMAL + FREQUENCY EQUATIONS
# ══════════════════════════════════════════════════════════════════════════
ax = AX_EQ2
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.axis('off')
ax.set_title('Thermal Model + Frequency Uplift Equations', color='white',
             fontsize=11, fontweight='bold')
ax.set_facecolor('#0c1018')

head(ax, 0.03, 0.97, '① Thermal resistance: logic junction → carrier top  (θ_jc)')
eqbox(ax, 0.03, 0.91,
      'R_logic   = t_logic / (k_Si  × A_logic)\n'
      'R_TIM     = t_TIM   / (k_TIM × A_logic)\n'
      '\n'
      '# Lateral spreading in carrier (heat spreads from A_logic → A_carrier)\n'
      'A_eff = A_logic + (A_carrier − A_logic) × [1 − exp(−t_carrier / t_spr)]\n'
      't_spr = 2 mm  (≈ die half-width / 2)\n'
      'R_carrier = t_carrier / (k_carrier × A_eff)\n'
      '\n'
      'θ_jc = R_logic + R_TIM + R_carrier')

anno(ax, 0.03, 0.55,
     'Spreading term: exp decay captures limited lateral spreading in thin carrier.\n'
     'At t_carrier → 0: A_eff → A_logic (1-D, no spread).\n'
     'At t_carrier → ∞: A_eff → A_carrier (full spread).')

head(ax, 0.03, 0.48, '② Junction temperature')
eqbox(ax, 0.03, 0.42,
      'T_j = T_amb + P_logic × (θ_jc + θ_heatsink)\n'
      '\n'
      'T_amb     = 25 °C\n'
      'P_logic   = 180 W\n'
      'θ_heatsink= 0.30 K/W  (air-cooled)\n'
      '\n'
      'Si  carrier baseline: T_j ≈ 82.0 °C  (θ_jc = 16.6 mK/W)\n'
      'SiC carrier baseline: T_j ≈ 80.7 °C  (θ_jc =  9.3 mK/W)\n'
      '                      ΔT_j ≈ 1.3 °C')

head(ax, 0.03, 0.21, '③ Frequency uplift  (CV²f / thermal model)')
eqbox(ax, 0.03, 0.15,
      'f(T_j) = f_ref × [1 + α_f × (T_j_baseline − T_j)]\n'
      '\n'
      'f_ref        = 3.5 GHz  (at T_j_baseline with Si carrier)\n'
      'α_f          = 0.4 %/°C  (CMOS frequency thermal sensitivity)\n'
      'T_j_baseline = 82.0 °C\n'
      '\n'
      'SiC uplift: Δf = 3.5 × 0.004 × 1.3 ≈ 18 MHz')
anno(ax, 0.03, 0.03,
     'CV²f link: P = α·C·V²·f → lower T_j allows ↓V_dd or ↑f at same power budget.')

for spine in ['left','right','top','bottom']:
    ax.spines[spine].set_color('#334455')

# ── Global title ──────────────────────────────────────────────────────────
fig.suptitle(
    '3D Chiplet Stack — Top View · Cross-Section · Warpage & Thermal Equations',
    color='white', fontsize=12, fontweight='bold', y=0.97
)

out = '/home/user/HyMu/3D-stack/topview_equations.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved → {out}')
plt.close()
