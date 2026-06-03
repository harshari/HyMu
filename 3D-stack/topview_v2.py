"""
Updated top-view + cross-section + equations — new die layout:
  XCD  (compute):  24×16 mm  — top half of memory die
  CCD1 (cache):    12×16 mm  — bottom-left
  CCD2 (cache):    12×16 mm  — bottom-right
  Total coverage:  768 mm²   = 24×32 mm  → φ = 1.0  (full tiling!)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

# ── Layout ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0e1117')

from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30,
              left=0.06, right=0.97, top=0.93, bottom=0.04)

AX_TOP  = fig.add_subplot(gs[0, 0])   # top view
AX_XSEC = fig.add_subplot(gs[0, 1])   # cross-section
AX_EQ1  = fig.add_subplot(gs[1, 0])   # warpage eqs
AX_EQ2  = fig.add_subplot(gs[1, 1])   # thermal + freq eqs

BG = '#141720'
for ax in [AX_TOP, AX_XSEC, AX_EQ1, AX_EQ2]:
    ax.set_facecolor(BG)
    ax.tick_params(colors='#666', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#333')

# ─────────────────────────────────────────────────────────────────────────
# PANEL 1 — TOP VIEW
# ─────────────────────────────────────────────────────────────────────────
ax = AX_TOP
ax.set_xlim(-3, 31);  ax.set_ylim(-4, 38)
ax.set_aspect('equal')
ax.set_title('Top View — Die Layout  (all dims in mm)', color='white',
             fontsize=11, fontweight='bold')
ax.set_xlabel('X  (mm)', color='#aaa', fontsize=8)
ax.set_ylabel('Y  (mm)', color='#aaa', fontsize=8)
ax.grid(color='#1e2030', lw=0.5, ls=':')

# Reticle boundary  26×33  (shifted so memory die starts at 0,0)
reticle = patches.Rectangle((-1, -0.5), 26, 33, lw=1.2,
                              edgecolor='#666666', facecolor='none', ls='--')
ax.add_patch(reticle)
ax.text(12, -0.5-1.5, '26 mm  (reticle boundary)', color='#666',
        ha='center', fontsize=7)
ax.text(-1-1.2, 16, '33 mm\n(reticle)', color='#666', ha='right',
        va='center', fontsize=7, rotation=90)

# Memory die (base): 24×32 mm
mem = patches.Rectangle((0, 0), 24, 32, lw=2.2,
                          edgecolor='#4477cc', facecolor='#0d2040', alpha=0.9)
ax.add_patch(mem)

# ── XCD:  24×16 mm, top half  (y = 16 → 32) ──
xcd = patches.Rectangle((0, 16), 24, 16, lw=2,
                          edgecolor='#ff8844', facecolor='#4a1a00', alpha=0.93)
ax.add_patch(xcd)
ax.text(12, 24,
        'XCD  (Compute Die)\n24 × 16 mm\nCTE = 2.6 ppm/°C\nE = 130 GPa\n~150 W',
        color='#ffbb77', fontsize=8, ha='center', va='center', fontweight='bold')

# ── CCD1: 12×16 mm, bottom-left  (x=0, y=0) ──
ccd1 = patches.Rectangle((0, 0), 12, 16, lw=2,
                           edgecolor='#44cc88', facecolor='#00301a', alpha=0.93)
ax.add_patch(ccd1)
ax.text(6, 8,
        'CCD 1\n12 × 16 mm\nCTE = 2.6\n~15 W',
        color='#88ffbb', fontsize=7.5, ha='center', va='center', fontweight='bold')

# ── CCD2: 12×16 mm, bottom-right  (x=12, y=0) ──
ccd2 = patches.Rectangle((12, 0), 12, 16, lw=2,
                           edgecolor='#44cc88', facecolor='#00301a', alpha=0.93)
ax.add_patch(ccd2)
ax.text(18, 8,
        'CCD 2\n12 × 16 mm\nCTE = 2.6\n~15 W',
        color='#88ffbb', fontsize=7.5, ha='center', va='center', fontweight='bold')

# ── Divider lines ──
ax.plot([0, 24], [16, 16], color='white', lw=1.2, ls='--', alpha=0.7)
ax.plot([12,12], [0, 16],  color='white', lw=1.2, ls='--', alpha=0.7)

# ── Dimension arrows ──
# Width 24
ax.annotate('', xy=(24, -2.2), xytext=(0, -2.2),
            arrowprops=dict(arrowstyle='<->', color='white', lw=1.2))
ax.text(12, -2.9, '24 mm', color='white', ha='center', fontsize=8.5)

# Height 32
ax.annotate('', xy=(25.8, 32), xytext=(25.8, 0),
            arrowprops=dict(arrowstyle='<->', color='white', lw=1.2))
ax.text(26.6, 16, '32 mm', color='white', ha='left', va='center', fontsize=8.5)

# XCD height 16
ax.annotate('', xy=(-1.5, 32), xytext=(-1.5, 16),
            arrowprops=dict(arrowstyle='<->', color='#ffbb77', lw=1))
ax.text(-2, 24, '16', color='#ffbb77', ha='right', va='center', fontsize=8)

# CCD height 16
ax.annotate('', xy=(-1.5, 16), xytext=(-1.5, 0),
            arrowprops=dict(arrowstyle='<->', color='#88ffbb', lw=1))
ax.text(-2, 8, '16', color='#88ffbb', ha='right', va='center', fontsize=8)

# XCD width 24
ax.annotate('', xy=(24, 33.5), xytext=(0, 33.5),
            arrowprops=dict(arrowstyle='<->', color='#ffbb77', lw=1))
ax.text(12, 34.2, '24 mm  (XCD)', color='#ffbb77', ha='center', fontsize=8)

# CCD widths
ax.annotate('', xy=(12, -3.8), xytext=(0, -3.8),
            arrowprops=dict(arrowstyle='<->', color='#88ffbb', lw=0.9))
ax.text(6, -4.5, '12', color='#88ffbb', ha='center', fontsize=7.5)
ax.annotate('', xy=(24, -3.8), xytext=(12, -3.8),
            arrowprops=dict(arrowstyle='<->', color='#88ffbb', lw=0.9))
ax.text(18, -4.5, '12', color='#88ffbb', ha='center', fontsize=7.5)

# Coverage annotation
ax.text(12, 35.8,
        'Total logic area: 24×16 + 2×(12×16) = 384 + 384 = 768 mm²\n'
        '= 24×32 mm  →  φ = 1.0  (full tiling, no gap!)',
        color='#ffff99', fontsize=8, ha='center', fontweight='bold')

# Memory label
ax.text(27.5, 32.5, 'Memory die\n(base)\n24×32 mm\nCTE = 3.5\nE = 110 GPa\n20 W',
        color='#88aaff', fontsize=7.5, ha='left', va='top')

# Legend
legend_handles = [
    patches.Patch(facecolor='#4a1a00', edgecolor='#ff8844', label='XCD — 24×16 mm compute'),
    patches.Patch(facecolor='#00301a', edgecolor='#44cc88', label='CCD × 2 — 12×16 mm each'),
    patches.Patch(facecolor='#0d2040', edgecolor='#4477cc', label='Memory die — 24×32 mm (base)'),
    Line2D([0],[0], color='#666', lw=1.2, ls='--', label='Reticle 26×33 mm'),
]
ax.legend(handles=legend_handles, fontsize=7, facecolor='#1e2130',
          edgecolor='#444', labelcolor='white', loc='upper right',
          bbox_to_anchor=(1.0, 0.98))

# ─────────────────────────────────────────────────────────────────────────
# PANEL 2 — CROSS-SECTION (X-slice at Y=24, through XCD; and Y=8, through CCDs)
# ─────────────────────────────────────────────────────────────────────────
ax = AX_XSEC
ax.set_xlim(-2, 27)
ax.set_ylim(-3.2, 13)
ax.set_title('Cross-Section (slice at Y=8 through CCDs)  +  Warpage Measurement',
             color='white', fontsize=11, fontweight='bold')
ax.set_xlabel('X  (mm)', color='#aaa', fontsize=8)
ax.set_ylabel('Z — thickness  (µm → mm, scale ×12)', color='#aaa', fontsize=8)
ax.grid(color='#1e2030', lw=0.4, ls=':')

SC = 12 / 1000    # µm → display-mm  (×12 for visibility)
T_MEM_d  = 40  * SC          # 0.48 mm
T_TIM_d  = max(1 * SC, 0.09) # boosted for visibility
T_LOG_d  = 40  * SC          # 0.48 mm
T_CAR_d  = min(719 * SC, 6.0) # capped for figure space

z0 = 0

# Memory die — full width 24 mm
rect_mem = patches.Rectangle((0, z0), 24, T_MEM_d,
                               fc='#0d2040', ec='#4477cc', lw=1.8)
ax.add_patch(rect_mem)
ax.text(-0.3, z0 + T_MEM_d/2,
        'Memory\n40 µm  CTE 3.5', color='#88aaff',
        fontsize=7, ha='right', va='center')
z0 += T_MEM_d

# TIM — full width (φ=1 now)
rect_tim = patches.Rectangle((0, z0), 24, T_TIM_d,
                               fc='#333333', ec='#aaaaaa', lw=0.9)
ax.add_patch(rect_tim)
ax.text(-0.3, z0 + T_TIM_d/2, 'TIM  1 µm', color='#aaa',
        fontsize=6.5, ha='right', va='center')
z0 += T_TIM_d

# Logic layer — CCD1 (x=0-12) and CCD2 (x=12-24) at this slice (y=8)
rect_ccd1 = patches.Rectangle((0,  z0), 12, T_LOG_d,
                                fc='#00301a', ec='#44cc88', lw=1.8)
rect_ccd2 = patches.Rectangle((12, z0), 12, T_LOG_d,
                                fc='#00301a', ec='#44cc88', lw=1.8)
ax.add_patch(rect_ccd1);  ax.add_patch(rect_ccd2)
ax.text(6,  z0 + T_LOG_d/2, 'CCD 1\n40 µm  CTE 2.6',
        color='#88ffbb', fontsize=7, ha='center', va='center')
ax.text(18, z0 + T_LOG_d/2, 'CCD 2\n40 µm  CTE 2.6',
        color='#88ffbb', fontsize=7, ha='center', va='center')
z0 += T_LOG_d

# Carrier — full width
z_car0 = z0
rect_car = patches.Rectangle((0, z_car0), 24, T_CAR_d,
                               fc='#1a2a4a', ec='#5588ff', lw=1.5, alpha=0.6)
ax.add_patch(rect_car)
ax.text(-0.3, z_car0 + T_CAR_d/2, 'Carrier\n719 µm',
        color='#88aaff', fontsize=7, ha='right', va='center')

z_top = z_car0 + T_CAR_d

# ── Warpage curves overlaid on bottom surface ─────────────────────────────
x_arr = np.linspace(0, 24, 300)
xc = 12.0

# Warpage magnitudes from analysis (µm), converted to display mm
d_Si  = 14.6 * SC    # ~0.175 mm display
d_SiC =  8.0 * SC

def bow(x, delta, sign):
    # parabola: δ at edges, 0 at centre → or vice versa
    # sign=+1: Si case, memory face concave (centre rises)
    # sign=-1: SiC case, reversed
    return sign * delta * (1 - ((x - xc)/xc)**2)

z_bow_Si  = bow(x_arr, d_Si,  +1)
z_bow_SiC = bow(x_arr, d_SiC, -1)

z_ref = -1.5    # draw bow curves below cross-section
ax.axhline(z_ref, color='white', lw=0.8, ls=':', alpha=0.5, label='Flat ref (no warp)')
ax.plot(x_arr, z_ref + z_bow_Si,  color='#5588ff', lw=2.2,
        label=f'Si carrier bow  (δ ≈ {14.6:.1f} µm, concave-up)')
ax.plot(x_arr, z_ref + z_bow_SiC, color='#ff8833', lw=2.2, ls='--',
        label=f'SiC carrier bow  (δ ≈ {8.0:.1f} µm, convex-up)')

# δ annotation for Si
peak_Si = z_ref + d_Si
ax.annotate('', xy=(xc, peak_Si), xytext=(xc, z_ref),
            arrowprops=dict(arrowstyle='<->', color='#88aaff', lw=1.3))
ax.text(xc + 0.8, (peak_Si + z_ref)/2, f'δ_Si\n{14.6} µm',
        color='#88aaff', fontsize=7, va='center')

# δ annotation for SiC
trough_SiC = z_ref + z_bow_SiC[0]   # at edge x=0
ax.annotate('', xy=(1, trough_SiC), xytext=(1, z_ref),
            arrowprops=dict(arrowstyle='<->', color='#ffaa66', lw=1.3))
ax.text(2, (trough_SiC + z_ref)/2, f'δ_SiC\n{8.0} µm',
        color='#ffaa66', fontsize=7, va='center')

# L_char span arrow
ax.annotate('', xy=(24, z_top+0.6), xytext=(0, z_top+0.6),
            arrowprops=dict(arrowstyle='<->', color='#aaaaaa', lw=1))
ax.text(12, z_top+0.9,
        'W = 24 mm   L_char = √(24²+32²) = 40 mm  →  δ = κ · L²/8',
        color='#aaaaaa', ha='center', fontsize=7.5)

ax.legend(fontsize=7.5, facecolor='#1e2130', edgecolor='#444',
          labelcolor='white', loc='upper right')

# ─────────────────────────────────────────────────────────────────────────
# PANEL 3 — WARPAGE EQUATIONS
# ─────────────────────────────────────────────────────────────────────────
ax = AX_EQ1
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.set_facecolor('#0c1018')
for sp in ax.spines.values(): sp.set_color('#334455')
ax.set_title('Warpage Model — ABD Plate Theory', color='white',
             fontsize=11, fontweight='bold')

EQ = '#e8f4ff';  HD = '#ffcc66';  AN = '#aaaaaa';  HL = '#aaddff'

def H(ax, y, t): ax.text(0.03, y, t, color=HD, fontsize=9.5,
                          fontweight='bold', va='top', transform=ax.transAxes)
def E(ax, y, t, fs=8.2): ax.text(0.03, y, t, color=EQ, fontsize=fs,
                                   va='top', transform=ax.transAxes,
                                   fontfamily='monospace')
def A(ax, y, t): ax.text(0.03, y, t, color=AN, fontsize=7.5,
                          va='top', transform=ax.transAxes, style='italic')

H(ax, 0.97, '① Layer definitions  (z = 0 at memory bottom)')
E(ax, 0.91,
  '  i   Layer     Ē_i = E_i/(1−ν_i)    α_i (ppm/°C)  t_i\n'
  '  0   Memory    110/(1−0.28)=152.8 GPa    3.5        t_mem\n'
  '  1   TIM       70/(1−0.17)=84.3 GPa      0.55       1 µm\n'
  '  2   Logic*    130/(1−0.28)=180.6 GPa    2.6        t_logic\n'
  '  3   Carrier   E_car/(1−ν_car)            α_car      t_carrier\n'
  '\n'
  '  * φ = 1.0 now (XCD+CCD tiles fully: 24×16+2×12×16=768mm²)',
  fs=7.5)

H(ax, 0.54, '② ABD stiffness matrix  (reference z = 0, not neutral axis)')
E(ax, 0.48,
  '  A = Σ Ē_i · t_i\n'
  '  B = Σ Ē_i · t_i · z_mid_i\n'
  '  D = Σ Ē_i · (t_i³/12  +  t_i · z_mid_i²)')

H(ax, 0.35, '③ Thermal resultants  (ΔT = 235 °C, reflow→room)')
E(ax, 0.29,
  '  N_T = ΔT · Σ Ē_i · α_i · t_i\n'
  '  M_T = ΔT · Σ Ē_i · α_i · t_i · z_mid_i')

H(ax, 0.20, '④ Curvature & peak warpage')
E(ax, 0.14,
  '  κ = (A·M_T − B·N_T) / (A·D − B²)      [1/m]\n'
  '  δ = κ · L_char² / 8                    L_char = 40 mm')
A(ax, 0.06,
  'Si carrier (CTE=2.6): κ > 0 → concave-up at memory  (sad face)\n'
  'SiC carrier (CTE=4.2): κ < 0 → convex-up at memory  (smile face)\n'
  '|δ_Si| ≈ 14.6 µm  vs  |δ_SiC| ≈ 8.0 µm  at baseline 40/40 µm')

# ─────────────────────────────────────────────────────────────────────────
# PANEL 4 — THERMAL + FREQUENCY EQUATIONS
# ─────────────────────────────────────────────────────────────────────────
ax = AX_EQ2
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.set_facecolor('#0c1018')
for sp in ax.spines.values(): sp.set_color('#334455')
ax.set_title('Thermal + Frequency  (updated: φ = 1.0)', color='white',
             fontsize=11, fontweight='bold')

H(ax, 0.97, '① Thermal resistance: logic junction → carrier top')
E(ax, 0.91,
  '  R_logic   = t_logic   / (k_Si  × A_full)\n'
  '  R_TIM     = t_TIM     / (k_TIM × A_full)\n'
  '\n'
  '  # φ=1 → logic tiles full area → NO spreading needed!\n'
  '  R_carrier = t_carrier / (k_carrier × A_full)\n'
  '\n'
  '  A_full = 24×32 = 768 mm²  (= A_logic = A_carrier)\n'
  '\n'
  '  θ_jc = R_logic + R_TIM + R_carrier')
A(ax, 0.61,
  'vs v1 (φ=1/3): spreading term ½ the resistance is now gone.\n'
  'Resistance drops because full area conducts heat.')

H(ax, 0.55, '② Junction temperature')
E(ax, 0.49,
  '  T_j = T_amb + P_logic × (θ_jc + θ_heatsink)\n'
  '\n'
  '  T_amb     = 25 °C\n'
  '  P_logic   = 180 W  (XCD ~150 W + CCD×2 ~30 W)\n'
  '  θ_heatsink= 0.30 K/W  (air-cooled)\n'
  '\n'
  '  Si  (φ=1): θ_jc ≈ 5.7 mK/W  → T_j ≈ 27.0°C above amb\n'
  '  SiC (φ=1): θ_jc ≈ 2.1 mK/W  → T_j ≈ 25.4°C above amb\n'
  '  ΔT_j ≈ 1.6 °C  (smaller than v1 because TIM now dominates)')

H(ax, 0.26, '③ Frequency uplift  (CV²f / thermal model)')
E(ax, 0.20,
  '  f(T_j) = f_ref × [1 + α_f × (T_j,baseline − T_j)]\n'
  '\n'
  '  f_ref = 3.5 GHz   α_f = 0.4%/°C\n'
  '\n'
  '  Δf ≈ 3500 MHz × 0.004 × 1.6 ≈ 22 MHz  (SiC vs Si)\n'
  '\n'
  '  CV²f link:  P = α·C·V²·f\n'
  '  Lower T_j → lower V_min → ↑f or ↓P at same frequency')
A(ax, 0.04,
  'Key: with full-tile φ=1, thermal differences between Si/SiC are\n'
  'smaller (TIM is now the bottleneck, not carrier spreading).')

# Global title
fig.suptitle(
    '3D Chiplet Stack  —  Updated Layout: XCD 24×16 mm  +  2×CCD 12×16 mm  '
    '(φ = 1.0, full tile)',
    color='white', fontsize=12, fontweight='bold', y=0.97
)

out = '/home/user/HyMu/3D-stack/topview_v2.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved → {out}')
plt.close()
