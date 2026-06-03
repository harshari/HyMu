"""
3D Chiplet Stack: Si vs SiC Carrier — Thermal–Warpage–Frequency Analysis
=========================================================================
Fixes vs v1:
  • Warpage: full ABD plate matrix (κ = (A·M_T − B·N_T)/(A·D − B²));
    v1 was double-correcting about neutral axis.
  • Thermal: adds lateral spreading resistance in carrier (dominant for SiC comparison).
  • Frequency uplift model: f(Tj) = f_ref × [1 + α_f × (Tj_baseline − Tj)]

Stack (bottom → top):
  Memory die  24×32 mm | CTE=3.5 ppm/°C | E=110 GPa | 20 W
  Oxide TIM   1 µm     | k=1 W/mK
  Logic dies  4×(8×8)mm| CTE=2.6 ppm/°C | E=130 GPa | 180 W
  Carrier     24×32 mm | Si or 4H-SiC
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Material properties ──────────────────────────────────────────────────
def biax(E, nu):
    return E / (1.0 - nu)

# Memory die (DRAM-type)
E_mem, nu_mem = 110e9, 0.28
alpha_mem     = 3.5e-6          # /°C
k_mem         = 100.0           # W/mK

# Logic / bare silicon
E_Si, nu_Si   = 130e9, 0.28
alpha_Si      = 2.6e-6
k_Si          = 150.0

# Oxide TIM (SiO₂-like)
E_TIM, nu_TIM = 70e9, 0.17
alpha_TIM     = 0.55e-6
k_TIM         = 1.0

CARRIERS = {
    'Si': dict(label='Silicon Carrier',
               E=130e9, nu=0.28, alpha=2.6e-6, k=150.0,
               color='#5588ff', ls='-'),
    'SiC': dict(label='4H-SiC Carrier',
                E=450e9, nu=0.21, alpha=4.2e-6, k=400.0,
                color='#ff7733', ls='--'),
}

# ── Geometry ─────────────────────────────────────────────────────────────
W_mem, L_mem = 24e-3, 32e-3
A_mem        = W_mem * L_mem            # 768 mm²

n_logic      = 4
A_logic      = n_logic * (8e-3)**2     # 256 mm²
phi          = A_logic / A_mem          # ≈ 0.333  (coverage fraction)

L_char       = np.sqrt(W_mem**2 + L_mem**2)   # 40 mm diagonal

# Spreading characteristic length: single die half-width
t_spr        = 8e-3 / 4                # 2 mm

# ── Fixed operating parameters ────────────────────────────────────────────
t_TIM        = 1e-6
T_TOTAL      = 800e-6
T_avail      = T_TOTAL - t_TIM         # 799 µm for dies

dT_asm       = 235.0        # °C  reflow → room
T_amb        = 25.0         # °C
theta_hs     = 0.30         # K/W  air-cooled heatsink
P_logic      = 180.0        # W
f_ref        = 3.5          # GHz  baseline at nominal Tj
alpha_f      = 0.004        # /°C  frequency thermal sensitivity (0.4 %/°C)

# ── Warpage: full ABD matrix ──────────────────────────────────────────────
def warpage_um(t_mem, t_logic, t_carrier, cprops, dT=dT_asm):
    """
    Signed warpage via the plate ABD matrix.
    Layers 0-3: memory | TIM (×φ) | logic (×φ) | carrier
    z measured from bottom of memory die.
    Returns signed warpage in µm (positive = convex up at memory side).
    """
    Ebar_car = biax(cprops['E'], cprops['nu'])

    z0  = [0,
           t_mem,
           t_mem + t_TIM,
           t_mem + t_TIM + t_logic]
    thk = [t_mem, t_TIM, t_logic, t_carrier]
    Eb  = [biax(E_mem, nu_mem),
           biax(E_TIM, nu_TIM) * phi,
           biax(E_Si,  nu_Si)  * phi,
           Ebar_car]
    alp = [alpha_mem, alpha_TIM, alpha_Si, cprops['alpha']]

    zm = [z0[i] + thk[i] / 2.0 for i in range(4)]

    # ABD stiffnesses (all about z = 0)
    A_m = sum(Eb[i] * thk[i]                                 for i in range(4))
    B_m = sum(Eb[i] * thk[i] * zm[i]                        for i in range(4))
    D_m = sum(Eb[i] * (thk[i]**3 / 12.0 + thk[i] * zm[i]**2) for i in range(4))

    # Thermal resultants (about z = 0)
    N_T = dT * sum(Eb[i] * alp[i] * thk[i]          for i in range(4))
    M_T = dT * sum(Eb[i] * alp[i] * thk[i] * zm[i]  for i in range(4))

    det   = A_m * D_m - B_m**2
    kappa = (A_m * M_T - B_m * N_T) / det
    return kappa * L_char**2 / 8.0 * 1e6   # µm (signed)

# ── Thermal: logic junction → carrier top (with spreading) ───────────────
def theta_jc(t_logic, t_carrier, k_car):
    """
    R_logic + R_TIM + R_carrier(with partial spreading).
    Effective area grows as heat spreads in carrier up to A_mem.
    """
    R_logic = t_logic / (k_Si  * A_logic)
    R_tim   = t_TIM   / (k_TIM * A_logic)

    spr     = 1.0 - np.exp(-t_carrier / t_spr)
    A_eff   = A_logic + (A_mem - A_logic) * spr
    R_car   = t_carrier / (k_car * A_eff)
    return R_logic + R_tim + R_car          # K/W

def Tj(th):
    return T_amb + P_logic * (th + theta_hs)

# ── Reference junction temperature (Si carrier, 40/40 µm baseline) ───────
tc_base     = T_avail - 80e-6              # 719 µm
Tj_baseline = Tj(theta_jc(40e-6, tc_base, CARRIERS['Si']['k']))

def freq_GHz(Tj_val):
    return f_ref * (1.0 + alpha_f * (Tj_baseline - Tj_val))

# ── Parameter sweep ───────────────────────────────────────────────────────
step   = 5e-6
t_vals = np.arange(20e-6, 101e-6, step)   # 20→100 µm

res = {k: dict(TM=[], TL=[], TC=[], W=[], TH=[], TJ=[], F=[])
       for k in CARRIERS}

for tm in t_vals:
    for tl in t_vals:
        tc = T_avail - tm - tl
        if tc < 10e-6:
            continue
        for key, cp in CARRIERS.items():
            r = res[key]
            w  = warpage_um(tm, tl, tc, cp)
            th = theta_jc(tl, tc, cp['k'])
            Tj_val = Tj(th)
            r['TM'].append(tm * 1e6);  r['TL'].append(tl * 1e6)
            r['TC'].append(tc * 1e6);  r['W'].append(w)
            r['TH'].append(th * 1e3);  r['TJ'].append(Tj_val)
            r['F'].append(freq_GHz(Tj_val))

for k in CARRIERS:
    for f in res[k]:
        res[k][f] = np.array(res[k][f])

# ── 2-D grids ─────────────────────────────────────────────────────────────
t_arr = t_vals * 1e6

def to_grid(xv, yv, zv):
    G = np.full((len(t_arr), len(t_arr)), np.nan)
    for n in range(len(xv)):
        i = int(round((xv[n] - 20) / 5))
        j = int(round((yv[n] - 20) / 5))
        if 0 <= i < len(t_arr) and 0 <= j < len(t_arr):
            G[j, i] = zv[n]
    return G

grids = {}
for k in CARRIERS:
    r = res[k]
    grids[k] = {
        'W':  to_grid(r['TM'], r['TL'], r['W']),
        'TH': to_grid(r['TM'], r['TL'], r['TH']),
        'F':  to_grid(r['TM'], r['TL'], r['F']),
    }

X, Y = np.meshgrid(t_arr, t_arr)

# Signed warpage magnitude reduction (|Si| − |SiC|): positive → SiC better
WARP_DELTA = np.abs(grids['Si']['W']) - np.abs(grids['SiC']['W'])
FREQ_UPLIFT = (grids['SiC']['F'] - grids['Si']['F']) * 1000   # MHz

# ── Pareto: minimise |warpage| AND θ_jc ──────────────────────────────────
def pareto(warp_arr, theta_arr):
    w, t = np.abs(warp_arr), theta_arr
    n = len(w)
    dom = np.zeros(n, bool)
    for i in range(n):
        if dom[i]: continue
        mask = (w <= w[i]) & (t <= t[i]) & ((w < w[i]) | (t < t[i]))
        if mask.any():
            dom[i] = True
    idx = np.where(~dom)[0]
    order = np.argsort(w[idx])
    return w[idx[order]], t[idx[order]], res['Si']['TC'][idx[order]]  # TC same shape

par = {}
for k in CARRIERS:
    r = res[k]
    pw, pt, pc = pareto(r['W'], r['TH'])
    par[k] = (pw, pt, pc)

# ── Discrete candidate table ──────────────────────────────────────────────
print(f"\n{'Carrier':6} {'t_mem':>7} {'t_log':>7} {'t_car':>7} "
      f"{'|warp|µm':>10} {'θ mK/W':>9} {'Tj °C':>8} {'f GHz':>7}")
print("─" * 70)
for key, cp in CARRIERS.items():
    for tm_um in [30, 40, 60, 80]:
        for tl_um in [30, 40, 60, 80]:
            tc_um = T_avail * 1e6 - tm_um - tl_um
            if tc_um < 10:
                continue
            tm, tl, tc = tm_um*1e-6, tl_um*1e-6, tc_um*1e-6
            w  = warpage_um(tm, tl, tc, cp)
            th = theta_jc(tl, tc, cp['k'])
            Tj_v = Tj(th)
            fv = freq_GHz(Tj_v)
            print(f"{key:6} {tm_um:>7} {tl_um:>7} {tc_um:>7.0f} "
                  f"{abs(w):>10.3f} {th*1e3:>9.2f} {Tj_v:>8.2f} {fv:>7.4f}")
    print()

# ── Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('#0e1117')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.32)

TKW  = dict(color='white', fontsize=10, fontweight='bold', pad=8)
LKW  = dict(color='#cccccc', fontsize=8.5)
TICK = dict(colors='#888888', labelsize=8)
CBKW = dict(color='#cccccc', fontsize=8)
GKW  = dict(color='#252535', lw=0.5, ls='--')

def sax(ax, xl='Memory die (µm)', yl='Logic die (µm)'):
    ax.set_facecolor('#141720')
    ax.tick_params(axis='both', **TICK)
    ax.grid(**GKW)
    ax.set_xlabel(xl, **LKW)
    ax.set_ylabel(yl, **LKW)

def cbar(fig, ax, cf, label):
    cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(label, **CBKW)
    cb.ax.tick_params(**TICK)

def star(ax, note='Baseline 40/40 µm'):
    ax.plot(40, 40, 'w*', ms=13, zorder=6)
    ax.text(41, 37, note, color='white', fontsize=6.5, va='top')

# Row 0 — warpage
vw_min = min(np.nanmin(np.abs(grids['Si']['W'])), np.nanmin(np.abs(grids['SiC']['W'])))
vw_max = max(np.nanmax(np.abs(grids['Si']['W'])), np.nanmax(np.abs(grids['SiC']['W'])))

for col, key in enumerate(['Si', 'SiC']):
    ax = fig.add_subplot(gs[0, col])
    sax(ax)
    cf = ax.contourf(X, Y, np.abs(grids[key]['W']), levels=25,
                     cmap='plasma', vmin=vw_min, vmax=vw_max)
    ax.contour(X, Y, np.abs(grids[key]['W']), levels=7,
               colors='white', linewidths=0.5, alpha=0.3)
    ax.set_title(f'{CARRIERS[key]["label"]} — |Assembly Warpage|  (µm)', **TKW)
    cbar(fig, ax, cf, '|Warpage|  (µm)')
    star(ax)

ax02 = fig.add_subplot(gs[0, 2])
sax(ax02)
vd = np.nanmax(np.abs(WARP_DELTA))
cf02 = ax02.contourf(X, Y, WARP_DELTA, levels=25, cmap='RdYlGn',
                     vmin=-vd, vmax=vd)
ax02.contour(X, Y, WARP_DELTA, levels=[0.0], colors='white',
             linewidths=1.5, linestyles='--')
ax02.set_title('Warpage Reduction  Si − SiC  (µm)\n'
               'Green = SiC has less warpage', **TKW)
cbar(fig, ax02, cf02, 'Reduction  (µm)')
sax(ax02)
ax02.plot(40, 40, 'w*', ms=13, zorder=6)

# Row 1 — θ_jc and frequency uplift
vt_min = min(np.nanmin(grids['Si']['TH']), np.nanmin(grids['SiC']['TH']))
vt_max = max(np.nanmax(grids['Si']['TH']), np.nanmax(grids['SiC']['TH']))

for col, key in enumerate(['Si', 'SiC']):
    ax = fig.add_subplot(gs[1, col])
    sax(ax)
    cf = ax.contourf(X, Y, grids[key]['TH'], levels=25,
                     cmap='inferno', vmin=vt_min, vmax=vt_max)
    ax.contour(X, Y, grids[key]['TH'], levels=7,
               colors='white', linewidths=0.5, alpha=0.3)
    ax.set_title(f'{CARRIERS[key]["label"]} — θ_jc  (mK/W)', **TKW)
    cbar(fig, ax, cf, 'θ_jc  (mK/W)')
    star(ax)

ax12 = fig.add_subplot(gs[1, 2])
sax(ax12)
cf12 = ax12.contourf(X, Y, FREQ_UPLIFT, levels=25, cmap='cool')
ax12.contour(X, Y, FREQ_UPLIFT, levels=6,
             colors='white', linewidths=0.5, alpha=0.3)
ax12.set_title(f'Frequency Uplift  SiC − Si  (MHz)\n'
               f'Model: Δf = {alpha_f*100:.1f}%/°C × ΔTj  |  '
               f'f_ref = {f_ref} GHz', **TKW)
cbar(fig, ax12, cf12, 'Freq uplift  (MHz)')
sax(ax12)
ax12.plot(40, 40, 'w*', ms=13, zorder=6)

# Row 2 — Pareto + absolute frequency maps
ax20 = fig.add_subplot(gs[2, 0])
sax(ax20, xl='|Warpage|  (µm)', yl='θ_jc  (mK/W)')
for key, cp in CARRIERS.items():
    r = res[key]
    ax20.scatter(np.abs(r['W']), r['TH'],
                 color=cp['color'], s=5, alpha=0.15, linewidths=0)
    pw, pt, _ = par[key]
    ax20.plot(pw, pt, color=cp['color'], lw=2.5,
              label=cp['label'], zorder=4)
    ax20.scatter(pw, pt, color=cp['color'], s=45,
                 edgecolors='white', lw=0.7, zorder=5)

# mark baseline on pareto
w0 = abs(warpage_um(40e-6, 40e-6, tc_base, CARRIERS['Si']))
t0 = theta_jc(40e-6, tc_base, CARRIERS['Si']['k']) * 1e3
ax20.scatter([w0], [t0], marker='*', s=280, color='white',
             zorder=7, label=f'Si baseline 40/40 µm')
ax20.set_title('Pareto Front: |Warpage| vs θ_jc\n(minimise both)', **TKW)
ax20.legend(fontsize=7.5, facecolor='#1e2130',
            edgecolor='#555', labelcolor='white')

for col, key in enumerate(['Si', 'SiC'], start=1):
    ax = fig.add_subplot(gs[2, col])
    sax(ax)
    vf_lo = np.nanmin(grids['Si']['F']) * 1000
    vf_hi = np.nanmax(grids['SiC']['F']) * 1000
    cf = ax.contourf(X, Y, grids[key]['F'] * 1000, levels=25,
                     cmap='plasma', vmin=vf_lo, vmax=vf_hi)
    ax.contour(X, Y, grids[key]['F'] * 1000, levels=7,
               colors='white', linewidths=0.5, alpha=0.3)
    ax.set_title(f'{CARRIERS[key]["label"]} — Achievable Freq (MHz)', **TKW)
    cbar(fig, ax, cf, 'Frequency  (MHz)')
    star(ax)

fig.suptitle(
    '3D Chiplet Stack — Si vs 4H-SiC Carrier: Warpage · Thermal · Frequency\n'
    'Memory 24×32 mm (CTE=3.5, E=110 GPa)  |  Logic 4×(8×8) mm (CTE=2.6)  '
    '|  TIM 1 µm oxide  |  Total=800 µm  |  P_logic=180 W  '
    f'|  θ_hs={theta_hs} K/W (air)  |  f_ref={f_ref} GHz @ baseline Tj',
    color='white', fontsize=9, fontweight='bold', y=1.002
)

out = '/home/user/HyMu/3D-stack/sic_vs_si_tradeoff.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'\nSaved → {out}')
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────
print(f'\nBaseline Tj (Si, 40/40 µm, tc={tc_base*1e6:.0f} µm): {Tj_baseline:.2f} °C')
for key, cp in CARRIERS.items():
    th = theta_jc(40e-6, tc_base, cp['k'])
    Tj_v = Tj(th)
    print(f"{cp['label']:20s}  θ_jc={th*1e3:.2f} mK/W  "
          f"Tj={Tj_v:.2f} °C  f={freq_GHz(Tj_v):.4f} GHz")
