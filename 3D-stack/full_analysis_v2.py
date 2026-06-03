"""
3D Chiplet Stack: Si vs SiC Carrier — CORRECTED GEOMETRY
=========================================================
Die layout (from sketch):
  XCD  24×16 mm  (top half)      — compute, ~150 W
  CCD1 12×16 mm  (bottom-left)   — cache,    ~15 W
  CCD2 12×16 mm  (bottom-right)  — cache,    ~15 W
  Total logic area = 24×16 + 2×(12×16) = 768 mm² = full 24×32 mm tile
  φ = 1.0  →  no spreading resistance, 1-D conduction through full area

Carrier options:
  Si  : E=130 GPa, CTE=2.6 ppm/°C, k=150 W/mK
  SiC : E=450 GPa, CTE=4.2 ppm/°C, k=400 W/mK

Constraint: t_mem + t_logic + t_carrier + t_TIM = 800 µm
Sweep: t_mem 20→100 µm, t_logic 20→100 µm, step 5 µm
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Material properties ──────────────────────────────────────────────────
def biax(E, nu): return E / (1.0 - nu)

E_mem, nu_mem, alpha_mem = 110e9, 0.28, 3.5e-6
E_Si,  nu_Si,  alpha_Si  = 130e9, 0.28, 2.6e-6
E_TIM, nu_TIM, alpha_TIM = 70e9,  0.17, 0.55e-6
k_Si, k_TIM = 150.0, 1.0

CARRIERS = {
    'Si':  dict(label='Silicon Carrier',  E=130e9, nu=0.28,
                alpha=2.6e-6, k=150.0,   color='#5588ff', ls='-'),
    'SiC': dict(label='4H-SiC Carrier',  E=450e9, nu=0.21,
                alpha=4.2e-6, k=400.0,   color='#ff7733', ls='--'),
}

# ── Geometry ─────────────────────────────────────────────────────────────
W_mem, L_mem = 24e-3, 32e-3
A_full       = W_mem * L_mem          # 768 mm²  — logic tiles completely
phi          = 1.0                    # full coverage (XCD+CCD×2 = 768 mm²)
L_char       = np.sqrt(W_mem**2 + L_mem**2)   # 40 mm diagonal

# ── Fixed parameters ─────────────────────────────────────────────────────
t_TIM    = 1e-6
T_TOTAL  = 800e-6
T_avail  = T_TOTAL - t_TIM

dT_asm   = 235.0       # °C  reflow→room
T_amb    = 25.0
theta_hs = 0.30        # K/W  air-cooled
P_logic  = 180.0       # W
f_ref    = 3.5         # GHz  baseline
alpha_f  = 0.004       # /°C  (0.4 %/°C)

# ── Warpage: ABD plate matrix (φ=1 → no area-fraction correction) ─────────
def warpage_um(t_mem, t_logic, t_carrier, cp, dT=dT_asm):
    Ebar_car = biax(cp['E'], cp['nu'])
    # z0, thickness, biaxial modulus, CTE
    # φ=1.0 → TIM and logic use full Ebar (no area fraction reduction)
    z0  = [0, t_mem, t_mem+t_TIM, t_mem+t_TIM+t_logic]
    thk = [t_mem, t_TIM, t_logic, t_carrier]
    Eb  = [biax(E_mem,nu_mem), biax(E_TIM,nu_TIM), biax(E_Si,nu_Si), Ebar_car]
    alp = [alpha_mem, alpha_TIM, alpha_Si, cp['alpha']]
    zm  = [z0[i] + thk[i]/2 for i in range(4)]

    A = sum(Eb[i]*thk[i]                                   for i in range(4))
    B = sum(Eb[i]*thk[i]*zm[i]                             for i in range(4))
    D = sum(Eb[i]*(thk[i]**3/12 + thk[i]*zm[i]**2)        for i in range(4))
    N_T = dT * sum(Eb[i]*alp[i]*thk[i]        for i in range(4))
    M_T = dT * sum(Eb[i]*alp[i]*thk[i]*zm[i]  for i in range(4))

    det   = A*D - B**2
    kappa = (A*M_T - B*N_T) / det
    return kappa * L_char**2 / 8.0 * 1e6   # µm, signed

# ── Thermal: pure 1-D (φ=1 → no spreading) ────────────────────────────
def theta_jc(t_logic, t_carrier, k_car):
    R_logic = t_logic   / (k_Si  * A_full)
    R_tim   = t_TIM     / (k_TIM * A_full)
    R_car   = t_carrier / (k_car  * A_full)
    return R_logic + R_tim + R_car    # K/W

def Tj(th):   return T_amb + P_logic * (th + theta_hs)
def freq(Tj_val, Tj_base): return f_ref * (1 + alpha_f*(Tj_base - Tj_val))

# ── Reference baseline (Si, 40/40 µm) ─────────────────────────────────
tc_base  = T_avail - 80e-6
Tj_base  = Tj(theta_jc(40e-6, tc_base, CARRIERS['Si']['k']))

# ── Sweep ──────────────────────────────────────────────────────────────
step   = 5e-6
t_vals = np.arange(20e-6, 101e-6, step)

res = {k: dict(TM=[], TL=[], TC=[], W=[], TH=[], TJ=[], F=[])
       for k in CARRIERS}

for tm in t_vals:
    for tl in t_vals:
        tc = T_avail - tm - tl
        if tc < 10e-6: continue
        for key, cp in CARRIERS.items():
            r  = res[key]
            w  = warpage_um(tm, tl, tc, cp)
            th = theta_jc(tl, tc, cp['k'])
            Tj_val = Tj(th)
            r['TM'].append(tm*1e6); r['TL'].append(tl*1e6)
            r['TC'].append(tc*1e6); r['W'].append(w)
            r['TH'].append(th*1e3); r['TJ'].append(Tj_val)
            r['F'].append(freq(Tj_val, Tj_base))

for k in CARRIERS:
    for f in res[k]: res[k][f] = np.array(res[k][f])

# ── 2-D grids ─────────────────────────────────────────────────────────
t_arr = t_vals*1e6

def to_grid(xv, yv, zv):
    G = np.full((len(t_arr), len(t_arr)), np.nan)
    for n in range(len(xv)):
        i = int(round((xv[n]-20)/5))
        j = int(round((yv[n]-20)/5))
        if 0<=i<len(t_arr) and 0<=j<len(t_arr): G[j,i] = zv[n]
    return G

grids = {k: {q: to_grid(res[k]['TM'], res[k]['TL'], res[k][q])
             for q in ('W','TH','F')} for k in CARRIERS}

X, Y = np.meshgrid(t_arr, t_arr)
ABS_W = {k: np.abs(grids[k]['W']) for k in CARRIERS}
WARP_DELTA  = ABS_W['Si'] - ABS_W['SiC']
FREQ_UPLIFT = (grids['SiC']['F'] - grids['Si']['F']) * 1000  # MHz

# ── Pareto (minimise |warpage| and θ_jc) ─────────────────────────────
def pareto_front(key):
    w = np.abs(res[key]['W']); t = res[key]['TH']
    dom = np.zeros(len(w), bool)
    for i in range(len(w)):
        if dom[i]: continue
        mask = (w<=w[i])&(t<=t[i])&((w<w[i])|(t<t[i]))
        if mask.any(): dom[i]=True
    idx = np.where(~dom)[0]; order = np.argsort(w[idx])
    return w[idx[order]], t[idx[order]], res[key]['TC'][idx[order]]

par = {k: pareto_front(k) for k in CARRIERS}

# ── Candidate table ────────────────────────────────────────────────────
print(f"\n{'Carrier':6} {'t_mem':>6} {'t_log':>6} {'t_car':>6}"
      f" {'|warp|µm':>10} {'θ mK/W':>8} {'Tj °C':>7} {'f GHz':>7}")
print("─"*60)
for key, cp in CARRIERS.items():
    for tm_um in [30, 40, 60, 80]:
        for tl_um in [30, 40, 60, 80]:
            tc_um = T_avail*1e6 - tm_um - tl_um
            if tc_um < 10: continue
            tm,tl,tc = tm_um*1e-6, tl_um*1e-6, tc_um*1e-6
            w  = warpage_um(tm,tl,tc,cp)
            th = theta_jc(tl,tc,cp['k'])
            Tj_v = Tj(th); fv = freq(Tj_v, Tj_base)
            print(f"{key:6} {tm_um:>6} {tl_um:>6} {tc_um:>6.0f}"
                  f" {abs(w):>10.3f} {th*1e3:>8.3f} {Tj_v:>7.2f} {fv:>7.4f}")
    print()

# ── Find global best trade-off point ─────────────────────────────────
for key in CARRIERS:
    r = res[key]
    # Normalise both objectives 0-1 then pick minimum sum
    w_norm = (np.abs(r['W']) - np.abs(r['W']).min()) / (np.abs(r['W']).max() - np.abs(r['W']).min())
    t_norm = (r['TH'] - r['TH'].min()) / (r['TH'].max() - r['TH'].min())
    best = np.argmin(w_norm + t_norm)
    print(f"[{CARRIERS[key]['label']}] Optimal: "
          f"M={r['TM'][best]:.0f}/L={r['TL'][best]:.0f}/C={r['TC'][best]:.0f} µm  "
          f"|warp|={abs(r['W'][best]):.2f} µm  θ={r['TH'][best]:.3f} mK/W  "
          f"Tj={r['TJ'][best]:.2f}°C  f={r['F'][best]:.4f} GHz")

# ── Figure: 3×3 grid ──────────────────────────────────────────────────
fig = plt.figure(figsize=(21, 15))
fig.patch.set_facecolor('#0e1117')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.30)

TKW  = dict(color='white', fontsize=9.5, fontweight='bold', pad=7)
LKW  = dict(color='#cccccc', fontsize=8)
TICK = dict(colors='#777777', labelsize=7.5)
CBKW = dict(color='#cccccc', fontsize=7.5)
GKW  = dict(color='#1e2030', lw=0.5, ls='--')

def sax(ax):
    ax.set_facecolor('#141720')
    ax.tick_params(axis='both', **TICK)
    ax.grid(**GKW)
    ax.set_xlabel('Memory die thickness  (µm)', **LKW)
    ax.set_ylabel('Logic die thickness  (µm)',  **LKW)

def cbar(fig, ax, cf, label):
    cb = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(label, **CBKW); cb.ax.tick_params(**TICK)

def star(ax, tm=40, tl=40, lbl='Baseline\n40/40 µm'):
    ax.plot(tm, tl, 'w*', ms=14, zorder=6)
    ax.text(tm+1, tl-3, lbl, color='white', fontsize=6.5)

# shared colour limits
vw = (min(np.nanmin(ABS_W['Si']), np.nanmin(ABS_W['SiC'])),
      max(np.nanmax(ABS_W['Si']), np.nanmax(ABS_W['SiC'])))
vt = (min(np.nanmin(grids['Si']['TH']), np.nanmin(grids['SiC']['TH'])),
      max(np.nanmax(grids['Si']['TH']), np.nanmax(grids['SiC']['TH'])))
vf = (min(np.nanmin(grids['Si']['F']), np.nanmin(grids['SiC']['F']))*1000,
      max(np.nanmax(grids['Si']['F']), np.nanmax(grids['SiC']['F']))*1000)

# Row 0 — |Warpage|
for col, key in enumerate(['Si','SiC']):
    ax = fig.add_subplot(gs[0, col])
    sax(ax)
    cf = ax.contourf(X, Y, ABS_W[key], levels=25, cmap='plasma',
                     vmin=vw[0], vmax=vw[1])
    ax.contour(X, Y, ABS_W[key], levels=8, colors='white', lw=0.5, alpha=0.3)
    ax.set_title(f'{CARRIERS[key]["label"]} — |Assembly Warpage|  (µm)', **TKW)
    cbar(fig, ax, cf, '|Warpage|  (µm)'); star(ax)

ax02 = fig.add_subplot(gs[0, 2])
sax(ax02)
vd = np.nanmax(np.abs(WARP_DELTA))
cf02 = ax02.contourf(X, Y, WARP_DELTA, levels=25, cmap='RdYlGn',
                     vmin=-vd, vmax=vd)
ax02.contour(X, Y, WARP_DELTA, levels=[0], colors='white', lw=1.5, ls='--')
ax02.set_title('Warpage Reduction  |Si| − |SiC|  (µm)\nGreen = SiC wins', **TKW)
cbar(fig, ax02, cf02, 'Reduction (µm)')
sax(ax02); ax02.plot(40, 40, 'w*', ms=14, zorder=6)

# Row 1 — θ_jc and frequency uplift
for col, key in enumerate(['Si','SiC']):
    ax = fig.add_subplot(gs[1, col])
    sax(ax)
    cf = ax.contourf(X, Y, grids[key]['TH'], levels=25, cmap='inferno',
                     vmin=vt[0], vmax=vt[1])
    ax.contour(X, Y, grids[key]['TH'], levels=8, colors='white', lw=0.5, alpha=0.3)
    ax.set_title(f'{CARRIERS[key]["label"]} — θ_jc  (mK/W)', **TKW)
    cbar(fig, ax, cf, 'θ_jc  (mK/W)'); star(ax)

ax12 = fig.add_subplot(gs[1, 2])
sax(ax12)
cf12 = ax12.contourf(X, Y, FREQ_UPLIFT, levels=25, cmap='cool')
ax12.contour(X, Y, FREQ_UPLIFT, levels=6, colors='white', lw=0.5, alpha=0.3)
ax12.set_title(f'Frequency Uplift  SiC − Si  (MHz)\n'
               f'α_f={alpha_f*100:.1f}%/°C · f_ref={f_ref} GHz', **TKW)
cbar(fig, ax12, cf12, 'Freq uplift  (MHz)'); sax(ax12)
ax12.plot(40, 40, 'w*', ms=14, zorder=6)

# Row 2 — Pareto + frequency maps
ax20 = fig.add_subplot(gs[2, 0])
sax(ax20); ax20.set_xlabel('|Warpage|  (µm)', **LKW)
ax20.set_ylabel('θ_jc  (mK/W)', **LKW)
for key, cp in CARRIERS.items():
    r = res[key]
    ax20.scatter(np.abs(r['W']), r['TH'], color=cp['color'],
                 s=5, alpha=0.15, linewidths=0)
    pw, pt, _ = par[key]
    ax20.plot(pw, pt, color=cp['color'], lw=2.5, label=cp['label'])
    ax20.scatter(pw, pt, color=cp['color'], s=45,
                 edgecolors='white', lw=0.7, zorder=5)

# Mark baseline and optimal points
w0 = abs(warpage_um(40e-6,40e-6,tc_base, CARRIERS['Si']))
t0 = theta_jc(40e-6,tc_base, CARRIERS['Si']['k'])*1e3
ax20.scatter([w0],[t0], marker='*', s=300, color='white', zorder=7,
             label='Si baseline 40/40 µm')

# Mark best trade-off for each carrier
for key, cp in CARRIERS.items():
    r = res[key]
    w_n = (np.abs(r['W'])-np.abs(r['W']).min())/(np.abs(r['W']).max()-np.abs(r['W']).min())
    t_n = (r['TH']-r['TH'].min())/(r['TH'].max()-r['TH'].min())
    b = np.argmin(w_n+t_n)
    ax20.scatter([np.abs(r['W'][b])],[r['TH'][b]], marker='D', s=120,
                 color=cp['color'], edgecolors='white', lw=1.2, zorder=8,
                 label=f'{key} optimal: M{r["TM"][b]:.0f}/L{r["TL"][b]:.0f}')

ax20.set_title('Pareto Front — |Warpage| vs θ_jc\n◆ = recommended design point', **TKW)
ax20.legend(fontsize=7, facecolor='#1e2130', edgecolor='#555', labelcolor='white')

for col, key in enumerate(['Si','SiC'], start=1):
    ax = fig.add_subplot(gs[2, col])
    sax(ax)
    cf = ax.contourf(X, Y, grids[key]['F']*1000, levels=25,
                     cmap='plasma', vmin=vf[0], vmax=vf[1])
    ax.contour(X, Y, grids[key]['F']*1000, levels=8,
               colors='white', lw=0.5, alpha=0.3)
    ax.set_title(f'{CARRIERS[key]["label"]} — Achievable Freq (MHz)', **TKW)
    cbar(fig, ax, cf, 'Frequency  (MHz)'); star(ax)

    # Mark recommended point
    r = res[key]
    w_n = (np.abs(r['W'])-np.abs(r['W']).min())/(np.abs(r['W']).max()-np.abs(r['W']).min())
    t_n = (r['TH']-r['TH'].min())/(r['TH'].max()-r['TH'].min())
    b   = np.argmin(w_n+t_n)
    ax.plot(r['TM'][b], r['TL'][b], 'D', color=CARRIERS[key]['color'],
            ms=10, mec='white', mew=1.2, zorder=7,
            label=f'◆ Rec: M{r["TM"][b]:.0f}/L{r["TL"][b]:.0f}/C{r["TC"][b]:.0f}')
    ax.legend(fontsize=7, facecolor='#1e2130', edgecolor='#555', labelcolor='white')

fig.suptitle(
    '3D Chiplet Stack — Si vs 4H-SiC Carrier  '
    '[XCD 24×16 + CCD×2 12×16 mm,  φ=1.0 full tile,  Total=800 µm]\n'
    'Memory CTE=3.5 ppm/°C  |  Logic CTE=2.6 ppm/°C  '
    '|  P_logic=180 W  |  θ_hs=0.30 K/W (air)  |  f_ref=3.5 GHz',
    color='white', fontsize=9.5, fontweight='bold', y=1.002
)

out = '/home/user/HyMu/3D-stack/full_analysis_v2.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'\nSaved → {out}')
plt.close()

# ── Summary ────────────────────────────────────────────────────────────
print(f'\nBaseline (Si 40/40 µm, tc={tc_base*1e6:.0f} µm):')
print(f'  Tj = {Tj_base:.2f} °C   f = {f_ref:.4f} GHz')
print()
for key, cp in CARRIERS.items():
    th = theta_jc(40e-6, tc_base, cp['k'])
    Tj_v = Tj(th)
    print(f"{cp['label']:20s}  θ_jc={th*1e3:.3f} mK/W  "
          f"Tj={Tj_v:.2f}°C  f={freq(Tj_v,Tj_base):.4f} GHz")
