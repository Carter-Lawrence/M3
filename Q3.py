import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = '/Users/carterlawrence/downloads'
US_ADULTS = 260_000_000
BETA = 0.07     
GAMMA = 0.018   
DELTA = 0.108   
US_M = US_ADULTS / 1e6
S0, I0, R0 = 0.74 * US_M, 0.22 * US_M, 0.04 * US_M

#Historical Suicide Data
suicide_years = np.array([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,
                          2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,
                          2020,2021,2022,2023])
suicide_rates = np.array([10.4,10.7,10.9,10.8,10.9,10.9,11.0,11.3,11.6,11.8,
                          12.1,12.3,12.6,12.6,13.0,13.3,13.5,14.0,14.2,13.9,
                          13.5,14.0,14.2,14.1])

PROBLEM_RATE = 0.17           
SUICIDE_EXCESS_PER_100K = 50  

#SIR Model
dt = 1/52
S, I, R = S0, I0, R0
S_h, I_h, R_h, t_h = [S], [I], [R], [0]

for w in range(1, 75 * 52 + 1):
    total = S + I + R
    s, i, r = S/total, I/total, R/total
    new_inf = BETA * s * i * total * dt
    recover = GAMMA * I * dt
    relapse = DELTA * R * dt
    S = max(0, S - new_inf)
    I = max(0, I + new_inf - recover + relapse)
    R = max(0, R + recover - relapse)
    S_h.append(S); I_h.append(I); R_h.append(R); t_h.append(w/52)

S_h, I_h, R_h, t_h = np.array(S_h), np.array(I_h), np.array(R_h), np.array(t_h)
sir_years = t_h + 2025

#Fit Pre-Gambling Suicide Trend
pre_mask = suicide_years <= 2017
suicide_coeffs = np.polyfit(suicide_years[pre_mask], suicide_rates[pre_mask], 1)
suicide_trend = np.poly1d(suicide_coeffs)

#Build projections
I_sir_frac = {}
for yr in range(2018, 2025):
    I_sir_frac[yr] = 0.15 + (0.22 - 0.15) * (yr - 2018) / 6
for i, yr_offset in enumerate(t_h):
    yr = int(2025 + yr_offset)
    if yr not in I_sir_frac:
        I_sir_frac[yr] = I_h[i] / US_M

projection_years = np.arange(2000, 2076)
suicide_baseline = np.array([suicide_trend(yr) for yr in projection_years])
suicide_with_gambling = suicide_baseline.copy()

for i, yr in enumerate(projection_years):
    if yr >= 2018 and yr in I_sir_frac:
        problem_frac = I_sir_frac[yr] * PROBLEM_RATE
        suicide_with_gambling[i] = suicide_baseline[i] + problem_frac * SUICIDE_EXCESS_PER_100K


#SIR Scatter
step = 8
idx_s = np.arange(0, len(t_h), step)

fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(sir_years[idx_s], S_h[idx_s], color='#3498db', s=20, marker='D', alpha=0.8, label='S — Susceptible (Non-gamblers)')
ax.scatter(sir_years[idx_s], I_h[idx_s], color='#e74c3c', s=20, marker='D', alpha=0.8, label='I — Active Gamblers')
ax.scatter(sir_years[idx_s], R_h[idx_s], color='#27ae60', s=20, marker='D', alpha=0.8, label='R — Recovered (Quit)')
ax.axvline(2050, color='gray', linestyle='--', alpha=0.3)
ax.text(2050.5, US_M + 5, '2050', fontsize=9, color='gray')
ax.set_xlabel('Year', fontsize=13); ax.set_ylabel('US Adults (Millions)', fontsize=13)
ax.set_title('Q3: SIR Gambling Contagion — Population Dynamics (2025–2100)', fontsize=14)
ax.legend(fontsize=11, loc='center right'); ax.grid(alpha=0.2)
ax.set_xlim(2025, 2100); ax.set_ylim(0, US_M + 10)
plt.tight_layout(); plt.savefig(f'{SAVE_DIR}/q3_sir_classic.png', dpi=150); plt.close()

#FIGURE 2: Gambling participation%
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(sir_years[idx_s], I_h[idx_s]/US_M*100, color='#e74c3c', s=20, marker='D', alpha=0.8)
ax.axvline(2050, color='gray', linestyle='--', alpha=0.3)
for yr in [2035, 2050, 2075, 2100]:
    i = int((yr - 2025) * 52)
    pct = I_h[i]/US_M*100
    ax.annotate(f'{pct:.1f}%', xy=(yr, pct), xytext=(yr+1.5, pct+3),
                fontsize=10, fontweight='bold', color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1))
ax.set_xlabel('Year', fontsize=13); ax.set_ylabel('% of US Adults Actively Gambling', fontsize=13)
ax.set_title('Q3: Projected Online Sports Gambling Participation (SIR Model)', fontsize=14)
ax.grid(alpha=0.3); ax.set_xlim(2025, 2100); ax.set_ylim(0, 100)
plt.tight_layout(); plt.savefig(f'{SAVE_DIR}/q3_sir_participation.png', dpi=150); plt.close()

#Suicide — Pre-gambling trend vs SIR-weighted
fig, ax = plt.subplots(figsize=(14, 7))
ax.scatter(suicide_years, suicide_rates, color='black', s=40, zorder=5, label='Historical Data (CDC)')
ax.plot(projection_years, suicide_baseline, color='#3498db', linewidth=2.5, linestyle='--',
        label='Pre-Gambling Trend (2000-2017 fit, projected)')
post_2017 = projection_years >= 2018
ax.plot(projection_years[post_2017], suicide_with_gambling[post_2017],
        color='#e74c3c', linewidth=2.5, label='With Gambling (SIR Model + Co-occurrence)')
ax.fill_between(projection_years[post_2017], suicide_baseline[post_2017],
                suicide_with_gambling[post_2017], alpha=0.2, color='#e74c3c',
                label='Gambling-Associated Excess')
ax.axvline(2018, color='gray', linestyle=':', alpha=0.5)
ax.text(2018.5, 9, 'PASPA\nOverturned\n(2018)', fontsize=9, color='gray')
idx_2050 = np.where(projection_years == 2050)[0][0]
gap_2050 = suicide_with_gambling[idx_2050] - suicide_baseline[idx_2050]
excess_deaths = gap_2050 / 100000 * US_ADULTS
ax.annotate(f'Gap: +{gap_2050:.1f} per 100k\n(+{excess_deaths:,.0f} deaths/yr)',
            xy=(2050, suicide_with_gambling[idx_2050]),
            xytext=(2053, suicide_with_gambling[idx_2050] + 1.5),
            fontsize=10, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))
ax.set_xlabel('Year', fontsize=13); ax.set_ylabel('Suicide Rate (per 100,000, age-adjusted)', fontsize=13)
ax.set_title('Q3: US Suicide Rate — Pre-Gambling Trend vs SIR-Weighted Projection', fontsize=14)
ax.legend(fontsize=10, loc='upper left'); ax.grid(alpha=0.3)
ax.set_xlim(2000, 2060); ax.set_ylim(8, 28)
ax.text(0.02, 0.02,
        'Historical: CDC NCHS | Trend: Linear fit to 2000-2017 (pre-PASPA)\n'
        'Co-occurrence: PMC — 19% of PG consider suicide vs 4.1% general pop\n'
        'SIR: β=0.07 (NCPG), γ=0.018 (Hodgins et al.), δ=0.108 (Grall-Bronnec et al.)',
        transform=ax.transAxes, fontsize=7, color='gray', va='bottom')
plt.tight_layout(); plt.savefig(f'{SAVE_DIR}/q3_suicide_trend.png', dpi=150); plt.close()