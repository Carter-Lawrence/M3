import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = '/Users/carterlawrence/downloads'
np.random.seed(42)

#Participation %
PARTICIPATION = {
    ('M', '18-29'): 0.40, ('M', '30-49'): 0.45, ('M', '50+'): 0.18,
    ('F', '18-29'): 0.14, ('F', '30-49'): 0.12, ('F', '50+'): 0.05,
}

#Tier Probabilities
TIER_PROBS = {
    ('M', '18-29'): {'Low': 0.50, 'Medium': 0.35, 'High': 0.15},
    ('M', '30-49'): {'Low': 0.55, 'Medium': 0.35, 'High': 0.10},
    ('M', '50+'):   {'Low': 0.70, 'Medium': 0.25, 'High': 0.05},
    ('F', '18-29'): {'Low': 0.65, 'Medium': 0.28, 'High': 0.07},
    ('F', '30-49'): {'Low': 0.70, 'Medium': 0.25, 'High': 0.05},
    ('F', '50+'):   {'Low': 0.78, 'Medium': 0.19, 'High': 0.03},
}

#Tier Variables
TIERS = {
    'Low': {
        'pct_range': (0.01, 0.049),
        'bets_range': (1, 2),
        'house_edge': 0.0455,
        'label': 'Casual Gambler\n(1-4.9% of DI Wagered)',
        'color': '#27ae60',
    },
    'Medium': {
        'pct_range': (0.05, 0.199),
        'bets_range': (3, 8),
        'house_edge': 0.0455,
        'label': 'Moderate Gambler\n(5-19.9% of DI Wagered)',
        'color': '#f39c12',
    },
    'High': {
        'pct_range': (0.20, 1.00),
        'bets_range': (9, 20),
        'house_edge': 0.0455,
        'label': 'Heavy Gambler\n(20-100% of DI Wagered)',
        'color': '#e74c3c',
    },
}

def get_age_group(age):
    if age < 30: return '18-29'
    elif age < 50: return '30-49'
    else: return '50+'

#Monte Carlo Simulation
def simulate_person(disposable, age, gender, n_sims=100000):
    age_grp = get_age_group(age)
    key = (gender, age_grp)
    participation = PARTICIPATION.get(key, 0.15)
    tier_probs = TIER_PROBS.get(key, {'Low': 0.60, 'Medium': 0.30, 'High': 0.10})

    outcomes = np.zeros(n_sims)

    for i in range(n_sims):
        if np.random.random() > participation:
            outcomes[i] = 0
            continue

        #Assign tier
        r = np.random.random()
        cum = 0
        assigned_tier = 'Low'
        for tier_name, prob in tier_probs.items():
            cum += prob
            if r < cum:
                assigned_tier = tier_name
                break

        tier = TIERS[assigned_tier]

        #Sample from ranges
        pct_di = np.random.uniform(*tier['pct_range'])
        bets_per_week = np.random.uniform(*tier['bets_range'])

        annual_budget = disposable * pct_di
        bets_per_year = int(bets_per_week * 52)

        if annual_budget <= 0 or bets_per_year == 0:
            outcomes[i] = 0
            continue

        wager = annual_budget / bets_per_year
        p_win = (1 - tier['house_edge']) / 2

        wins = np.random.binomial(bets_per_year, p_win)
        losses = bets_per_year - wins
        outcomes[i] = (wins - losses) * wager

    #Per-tier results
    tier_results = {}
    for tier_name in ['Low', 'Medium', 'High']:
        tier = TIERS[tier_name]
        tier_outcomes = np.zeros(n_sims)
        for i in range(n_sims):
            pct_di = np.random.uniform(*tier['pct_range'])
            bets_per_week = np.random.uniform(*tier['bets_range'])
            budget = disposable * pct_di
            bets_yr = int(bets_per_week * 52)
            if budget <= 0 or bets_yr == 0:
                continue
            wager = budget / bets_yr
            p_win = (1 - tier['house_edge']) / 2
            wins = np.random.binomial(bets_yr, p_win)
            losses = bets_yr - wins
            tier_outcomes[i] = (wins - losses) * wager

        tier_results[tier_name] = {
            'budget_range': (disposable * tier['pct_range'][0], disposable * tier['pct_range'][1]),
            'exp_loss': np.mean(tier_outcomes),
            'median': np.median(tier_outcomes),
            'std': np.std(tier_outcomes),
            'p5': np.percentile(tier_outcomes, 5),
            'p95': np.percentile(tier_outcomes, 95),
            'prob_profit': np.mean(tier_outcomes > 0) * 100,
            'outcomes': tier_outcomes,
        }

    return {
        'participation': participation,
        'tier_probs': tier_probs,
        'weighted_exp_loss': np.mean(outcomes),
        'weighted_prob_any_loss': np.mean(outcomes < 0) * 100,
        'weighted_prob_profit': np.mean(outcomes > 0) * 100,
        'weighted_prob_zero': np.mean(outcomes == 0) * 100,
        'weighted_p5': np.percentile(outcomes, 5),
        'weighted_p95': np.percentile(outcomes, 95),
        'outcomes': outcomes,
        'tier_results': tier_results,
    }

#Demographic Profiles for Testing
profiles = [
    ("Mary White",   17376, 40, 'F'),
    ("Tim Simons",   37862, 70, 'M'),
    ("George Brown",  5975, 20, 'M'),
]

all_results = {}
for name, disp, age, gender in profiles:
    r = simulate_person(disp, age, gender)
    all_results[name] = r
    age_grp = get_age_group(age)
    print(f"\n{name} ({gender}, {age}, Disp: ${disp:,})")
    print(f"  Participation: {r['participation']*100:.0f}%")
    print(f"  Weighted E[P/L]: ${r['weighted_exp_loss']:,.0f}")
    for tier_name in ['Low', 'Medium', 'High']:
        tr = r['tier_results'][tier_name]
        pct = abs(tr['exp_loss']) / disp * 100
        print(f"  {tier_name:8s}: E[Loss]=${tr['exp_loss']:>8,.0f} ({pct:.2f}% of DI) | P(profit)={tr['prob_profit']:.1f}%")

#Weighted distributions for Mary, Tim, George
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, disp, age, gender) in zip(axes, profiles):
    r = all_results[name]
    nonzero = r['outcomes'][r['outcomes'] != 0]
    if len(nonzero) > 0:
        ax.hist(nonzero, bins=50, density=True, alpha=0.7, color='#3498db', edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(r['weighted_exp_loss'], color='darkred', linestyle='-', lw=2,
               label=f"E[P/L] = ${r['weighted_exp_loss']:,.0f}")
    age_grp = get_age_group(age)
    ax.set_title(f"{name}\n{gender}, {age} ({age_grp}), Disp: ${disp:,}", fontsize=11)
    ax.set_xlabel('Annual P/L ($)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.0f}'))
plt.suptitle('Q2: Demographic-Weighted Annual Gambling Outcomes (Among Gamblers Only)',
             fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/q2_weighted_distributions.png', dpi=150)
plt.close()

#Generalized % of DI distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
n_sims = 100000
disposable = 10000

for ax, tier_name in zip(axes, ['Low', 'Medium', 'High']):
    tier = TIERS[tier_name]
    outcomes_pct = np.zeros(n_sims)

    for i in range(n_sims):
        pct_di = np.random.uniform(*tier['pct_range'])
        bets_per_week = np.random.uniform(*tier['bets_range'])
        budget = disposable * pct_di
        bets_yr = int(bets_per_week * 52)
        if bets_yr == 0:
            continue
        wager = budget / bets_yr
        p_win = (1 - tier['house_edge']) / 2
        wins = np.random.binomial(bets_yr, p_win)
        losses = bets_yr - wins
        outcomes_pct[i] = (wins - losses) * wager / disposable * 100

    exp_pct = np.mean(outcomes_pct)
    n_bins = 25 if tier_name == 'Low' else 30
    ax.hist(outcomes_pct, bins=n_bins, density=True, alpha=0.7,
            color=tier['color'], edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(exp_pct, color='darkred', linestyle='-', lw=2,
               label=f"E[Loss] = {exp_pct:.2f}%")
    ax.set_title(tier['label'], fontsize=11)
    ax.set_xlabel('Annual P/L (% of Disposable Income)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

plt.suptitle('Q2: Distribution of Annual Gambling Outcomes as % of Disposable Income',
             fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/q2_distributions_pct.png', dpi=150)
plt.close()