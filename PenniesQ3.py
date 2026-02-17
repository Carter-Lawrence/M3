import random
import matplotlib.pyplot as plt

N = 100000  # number of simulated transactions

rounding_rules = {
    "round_up": lambda x, step: ((int(x / step) + 1) * step),
    "round_down": lambda x, step: (int(x / step) * step),
    "round_nearest": lambda x, step: round(x / step) * step
}

coin_steps = [0.05, 0.10, 0.25]  # possible coin denominations

# simulate for one coin type
step = 0.05

results = {rule: [] for rule in rounding_rules}

for _ in range(N):
    subtotal = random.uniform(1, 100)          # random transaction $1-$100
    tax_rate = 0.07
    total = subtotal * (1 + tax_rate)          # add tax

    for rule_name, func in rounding_rules.items():
        rounded_total = func(total, step)
        diff = rounded_total - total          # profit/loss per transaction
        results[rule_name].append(diff)

# compute averages
for rule_name, diffs in results.items():
    avg_diff = sum(diffs) / N
    print(f"{rule_name} avg rounding effect per transaction: ${avg_diff:.4f}")

# histogram of rounding differences
plt.hist(results['round_nearest'], bins=50)
plt.xlabel("Rounding difference ($)")
plt.ylabel("Frequency")
plt.title("Distribution of rounding differences (nearest 5c)")
plt.show()
