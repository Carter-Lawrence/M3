import random
import matplotlib.pyplot as plt

# 1. setup parameters
N = 1728         # transactions over 3 years
results = []

for x in range(1000000):
    print(x)
    result = 0

    for i in range(N):
        numPennies = random.randint(0,4)
        percentSaved = random.random()

        if percentSaved < 0.55:
            result += numPennies

    results.append(result)

# ---- print mean ----
mean_value = sum(results) / len(results)
print("Mean pennies saved:", mean_value)

# ---- histogram ----
plt.hist(results, bins=50)   # adjust bins if you want
plt.xlabel("total pennies saved")
plt.ylabel("frequency")
plt.title("Monte Carlo distribution of pennies saved")
plt.show()
