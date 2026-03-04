"""
Q1: Disposable Income Model
- Data cleaning
- Correlation heatmap
- Multiple linear regression
- BLS adjustment for disposable income
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats

#Load and clean data
df = pd.read_csv("/Users/carterlawrence/Downloads/cps_00002.csv")

#Filter: working-age adults 18-70
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 70)]

#Filter: positive wage income (working people)
df = df[df['INCWAGE'] > 0]

#Remove IPUMS missing/NA codes
df = df[df['INCWAGE'] < 9999999]

#Compute after-tax income
df['AFTER_TAX'] = df['INCWAGE'] - df['FEDTAX'] - df['STATETAX'] - df['FICA']

#Remove cases where after-tax is negative or zero
df = df[df['AFTER_TAX'] > 0]

#Remove top and bottom 1% outliers of AFTER_TAX
low = df['AFTER_TAX'].quantile(0.01)
high = df['AFTER_TAX'].quantile(0.99)
df = df[(df['AFTER_TAX'] >= low) & (df['AFTER_TAX'] <= high)]

#Quick summary
print(f"\nAfter-tax income summary:")
print(f"  Mean:   ${df['AFTER_TAX'].mean():,.0f}")
print(f"  Median: ${df['AFTER_TAX'].median():,.0f}")
print(f"  Std:    ${df['AFTER_TAX'].std():,.0f}")
print(f"  Min:    ${df['AFTER_TAX'].min():,.0f}")
print(f"  Max:    ${df['AFTER_TAX'].max():,.0f}")

#Variable for analysis

#SEX: 1 = Male, 2 = Female
df['SEX_NUM'] = df['SEX'].map({1: 0, 2: 1})# 0 = Male, 1 = Female

#RACE: simplify to major categories
#100 = White, 200 = Black, 300 = American Indian, 650+ = Asian, 651+ = multiracial etc.
def simplify_race(r):
    if r == 100: return 0  # White
    elif r == 200: return 1  # Black
    elif r >= 650 and r < 700: return 2  # Asian
    else: return 3  # Other
df['RACE_NUM'] = df['RACE'].apply(simplify_race)

#EDUC: recode to approximate years of education
#Key codes: 10=no school, 30=some elem, 60=9th grade, 73=HS diploma,
#81=some college, 91=Bachelors, 111=Masters, 123=Doctorate, 124=Professional
educ_map = {
    2: 0, 10: 0, 20: 4, 30: 6, 40: 8, 50: 9, 60: 10, 71: 11,
    73: 12, 81: 13, 91: 14, 92: 16, 111: 18, 123: 20, 124: 20,
    125: 20
}
df['EDUC_YEARS'] = df['EDUC'].map(educ_map)
# Fill any unmapped values with median
df['EDUC_YEARS'] = df['EDUC_YEARS'].fillna(df['EDUC_YEARS'].median())

#STATEFIP -> convert to cost-of-living proxy using state median income ranking
#Group states into tiers (simplified)
high_cost_states = [6, 36, 34, 25, 9, 15, 11, 53]  # CA, NY, NJ, MA, CT, HI, DC, WA
low_cost_states = [28, 5, 54, 40, 22, 47, 21, 1, 45, 29]  # MS, AR, WV, OK, LA, TN, KY, AL, SC, MO

def col_tier(fip):
    if fip in high_cost_states: return 2  #High cost
    elif fip in low_cost_states: return 0  #Low cost
    else: return 1  #Average

df['COL_TIER'] = df['STATEFIP'].apply(col_tier)

print("Variable encoding complete:")
print(f"  SEX_NUM:    {df['SEX_NUM'].value_counts().to_dict()}")
print(f"  RACE_NUM:   {df['RACE_NUM'].value_counts().to_dict()}")
print(f"  EDUC_YEARS: mean={df['EDUC_YEARS'].mean():.1f}, min={df['EDUC_YEARS'].min()}, max={df['EDUC_YEARS'].max()}")
print(f"  COL_TIER:   {df['COL_TIER'].value_counts().to_dict()}")
print(f"  FAMSIZE:    mean={df['FAMSIZE'].mean():.1f}, range={df['FAMSIZE'].min()}-{df['FAMSIZE'].max()}")

#Heatmap

#Select columns for heatmap
heatmap_cols = ['AGE', 'SEX_NUM', 'RACE_NUM', 'EDUC_YEARS', 'FAMSIZE', 'COL_TIER', 'AFTER_TAX']
heatmap_labels = ['Age', 'Sex\n(0=M,1=F)', 'Race', 'Education\n(Years)', 'Family\nSize', 'Cost of\nLiving', 'After-Tax\nIncome']

corr_matrix = df[heatmap_cols].corr()

#Print correlation with AFTER_TAX
for col, label in zip(heatmap_cols[:-1], heatmap_labels[:-1]):
    r = corr_matrix.loc[col, 'AFTER_TAX']
    print(f"  {label.replace(chr(10), ' '):25s}: {r:+.4f}")

#Plot heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(corr_matrix, dtype=bool)

sns.heatmap(corr_matrix, 
            annot=True, fmt='.3f', 
            cmap='RdBu_r', center=0,
            vmin=-1, vmax=1,
            xticklabels=heatmap_labels,
            yticklabels=heatmap_labels,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
            ax=ax)

ax.set_title('Q1: Correlation Heatmap of Demographic Variables and After-Tax Income', 
             fontsize=13, pad=15)
plt.tight_layout()
plt.savefig('/Users/carterlawrence/downloads/q1_heatmap.png', dpi=150)
plt.close()

#Multiple Lin Reg

#Add engineered features
df['AGE_SQ'] = df['AGE'] ** 2
df['AGE_x_EDUC'] = df['AGE'] * df['EDUC_YEARS']

#Features: salary + demographics + engineered terms
feature_cols = ['INCWAGE', 'AGE', 'AGE_SQ', 'SEX_NUM', 'EDUC_YEARS', 'COL_TIER', 'AGE_x_EDUC']
feature_labels = ['Wage Income', 'Age', 'Age²', 'Sex (0=M,1=F)', 'Education (Years)', 'Cost of Living Tier', 'Age × Education']

X = df[feature_cols].values
y = df['AFTER_TAX'].values

#Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)} rows")
print(f"Test set:     {len(X_test)} rows")

#Fit regression
model = LinearRegression()
model.fit(X_train, y_train)

#Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#Metrics
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nModel Performance:")
print(f"  R² (train): {r2_train:.4f}")
print(f"  R² (test):  {r2_test:.4f}")
print(f"  MAE (test): ${mae_test:,.0f}")
print(f"  RMSE (test):${rmse_test:,.0f}")

#Coefficients and p-values
#Compute p-values manually using t-statistics
n = len(X_train)
p = X_train.shape[1]
y_pred_all = model.predict(X_train)
residuals = y_train - y_pred_all
MSE = np.sum(residuals**2) / (n - p - 1)
#Variance-covariance matrix of coefficients
X_with_const = np.column_stack([np.ones(n), X_train])
try:
    var_covar = MSE * np.linalg.inv(X_with_const.T @ X_with_const)
    se = np.sqrt(np.diag(var_covar))
    # t-statistics
    coefs_with_intercept = np.concatenate([[model.intercept_], model.coef_])
    t_stats = coefs_with_intercept / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-p-1))
except:
    p_values = [None] * (p + 1)

print(f"\nRegression Coefficients:")
print(f"  {'Variable':25s} {'Coefficient':>12s} {'p-value':>12s}")
print(f"  {'-'*25} {'-'*12} {'-'*12}")
print(f"  {'Intercept':25s} {model.intercept_:>12,.2f} {p_values[0]:>12.6f}" if p_values[0] is not None else f"  {'Intercept':25s} {model.intercept_:>12,.2f}")

for i, (col, label) in enumerate(zip(feature_cols, feature_labels)):
    coef = model.coef_[i]
    pv = p_values[i+1] if p_values[i+1] is not None else 'N/A'
    sig = '***' if isinstance(pv, float) and pv < 0.001 else '**' if isinstance(pv, float) and pv < 0.01 else '*' if isinstance(pv, float) and pv < 0.05 else ''
    if isinstance(pv, float):
        print(f"  {label:25s} {coef:>12,.2f} {pv:>12.6f} {sig}")
    else:
        print(f"  {label:25s} {coef:>12,.2f} {pv:>12s}")

print(f"\n  Significance: *** p<0.001, ** p<0.01, * p<0.05")

#Plots

#Plot 1: Actual vs Predicted
fig, ax = plt.subplots(figsize=(8, 8))
sample_idx = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
ax.scatter(y_test[sample_idx], y_pred_test[sample_idx], alpha=0.15, s=10, color='#3498db')
max_val = max(y_test.max(), y_pred_test.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual After-Tax Income ($)')
ax.set_ylabel('Predicted After-Tax Income ($)')
ax.set_title(f'Q1: Actual vs. Predicted After-Tax Income (R² = {r2_test:.4f})')
ax.legend()
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.0f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.0f}'))
plt.tight_layout()
plt.savefig('/Users/carterlawrence/downloads/q1_actual_vs_predicted', dpi=150)
plt.close()

#Plot 2: Coefficient bar chart
fig, ax = plt.subplots(figsize=(10, 6))
# Normalize coefficients by multiplying by std of each feature to show relative importance
feature_stds = df[feature_cols].std().values
standardized_coefs = model.coef_ * feature_stds
colors = ['#e74c3c' if c < 0 else '#27ae60' for c in standardized_coefs]
bars = ax.barh(feature_labels, standardized_coefs, color=colors, alpha=0.85, edgecolor='white')
ax.set_xlabel('Standardized Coefficient (Effect on After-Tax Income)')
ax.set_title('Q1: Relative Importance of Demographic Variables')
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/carterlawrence/downloads/q1_coefficients.png', dpi=150)
plt.close()

#Plot 3: Residual distribution
fig, ax = plt.subplots(figsize=(10, 6))
residuals_test = y_test - y_pred_test
ax.hist(residuals_test, bins=60, color='#3498db', alpha=0.7, edgecolor='white', density=True)
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Residual (Actual - Predicted) ($)')
ax.set_ylabel('Density')
ax.set_title('Q1: Distribution of Prediction Residuals')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:,.0f}'))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/carterlawrence/downloads/q1_residuals.png', dpi=150)
plt.close()

#Predictions w BLS Adjustments

#BLS Consumer Expenditure Survey: annual essential costs by age group (single person, avg COL)
#Categories: housing, food, healthcare, transportation, utilities
BLS_ESSENTIALS = {
    '18-24': 24600,
    '25-34': 29100,
    '35-44': 33000,
    '45-54': 32400,
    '55-64': 31500,
    '65-70': 29700,
}
#Household size multiplier for essentials
HH_MULT = {1: 1.0, 2: 1.55, 3: 1.95, 4: 2.25, 5: 2.50}

def get_bls_essentials(age, famsize):
    if age < 25: base = BLS_ESSENTIALS['18-24']
    elif age < 35: base = BLS_ESSENTIALS['25-34']
    elif age < 45: base = BLS_ESSENTIALS['35-44']
    elif age < 55: base = BLS_ESSENTIALS['45-54']
    elif age < 65: base = BLS_ESSENTIALS['55-64']
    else: base = BLS_ESSENTIALS['65-70']
    mult = HH_MULT.get(min(famsize, 5), 2.5)
    return base * mult

#Demo profiles: [INCWAGE, AGE, AGE², SEX, EDUC_YRS, COL_TIER, AGE×EDUC]
demos = [
    ("Male, 22, HS, $30K, avg",          [30000, 22, 22**2, 0, 12, 1, 22*12]),
    ("Female, 28, BA, $55K, high",        [55000, 28, 28**2, 1, 16, 2, 28*16]),
    ("Male, 30, BA, $75K, avg",           [75000, 30, 30**2, 0, 16, 1, 30*16]),
    ("Male, 35, MA, $95K, avg",           [95000, 35, 35**2, 0, 18, 1, 35*18]),
    ("Female, 25, Some col, $40K, low",   [40000, 25, 25**2, 1, 14, 0, 25*14]),
    ("Male, 40, HS, $50K, low",           [50000, 40, 40**2, 0, 12, 0, 40*12]),
    ("Male, 45, PhD, $150K, high",        [150000, 45, 45**2, 0, 20, 2, 45*20]),
    ("Female, 30, BA, $65K, avg",         [65000, 30, 30**2, 1, 16, 1, 30*16]),
]

for name, features in demos:
    pred_after_tax = model.predict([features])[0]
    age = features[1]
    famsize = 1  # default single for demos
    essentials = get_bls_essentials(age, famsize)
    disposable = max(0, pred_after_tax - essentials)
    print(f"{name:45s} ${pred_after_tax:>10,.0f} ${essentials:>10,.0f} ${disposable:>10,.0f}")

