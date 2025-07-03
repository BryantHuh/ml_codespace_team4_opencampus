import pandas as pd
import statsmodels.formula.api as smf
# Load the dataset
url = "https://raw.githubusercontent.com/opencampus-sh/einfuehrung-in-data-science-und-ml/main/house_pricing_data/house_pricing_train.csv"
house_pricing = pd.read_csv(url)
# Fit the linear model
mod = smf.ols('price ~ sqft_lot15 + C(condition)', data=house_pricing).fit()
# Print the summary
print(mod.summary())
