import pandas as pd

df = pd.read_csv("behavioural_dataset.csv")

print(df.info())

print(df.head())
print("\nMissing Values:\n", df.isnull().sum())

df.rename(columns={"Marrital Status": "Marital Status"}, inplace=True)

df["Personal loan"] = df["Personal loan"].map({"Yes": 1, "No": 0})

print("\nDuplicate Records:", df.duplicated().sum())

print("\nStatistical Summary:\n", df.describe())

# Save the cleaned dataset
df.to_csv("cleaned_behavioural_dataset.csv", index=False)
import matplotlib.pyplot as plt
import seaborn as sns

# Convert Salary & Price to Lakhs for better readability
df["Total Salary (Lakhs)"] = df["Total Salary"] / 100000
df["Price (Lakhs)"] = df["Price"] / 100000

# Set plot style
sns.set_style("whitegrid")

### Age Distribution ###
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=10, kde=True, color="blue")
plt.title("Age Distribution of Automobile Buyers", fontsize=14)
plt.xlabel("Age (Years) → Represents the buyer's age", fontsize=12)
plt.ylabel("Number of Buyers → Count of people in that age group", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.show()

### Total Salary vs Price Scatterplot ###
plt.figure(figsize=(8, 5))
scatter = sns.scatterplot(
    x=df["Total Salary (Lakhs)"], 
    y=df["Price (Lakhs)"], 
    hue=df["Personal loan"], 
    palette={1: "green", 0: "red"}
)
plt.title("Total Salary vs Vehicle Price", fontsize=14)
plt.xlabel("Total Salary (Lakhs) → How much the buyer earns", fontsize=12)
plt.ylabel("Vehicle Price (Lakhs) → How much they spent on a vehicle", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title="Personal Loan", labels=["No Loan (Red)", "Has Loan (Green)"])

# Annotate explanation
plt.text(30, 40, "Top-right: High salary, expensive car", fontsize=10, color="black")
plt.text(5, 5, "Bottom-left: Low salary, affordable car", fontsize=10, color="black")
plt.grid(True)
plt.show()

### Correlation Heatmap ###
plt.figure(figsize=(8, 5))
heatmap = sns.heatmap(
    df[["Age", "No of Dependents", "Total Salary (Lakhs)", "Price (Lakhs)", "Personal loan"]].corr(), 
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Correlation Heatmap", fontsize=14)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)

# Explanation for correlation values
plt.figtext(0.5, -0.1, 
    "🔹 Correlation values (0 to 1):\n"
    "   - 1.0 = Strong positive relationship (as X increases, Y increases)\n"
    "   - 0.0 = No relationship\n"
    "   - -1.0 = Strong negative relationship (as X increases, Y decreases)",
    wrap=True, horizontalalignment='center', fontsize=10
)
plt.show()
