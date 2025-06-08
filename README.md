**Titanic Dataset EDA and Visualization**
**Project Objective:**
To clean, explore, and visualize the Titanic dataset to derive meaningful insights and patterns about the passengers and their likelihood of survival.
**Loading the Dataset**
import pandas as pd                
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset from seaborn
titanic = sns.load_dataset('titanic')
**Data Cleaning**
# Check missing values
print("Missing Values:\n", titanic.isnull().sum())

# Fill missing age with median
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Drop 'deck' due to too many missing values
titanic.drop(columns='deck', inplace=True)

# Drop rows where 'embarked' is missing
titanic.dropna(subset=['embarked'], inplace=True)
**REMOVING  Duplicates**
titanic.drop_duplicates(inplace=True)
**Outlier Detection in Age**
# Detect outliers using IQR
Q1 = titanic['age'].quantile(0.25)
Q3 = titanic['age'].quantile(0.75)
IQR = Q3 - Q1

outliers = titanic[(titanic['age'] < Q1 - 1.5 * IQR) | (titanic['age'] > Q3 + 1.5 * IQR)]
print(f"Number of outliers in age: {len(outliers)}")
**Visualizations**
Survival Count by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(data=titanic, x='pclass', hue='survived')
plt.title('Passenger Class vs Survival')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()
**Survival Count by Sex**
plt.figure(figsize=(6, 4))
sns.countplot(data=titanic, x='sex', hue='survived')
plt.title('Sex vs Survival')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived')
plt.show()
**Age Distribution**
plt.figure(figsize=(6, 4))
titanic['age'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
**Fare Distribution**
plt.figure(figsize=(6, 4))
titanic['fare'].hist(bins=30, color='lightgreen', edgecolor='black')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
**Correlation Heatmap**
plt.figure(figsize=(8, 6))
corr = titanic.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
**Summary of Key Insights**
print("\nSummary of Key Insights:")
print("- Age and Fare have significant variability; some outliers are present.")
print("- Females had a higher survival rate than males.")
print("- 1st class passengers were more likely to survive.")
print("- Missing values in 'age' were imputed with the median.")
print("- 'deck' column was dropped due to high missing data.")
print("- Correlation shows 'fare' and 'pclass' relate to survival.")

**How to Run This Script**
Install the required Python libraries
pip install pandas numpy seaborn matplotlib
**Run the script**
Save the script as titanic_eda.py or open the notebook in Jupyter and run:
python titanic_eda.py












