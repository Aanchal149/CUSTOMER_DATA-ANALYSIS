#500120160
#Anchal Rastogi
#B5
#Fundamental Of DS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel("customer_data.xlsx")

# Convert Purchase_Date to datetime format
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])

# Total number of transactions
total_transactions = len(df)

# Probability of purchasing each Product Category
product_probabilities = df['Product_Category'].value_counts() / total_transactions

# Expected Purchase Amount (E[X])
expected_value = df['Purchase_Amount'].mean()

# Spending Probability Distribution
bins = [0, 50, 100, 200, float('inf')]
labels = ["$0-50", "$50-100", "$100-200", "$200+"]
df['Spending_Category'] = pd.cut(df['Purchase_Amount'], bins=bins, labels=labels, right=False)
spending_distribution = df['Spending_Category'].value_counts(normalize=True)

# Joint Probability (Product & Payment Method)
joint_counts = pd.crosstab(df['Product_Category'], df['Payment_Method'])
joint_probability = joint_counts / total_transactions

# Conditional Probability P(Payment Method | Product Category)
conditional_probability = joint_counts.div(joint_counts.sum(axis=1), axis=0)

# Set plot style
sns.set(style="whitegrid")

# 1. Monthly Sales Trend (Bar Chart)
monthly_sales = df.groupby(df['Purchase_Date'].dt.to_period('M'))['Purchase_Amount'].sum()
plt.figure(figsize=(10, 5))
monthly_sales.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Purchase Amount")
plt.xticks(rotation=45)
plt.show()

# 2. Category-Wise Revenue (Pie Chart)
category_revenue = df.groupby("Product_Category")["Purchase_Amount"].sum()
plt.figure(figsize=(7, 7))
category_revenue.plot(kind="pie", autopct="%1.1f%%", colors=sns.color_palette("pastel"))
plt.title("Category-Wise Revenue Share")
plt.ylabel("")
plt.show()

# 3. Payment Method Trends (Stacked Bar Chart)
payment_trend = pd.crosstab(df['Purchase_Date'].dt.to_period('M'), df['Payment_Method'], normalize="index") * 100
payment_trend.plot(kind='bar', stacked=True, figsize=(10, 5), colormap="viridis", edgecolor="black")
plt.title("Payment Method Trends Over Months")
plt.xlabel("Month")
plt.ylabel("Percentage of Transactions")
plt.legend(title="Payment Method")
plt.xticks(rotation=45)
plt.show()

# 4. Customer Purchase Frequency (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(df["Customer_ID"].value_counts(), bins=10, kde=True, color="blue")
plt.title("Customer Purchase Frequency")
plt.xlabel("Number of Purchases")
plt.ylabel("Number of Customers")
plt.show()

# 5. Spending Pattern (Box Plot)
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Purchase_Amount"], color="lightcoral")
plt.title("Spending Pattern (Outlier Detection)")
plt.xlabel("Purchase Amount")
plt.show()

# 6. Joint Probability Heatmap (Product Category & Payment Method)
plt.figure(figsize=(8, 6))
sns.heatmap(joint_probability, annot=True, cmap="coolwarm", fmt=".2%")
plt.title("Joint Probability: Product Category & Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Product Category")
plt.show()

