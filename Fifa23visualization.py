import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the cleaned data
df = pd.read_csv("cleaned_fifa23_data.csv")

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 16))

# 1. Distribution of Player Overall Ratings
ax1 = fig.add_subplot(3, 3, 1)
df['Overall'].hist(bins=30, edgecolor='black', alpha=0.7)
ax1.set_title('Distribution of Player Overall Ratings', fontsize=14, fontweight='bold')
ax1.set_xlabel('Overall Rating')
ax1.set_ylabel('Number of Players')
ax1.axvline(df['Overall'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df["Overall"].mean():.1f}')
ax1.legend()
# 2. Top 10 Most Valuable Players
ax4 = fig.add_subplot(3, 3, 4)
top_valuable = df.nlargest(10, 'Value')[['Name', 'Value', 'Club']]
bars = ax4.barh(range(len(top_valuable)), top_valuable['Value'].values / 1e6, color='coral')
ax4.set_yticks(range(len(top_valuable)))
ax4.set_yticklabels(top_valuable['Name'].values, fontsize=8)
ax4.set_xlabel('Value (Millions €)')
ax4.set_title('Top 10 Most Valuable Players', fontsize=14, fontweight='bold')
ax4.invert_yaxis()

# Add club names as text
for i, (idx, row) in enumerate(top_valuable.iterrows()):
    ax4.text(row['Value']/1e6 + 0.5, i, row['Club'], va='center', fontsize=8)
#3. Top 10 Clubs by Average Overall Rating
ax8 = fig.add_subplot(3, 3, 8)
top_clubs = df.groupby('Club').agg({'Overall': 'mean', 'Name': 'count'}).nlargest(10, 'Overall')
bars = ax8.bar(range(len(top_clubs)), top_clubs['Overall'].values, color='teal')
ax8.set_xticks(range(len(top_clubs)))
ax8.set_xticklabels(top_clubs.index, rotation=45, ha='right', fontsize=8)
ax8.set_ylabel('Average Overall Rating')
ax8.set_title('Top 10 Clubs by Average Overall Rating', fontsize=14, fontweight='bold')
# Add player count as text
for i, (idx, row) in enumerate(top_clubs.iterrows()):
    ax8.text(i, row['Overall'] + 0.5, f'n={int(row["Name"])}', ha='center', fontsize=8)
#4. Correlation Heatmap
ax9 = fig.add_subplot(3, 3, 9)
numeric_cols = ['Age', 'Overall', 'Potential', 'Value', 'Wage', 'Attack Score', 'Efficiency', 'Value Efficiency']
correlation_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax9)
ax9.set_title('Correlation Heatmap of Player Attributes', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
# Additional Visualizations

# Figure 2: Value Efficiency Analysis
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top 10 Best Value for Money (highest Overall per Value)
ax = axes[0, 0]
best_value = df.nlargest(10, 'Value Efficiency')[['Name', 'Value Efficiency', 'Overall', 'Value']]
bars = ax.barh(range(len(best_value)), best_value['Value Efficiency'].values, color='green')
ax.set_yticks(range(len(best_value)))
ax.set_yticklabels(best_value['Name'].values, fontsize=8)
ax.set_xlabel('Overall Rating per Million €')
ax.set_title('Top 10 Best Value-for-Money Players', fontsize=12, fontweight='bold')
ax.invert_yaxis()

# Age vs Potential scatter
ax = axes[0, 1]
scatter = ax.scatter(df['Age'], df['Potential'], c=df['Overall'], cmap='plasma', alpha=0.6, s=50)
ax.set_xlabel('Age')
ax.set_ylabel('Potential Rating')
ax.set_title('Age vs Potential (colored by Overall)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='Overall Rating')

# Histogram of Value Efficiency
ax = axes[1, 0]
df['Value Efficiency'].hist(bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Overall Rating per Million €')
ax.set_ylabel('Number of Players')
ax.set_title('Distribution of Value Efficiency', fontsize=12, fontweight='bold')
ax.axvline(df['Value Efficiency'].median(), color='red', linestyle='--', label=f'Median: {df["Value Efficiency"].median():.2f}')
ax.legend()

# Attack Score by Player Level (Violin Plot)
ax = axes[1, 1]
player_levels = ['Average', 'Good', 'Elite']
attack_data = [df[df['Player Level'] == level]['Attack Score'].values for level in player_levels]
parts = ax.violinplot(attack_data, positions=[1, 2, 3], showmeans=True, showmedians=True)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(player_levels)
ax.set_ylabel('Attack Score')
ax.set_title('Attack Score Distribution by Player Level', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Figure 3: Top Players Radar Chart (Select top 3 players)
from math import pi

fig3, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))

# Select top 3 players by Overall rating
top_3_players = df.nlargest(3, 'Overall')[['Name', 'Overall', 'Potential', 'Attack Score', 'Efficiency']]

# Attributes to plot
attributes = ['Overall', 'Potential', 'Attack Score', 'Efficiency']
angles = [n / float(len(attributes)) * 2 * pi for n in range(len(attributes))]
angles += angles[:1]  # Close the loop

for idx, (_, player) in enumerate(top_3_players.iterrows()):
    ax = axes[idx]
    values = [player[attr] for attr in attributes]
    values += values[:1]  # Close the loop
    
    ax.plot(angles, values, 'o-', linewidth=2, label=player['Name'])
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)
    ax.set_ylim(0, 100)
    ax.set_title(f"{player['Name']}\nOverall: {player['Overall']}", fontsize=10, fontweight='bold')
    ax.grid(True)

plt.suptitle('Top 3 Players - Attribute Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("KEY INSIGHTS FROM THE DATA")
print("="*60)
print(f"\n📊 Total Players: {len(df)}")
print(f"📊 Average Age: {df['Age'].mean():.1f} years")
print(f"📊 Average Overall Rating: {df['Overall'].mean():.1f}")
print(f"📊 Average Attack Score: {df['Attack Score'].mean():.1f}")
print(f"\n💰 Most Expensive Player: {df.loc[df['Value'].idxmax(), 'Name']} - €{df['Value'].max()/1e6:.1f}M")
print(f"💰 Highest Paid Player: {df.loc[df['Wage'].idxmax(), 'Name']} - €{df['Wage'].max():,.0f}")
print(f"\n⭐ Elite Players (85+): {len(df[df['Player Level'] == 'Elite'])}")
print(f"⭐ Good Players (75-84): {len(df[df['Player Level'] == 'Good'])}")
print(f"⭐ Average Players (<75): {len(df[df['Player Level'] == 'Average'])}")
print(f"\n🏆 Best Club by Avg Overall: {df.groupby('Club')['Overall'].mean().idxmax()} ({df.groupby('Club')['Overall'].mean().max():.1f})")
