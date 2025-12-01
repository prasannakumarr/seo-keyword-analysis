import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load data
df = pd.read_csv('SEO_analysis_enhanced.csv')

print("Creating comprehensive SEO visualizations...")

# Create a large figure with multiple subplots
fig = plt.figure(figsize=(20, 24))
gs = GridSpec(6, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. Position Distribution
ax1 = fig.add_subplot(gs[0, 0])
position_bins = [0, 3, 10, 20, 50, 110]
position_labels = ['Top 3', '4-10', '11-20', '21-50', '50+']
df['Position_Category'] = pd.cut(df['Current position'], bins=position_bins, labels=position_labels)
position_counts = df['Position_Category'].value_counts().sort_index()
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
ax1.bar(position_counts.index, position_counts.values, color=colors)
ax1.set_xlabel('Position Range', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Keywords', fontsize=12, fontweight='bold')
ax1.set_title('Keyword Distribution by Position Range', fontsize=14, fontweight='bold')
for i, v in enumerate(position_counts.values):
    ax1.text(i, v + 2, str(v), ha='center', va='bottom', fontweight='bold')

# 2. Volume vs Position Scatter
ax2 = fig.add_subplot(gs[0, 1])
scatter = ax2.scatter(df['Current position'], df['Volume'],
                     c=df['KD'], s=df['CPC']*20,
                     alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)
ax2.set_xlabel('Current Position', fontsize=12, fontweight='bold')
ax2.set_ylabel('Search Volume', fontsize=12, fontweight='bold')
ax2.set_title('Volume vs Position (Size = CPC, Color = KD)', fontsize=14, fontweight='bold')
ax2.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Top 10 threshold')
ax2.axvline(x=20, color='orange', linestyle='--', alpha=0.5, label='Top 20 threshold')
ax2.legend()
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Keyword Difficulty', fontsize=10)

# 3. CPC Distribution
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(df['CPC'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
ax3.axvline(df['CPC'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["CPC"].mean():.2f}')
ax3.axvline(df['CPC'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${df["CPC"].median():.2f}')
ax3.set_xlabel('Cost Per Click ($)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('CPC Distribution', fontsize=14, fontweight='bold')
ax3.legend()

# 4. Keyword Difficulty Distribution
ax4 = fig.add_subplot(gs[1, 1])
kd_bins = [0, 30, 60, 100]
kd_labels = ['Easy (0-30)', 'Medium (31-60)', 'Hard (60+)']
df['KD_Category'] = pd.cut(df['KD'], bins=kd_bins, labels=kd_labels)
kd_counts = df['KD_Category'].value_counts()
colors_kd = ['#2ecc71', '#f39c12', '#e74c3c']
wedges, texts, autotexts = ax4.pie(kd_counts.values, labels=kd_counts.index, autopct='%1.1f%%',
                                     colors=colors_kd, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax4.set_title('Keyword Difficulty Distribution', fontsize=14, fontweight='bold')

# 5. Volume Distribution
ax5 = fig.add_subplot(gs[2, 0])
volume_bins = [0, 300, 500, 1000, 2500]
volume_labels = ['Low (200-300)', 'Medium (300-500)', 'High (500-1000)', 'Very High (1000+)']
df['Volume_Category'] = pd.cut(df['Volume'], bins=volume_bins, labels=volume_labels)
volume_counts = df['Volume_Category'].value_counts().sort_index()
ax5.bar(range(len(volume_counts)), volume_counts.values, color='#9b59b6')
ax5.set_xticks(range(len(volume_counts)))
ax5.set_xticklabels(volume_counts.index, rotation=15, ha='right')
ax5.set_xlabel('Volume Category', fontsize=12, fontweight='bold')
ax5.set_ylabel('Number of Keywords', fontsize=12, fontweight='bold')
ax5.set_title('Search Volume Distribution', fontsize=14, fontweight='bold')
for i, v in enumerate(volume_counts.values):
    ax5.text(i, v + 2, str(v), ha='center', va='bottom', fontweight='bold')

# 6. Correlation Heatmap
ax6 = fig.add_subplot(gs[2, 1])
corr_matrix = df[['Volume', 'KD', 'CPC', 'Current position']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, ax=ax6, cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1)
ax6.set_title('Correlation Matrix', fontsize=14, fontweight='bold')

# 7. Top 15 Performers
ax7 = fig.add_subplot(gs[3, :])
top_15 = df.nlargest(15, 'Performance_Score')[['Keyword', 'Performance_Score', 'Current position']]
y_pos = range(len(top_15))
bars = ax7.barh(y_pos, top_15['Performance_Score'], color='#2ecc71', alpha=0.7)
ax7.set_yticks(y_pos)
ax7.set_yticklabels(top_15['Keyword'].values, fontsize=10)
ax7.set_xlabel('Performance Score', fontsize=12, fontweight='bold')
ax7.set_title('Top 15 Performing Keywords', fontsize=14, fontweight='bold')
ax7.invert_yaxis()
for i, (score, pos) in enumerate(zip(top_15['Performance_Score'], top_15['Current position'])):
    ax7.text(score + 1, i, f'Pos: {pos:.0f}', va='center', fontsize=9)

# 8. Low-Hanging Fruit Opportunities
ax8 = fig.add_subplot(gs[4, :])
low_hanging = df[(df['KD'] <= 40) & (df['Volume'] >= 200) & (df['Current position'] > 10)]
top_opportunities = low_hanging.nsmallest(15, 'KD')[['Keyword', 'Volume', 'KD', 'Current position']]
x = range(len(top_opportunities))
width = 0.4
ax8_twin = ax8.twinx()
bars1 = ax8.bar([i - width/2 for i in x], top_opportunities['Volume'], width,
                label='Volume', color='#3498db', alpha=0.7)
bars2 = ax8_twin.bar([i + width/2 for i in x], top_opportunities['Current position'], width,
                     label='Position', color='#e74c3c', alpha=0.7)
ax8.set_xticks(x)
ax8.set_xticklabels(top_opportunities['Keyword'].values, rotation=45, ha='right', fontsize=9)
ax8.set_ylabel('Search Volume', fontsize=12, fontweight='bold', color='#3498db')
ax8_twin.set_ylabel('Current Position', fontsize=12, fontweight='bold', color='#e74c3c')
ax8.set_title('Top 15 Low-Hanging Fruit Opportunities (Easy KD, Poor Position)', fontsize=14, fontweight='bold')
ax8.legend(loc='upper left')
ax8_twin.legend(loc='upper right')
ax8.tick_params(axis='y', labelcolor='#3498db')
ax8_twin.tick_params(axis='y', labelcolor='#e74c3c')

# 9. High-Value Underperformers
ax9 = fig.add_subplot(gs[5, :])
underperformers = df[(df['Volume'] >= 500) & (df['CPC'] >= 5) & (df['Current position'] > 20)]
top_under = underperformers.nlargest(12, 'Volume')[['Keyword', 'Volume', 'CPC', 'Current position']]
x = range(len(top_under))
ax9_twin = ax9.twinx()
bars1 = ax9.bar([i - 0.2 for i in x], top_under['Volume'], 0.4,
                label='Volume', color='#9b59b6', alpha=0.7)
bars2 = ax9_twin.bar([i + 0.2 for i in x], top_under['CPC'], 0.4,
                     label='CPC ($)', color='#f39c12', alpha=0.7)
ax9.set_xticks(x)
ax9.set_xticklabels(top_under['Keyword'].values, rotation=45, ha='right', fontsize=9)
ax9.set_ylabel('Search Volume', fontsize=12, fontweight='bold', color='#9b59b6')
ax9_twin.set_ylabel('CPC ($)', fontsize=12, fontweight='bold', color='#f39c12')
ax9.set_title('Top 12 High-Value Underperformers (High Volume & CPC, Poor Position)', fontsize=14, fontweight='bold')
ax9.legend(loc='upper left')
ax9_twin.legend(loc='upper right')
ax9.tick_params(axis='y', labelcolor='#9b59b6')
ax9_twin.tick_params(axis='y', labelcolor='#f39c12')

plt.suptitle('SEO Marketing Performance Dashboard', fontsize=20, fontweight='bold', y=0.995)
plt.savefig('seo_dashboard.png', dpi=300, bbox_inches='tight')
print("Dashboard saved as: seo_dashboard.png")

# Create additional focused visualizations
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Additional SEO Insights', fontsize=18, fontweight='bold')

# 10. KD vs CPC
ax = axes[0, 0]
scatter = ax.scatter(df['KD'], df['CPC'], s=df['Volume']/5,
                    c=df['Current position'], cmap='RdYlGn_r',
                    alpha=0.6, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Keyword Difficulty', fontsize=12, fontweight='bold')
ax.set_ylabel('Cost Per Click ($)', fontsize=12, fontweight='bold')
ax.set_title('KD vs CPC (Size = Volume, Color = Position)', fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Position', fontsize=10)

# 11. Position Histogram
ax = axes[0, 1]
ax.hist(df['Current position'], bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
ax.axvline(df['Current position'].mean(), color='blue', linestyle='--',
          linewidth=2, label=f'Mean: {df["Current position"].mean():.1f}')
ax.axvline(df['Current position'].median(), color='green', linestyle='--',
          linewidth=2, label=f'Median: {df["Current position"].median():.0f}')
ax.set_xlabel('Current Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Position Distribution', fontsize=13, fontweight='bold')
ax.legend()

# 12. Volume by Position Category
ax = axes[1, 0]
position_volume = df.groupby('Position_Category')['Volume'].sum().sort_index()
bars = ax.bar(range(len(position_volume)), position_volume.values, color=colors)
ax.set_xticks(range(len(position_volume)))
ax.set_xticklabels(position_volume.index)
ax.set_xlabel('Position Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Search Volume', fontsize=12, fontweight='bold')
ax.set_title('Total Search Volume by Position Range', fontsize=13, fontweight='bold')
for i, v in enumerate(position_volume.values):
    ax.text(i, v + 500, f'{v:,}', ha='center', va='bottom', fontweight='bold')

# 13. CPC by KD Category
ax = axes[1, 1]
kd_cpc = df.groupby('KD_Category')['CPC'].agg(['mean', 'median'])
x = range(len(kd_cpc))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], kd_cpc['mean'], width, label='Mean CPC', color='#3498db', alpha=0.7)
bars2 = ax.bar([i + width/2 for i in x], kd_cpc['median'], width, label='Median CPC', color='#2ecc71', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(kd_cpc.index)
ax.set_ylabel('CPC ($)', fontsize=12, fontweight='bold')
ax.set_xlabel('Keyword Difficulty', fontsize=12, fontweight='bold')
ax.set_title('Average CPC by Keyword Difficulty', fontsize=13, fontweight='bold')
ax.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('seo_insights.png', dpi=300, bbox_inches='tight')
print("Additional insights saved as: seo_insights.png")

print("\nAll visualizations created successfully!")
print("\nFiles generated:")
print("  1. seo_dashboard.png - Comprehensive dashboard with 9 key visualizations")
print("  2. seo_insights.png - Additional focused insights")
print("  3. SEO_analysis_enhanced.csv - Enhanced dataset with performance scores")
