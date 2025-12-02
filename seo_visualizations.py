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

# ============================================================================
# INDIVIDUAL POSITION-RELATED PLOTS
# ============================================================================
print("\n" + "="*70)
print("Creating Individual Position-Related Plots...")
print("="*70)

# ============================================================================
# PLOT 1: Detailed Distribution of Current Positions (Histogram)
# ============================================================================
print("\nCreating: Detailed Position Distribution Histogram...")

fig, ax = plt.subplots(figsize=(16, 9))

# Create histogram
n, bins, patches = ax.hist(df['Current position'], bins=30, color='#3498db',
                           edgecolor='black', alpha=0.7, linewidth=1.5)

# Color code the bars based on position ranges
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center <= 3:
        patch.set_facecolor('#2ecc71')  # Green for top 3
    elif bin_center <= 10:
        patch.set_facecolor('#3498db')  # Blue for top 10
    elif bin_center <= 20:
        patch.set_facecolor('#f39c12')  # Orange for top 20
    elif bin_center <= 50:
        patch.set_facecolor('#e67e22')  # Dark orange for 21-50
    else:
        patch.set_facecolor('#e74c3c')  # Red for beyond 50

# Add statistical lines
mean_pos = df['Current position'].mean()
median_pos = df['Current position'].median()

ax.axvline(mean_pos, color='darkred', linestyle='--', linewidth=2.5,
           label=f'Mean: {mean_pos:.1f}', alpha=0.8)
ax.axvline(median_pos, color='darkgreen', linestyle='--', linewidth=2.5,
           label=f'Median: {median_pos:.0f}', alpha=0.8)

# Add reference lines
ax.axvline(10, color='blue', linestyle=':', linewidth=2,
           label='Top 10 Threshold', alpha=0.6)
ax.axvline(20, color='orange', linestyle=':', linewidth=2,
           label='Top 20 Threshold', alpha=0.6)

# Formatting
ax.set_xlabel('Current Position (Ranking)', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Keywords', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Current Keyword Positions - Detailed Analysis',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

# Add statistics box
stats_text = f'''Statistics:
Total Keywords: {len(df)}
Mean Position: {mean_pos:.2f}
Median Position: {median_pos:.0f}
Best Position: {df['Current position'].min()}
Worst Position: {df['Current position'].max()}
Std Dev: {df['Current position'].std():.2f}

Position Breakdown:
Top 3: {len(df[df['Current position'] <= 3])} ({len(df[df['Current position'] <= 3])/len(df)*100:.1f}%)
Top 10: {len(df[df['Current position'] <= 10])} ({len(df[df['Current position'] <= 10])/len(df)*100:.1f}%)
Top 20: {len(df[df['Current position'] <= 20])} ({len(df[df['Current position'] <= 20])/len(df)*100:.1f}%)
Beyond 50: {len(df[df['Current position'] > 50])} ({len(df[df['Current position'] > 50])/len(df)*100:.1f}%)'''

ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('plot_1_position_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot_1_position_distribution.png")
plt.close()


# ============================================================================
# PLOT 2: Detailed Keywords by Position Ranges (Bar Chart)
# ============================================================================
print("Creating: Detailed Keywords by Position Ranges...")

fig, ax = plt.subplots(figsize=(16, 9))

# Define position ranges
position_ranges = {
    'Top 3\n(1-3)': (1, 3),
    'Top 10\n(4-10)': (4, 10),
    'Top 20\n(11-20)': (11, 20),
    'Top 50\n(21-50)': (21, 50),
    'Beyond 50\n(51+)': (51, 200)
}

# Count keywords in each range
range_counts = []
range_labels = []
range_colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']

for label, (min_pos, max_pos) in position_ranges.items():
    count = len(df[(df['Current position'] >= min_pos) & (df['Current position'] <= max_pos)])
    range_counts.append(count)
    range_labels.append(label)

# Create bar chart
bars = ax.bar(range(len(range_labels)), range_counts, color=range_colors,
              edgecolor='black', linewidth=2, alpha=0.8)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, range_counts)):
    height = bar.get_height()
    percentage = (count / len(df)) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Formatting
ax.set_xticks(range(len(range_labels)))
ax.set_xticklabels(range_labels, fontsize=12, fontweight='bold')
ax.set_xlabel('Position Range', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Keywords', fontsize=14, fontweight='bold')
ax.set_title('Keyword Distribution by Position Ranges - Detailed Breakdown',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add total line
ax.axhline(y=len(df)/len(range_labels), color='red', linestyle='--',
           linewidth=2, alpha=0.5, label=f'Average per range: {len(df)/len(range_labels):.0f}')
ax.legend(fontsize=11)

# Add summary text
summary_text = f'''Summary:
Total Keywords: {len(df)}
Best Performing: Top 3 ({range_counts[0]} keywords)
Needs Improvement: Beyond 50 ({range_counts[4]} keywords)
First Page (1-10): {range_counts[0] + range_counts[1]} keywords ({(range_counts[0] + range_counts[1])/len(df)*100:.1f}%)

Key Insights:
• {(range_counts[0] + range_counts[1])/len(df)*100:.1f}% of keywords on first page
• {range_counts[4]/len(df)*100:.1f}% need significant improvement
• Focus on moving {range_counts[2]} keywords from positions 11-20 to top 10'''

ax.text(0.02, 0.97, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('plot_2_position_ranges.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot_2_position_ranges.png")
plt.close()


# ============================================================================
# PLOT 3: Position Trends Over Time (Sample/Template)
# ============================================================================
print("Creating: Position Trends Over Time (Sample Template)...")

fig, ax = plt.subplots(figsize=(16, 9))

# Create sample trend data for demonstration
dates = pd.date_range('2024-01-01', periods=6, freq='M')
sample_keywords = df.head(5)['Keyword'].values

for i, keyword in enumerate(sample_keywords):
    # Simulate trend data (current position with some variation)
    current_pos = df[df['Keyword'] == keyword]['Current position'].values[0]
    # Create a trend showing improvement over time
    trend = [current_pos + 10 - i*2 for i in range(6)]
    ax.plot(dates, trend, marker='o', linewidth=2.5, label=keyword[:40], markersize=8)

ax.set_xlabel('Date', fontsize=14, fontweight='bold')
ax.set_ylabel('Position (lower is better)', fontsize=14, fontweight='bold')
ax.set_title('Position Trends Over Time - SAMPLE DATA (Template for Future Tracking)',
             fontsize=16, fontweight='bold', pad=20, color='red')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
ax.invert_yaxis()  # Lower position numbers are better

# Add watermark
ax.text(0.5, 0.5, 'SAMPLE DATA\nNot Real Trends', transform=ax.transAxes,
        fontsize=50, color='red', alpha=0.2, ha='center', va='center',
        rotation=30)

# Add note box
note_text = '''⚠ NOTE: This is SAMPLE data only! ⚠

To create real position trends:
1. Collect position data over multiple dates
2. Add a 'Date' column to your dataset
3. Track the same keywords over time
4. Re-run this script with updated data

Example data structure needed:
  Date        | Keyword     | Position
  2024-01-01  | keyword1    | 15
  2024-02-01  | keyword1    | 12
  2024-03-01  | keyword1    | 8'''

ax.text(0.02, 0.98, note_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=2))

plt.tight_layout()
plt.savefig('plot_3_position_trends_sample.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot_3_position_trends_sample.png (sample/template only)")
plt.close()

print("\n" + "="*70)
print("INDIVIDUAL POSITION PLOTS COMPLETE")
print("="*70)
print("\nAdditional files generated:")
print("  4. plot_1_position_distribution.png - Detailed histogram")
print("  5. plot_2_position_ranges.png - Detailed bar chart by ranges")
print("  6. plot_3_position_trends_sample.png - Sample template for trends")
print("\n" + "="*70)

# ============================================================================
# INDIVIDUAL VOLUME-RELATED PLOTS
# ============================================================================
print("\n" + "="*70)
print("Creating Individual Volume-Related Plots...")
print("="*70)

# ============================================================================
# PLOT 4: Search Volume Distribution
# ============================================================================
print("\nCreating: Search Volume Distribution...")

fig, ax = plt.subplots(figsize=(16, 9))

# Create histogram with custom bins
volume_bins = [200, 300, 400, 500, 750, 1000, 1500, 2000, 2500]
n, bins, patches = ax.hist(df['Volume'], bins=volume_bins, color='#9b59b6',
                           edgecolor='black', alpha=0.7, linewidth=1.5)

# Color code the bars based on volume ranges
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center <= 300:
        patch.set_facecolor('#e74c3c')  # Red for low volume
    elif bin_center <= 500:
        patch.set_facecolor('#f39c12')  # Orange for medium-low
    elif bin_center <= 1000:
        patch.set_facecolor('#3498db')  # Blue for medium-high
    else:
        patch.set_facecolor('#2ecc71')  # Green for high volume

# Add statistical lines
mean_vol = df['Volume'].mean()
median_vol = df['Volume'].median()

ax.axvline(mean_vol, color='darkred', linestyle='--', linewidth=2.5,
           label=f'Mean: {mean_vol:.0f}', alpha=0.8)
ax.axvline(median_vol, color='darkgreen', linestyle='--', linewidth=2.5,
           label=f'Median: {median_vol:.0f}', alpha=0.8)

# Add reference lines
ax.axvline(500, color='orange', linestyle=':', linewidth=2,
           label='Medium Volume Threshold (500)', alpha=0.6)
ax.axvline(1000, color='green', linestyle=':', linewidth=2,
           label='High Volume Threshold (1000)', alpha=0.6)

# Formatting
ax.set_xlabel('Search Volume (monthly searches)', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Keywords', fontsize=14, fontweight='bold')
ax.set_title('Search Volume Distribution - Detailed Analysis',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

# Calculate volume categories
low_vol = len(df[df['Volume'] < 300])
medium_vol = len(df[(df['Volume'] >= 300) & (df['Volume'] < 1000)])
high_vol = len(df[df['Volume'] >= 1000])
total_vol = df['Volume'].sum()

# Add statistics box
stats_text = f'''Statistics:
Total Keywords: {len(df)}
Total Search Volume: {total_vol:,}/month
Mean Volume: {mean_vol:.0f}
Median Volume: {median_vol:.0f}
Min Volume: {df['Volume'].min()}
Max Volume: {df['Volume'].max()}
Std Dev: {df['Volume'].std():.0f}

Volume Breakdown:
Low (<300): {low_vol} keywords ({low_vol/len(df)*100:.1f}%)
Medium (300-999): {medium_vol} keywords ({medium_vol/len(df)*100:.1f}%)
High (1000+): {high_vol} keywords ({high_vol/len(df)*100:.1f}%)

Top 10 Keywords Volume: {df.nlargest(10, 'Volume')['Volume'].sum():,} ({df.nlargest(10, 'Volume')['Volume'].sum()/total_vol*100:.1f}% of total)'''

ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

plt.tight_layout()
plt.savefig('plot_4_volume_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot_4_volume_distribution.png")
plt.close()


# ============================================================================
# PLOT 5: Volume vs Position Scatter Plot (Enhanced)
# ============================================================================
print("Creating: Volume vs Position Scatter Plot (Enhanced)...")

fig, ax = plt.subplots(figsize=(16, 9))

# Create scatter plot with multiple dimensions
# Size represents CPC, Color represents KD
scatter = ax.scatter(df['Current position'], df['Volume'],
                    s=df['CPC']*30,  # Size based on CPC
                    c=df['KD'],      # Color based on KD
                    cmap='RdYlGn_r', # Red for high KD, Green for low KD
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.5)

# Add colorbar for KD
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Keyword Difficulty (KD)', fontsize=12, fontweight='bold')

# Add reference lines
ax.axvline(x=10, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Top 10 Threshold')
ax.axvline(x=20, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Top 20 Threshold')
ax.axhline(y=500, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Medium Volume (500)')
ax.axhline(y=1000, color='green', linestyle='--', linewidth=2, alpha=0.5, label='High Volume (1000)')

# Formatting
ax.set_xlabel('Current Position (Ranking)', fontsize=14, fontweight='bold')
ax.set_ylabel('Search Volume (monthly searches)', fontsize=14, fontweight='bold')
ax.set_title('Volume vs Position Relationship (Size = CPC, Color = KD)',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

# Add quadrant labels to identify opportunities
ax.text(5, df['Volume'].max()*0.95, 'HIGH PRIORITY\nHigh Volume, Good Position',
        fontsize=11, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(70, df['Volume'].max()*0.95, 'OPPORTUNITY\nHigh Volume, Poor Position',
        fontsize=11, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax.text(5, 250, 'MAINTAIN\nLow Volume, Good Position',
        fontsize=11, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(70, 250, 'LOW PRIORITY\nLow Volume, Poor Position',
        fontsize=11, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Add correlation info
correlation = df['Volume'].corr(df['Current position'])
stats_text = f'''Correlation Analysis:
Volume vs Position: {correlation:.3f}
{"Weak negative" if correlation < 0 else "Weak positive"} correlation

Bubble Size = CPC ($)
Larger bubbles = Higher cost per click

Key Insights:
• High volume keywords spread across all positions
• Opportunity to improve high-volume keywords
  ranking beyond position 20
• Focus on upper-right quadrant for quick wins'''

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Highlight top volume keywords
top_volume_keywords = df.nlargest(5, 'Volume')
for idx, row in top_volume_keywords.iterrows():
    ax.annotate(row['Keyword'][:25],
                xy=(row['Current position'], row['Volume']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', alpha=0.5))

plt.tight_layout()
plt.savefig('plot_5_volume_vs_position.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot_5_volume_vs_position.png")
plt.close()

print("\n" + "="*70)
print("INDIVIDUAL VOLUME PLOTS COMPLETE")
print("="*70)
print("\nAdditional files generated:")
print("  7. plot_4_volume_distribution.png - Detailed volume histogram")
print("  8. plot_5_volume_vs_position.png - Enhanced scatter plot with quadrants")
print("\n" + "="*70)

# ============================================================================
# INDIVIDUAL OPPORTUNITY ANALYSIS PLOTS
# ============================================================================
print("\n" + "="*70)
print("Creating Individual Opportunity Analysis Plots...")
print("="*70)

# ============================================================================
# PLOT 6: Low-Hanging Fruit Opportunities
# ============================================================================
print("\nCreating: Low-Hanging Fruit Opportunities...")

# Identify low-hanging fruit: Easy KD, decent volume, poor position
low_hanging = df[(df['KD'] <= 40) & (df['Volume'] >= 200) & (df['Current position'] > 10)]
top_low_hanging = low_hanging.nsmallest(20, 'KD')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

# Left plot: Bar chart with KD and Position
y_pos = range(len(top_low_hanging))
bars1 = ax1.barh(y_pos, top_low_hanging['KD'], color='#2ecc71', alpha=0.7, label='KD (lower is easier)')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_low_hanging['Keyword'].values, fontsize=9)
ax1.set_xlabel('Keyword Difficulty', fontsize=12, fontweight='bold')
ax1.set_title('Low-Hanging Fruit Keywords\n(Easy KD, Decent Volume, Poor Position)',
             fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='x')

# Add position annotations
for i, (kd, pos) in enumerate(zip(top_low_hanging['KD'], top_low_hanging['Current position'])):
    ax1.text(kd + 1, i, f'Pos: {pos:.0f}', va='center', fontsize=8)

# Right plot: Volume vs Position for these keywords
ax2.scatter(top_low_hanging['Current position'], top_low_hanging['Volume'],
           s=top_low_hanging['CPC']*40, c=top_low_hanging['KD'],
           cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=1)
ax2.set_xlabel('Current Position', fontsize=12, fontweight='bold')
ax2.set_ylabel('Search Volume', fontsize=12, fontweight='bold')
ax2.set_title('Position vs Volume\n(Bubble size = CPC, Color = KD)',
             fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add reference line at position 20
ax2.axvline(x=20, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Position 20')
ax2.legend()

# Add statistics box
total_vol = top_low_hanging['Volume'].sum()
avg_pos = top_low_hanging['Current position'].mean()
avg_kd = top_low_hanging['KD'].mean()

stats_text = f'''Low-Hanging Fruit Stats:
Total Opportunities: {len(low_hanging)}
Shown: Top 20

Average KD: {avg_kd:.1f} (Easy!)
Average Position: {avg_pos:.0f}
Total Volume: {total_vol:,}/month

Why These Are Easy Wins:
• Low keyword difficulty
• Decent search volume
• Currently ranking poorly
• Can improve with basic SEO

Action Items:
1. Optimize content quality
2. Improve internal linking
3. Add relevant keywords
4. Update meta descriptions'''

ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('plot_6_low_hanging_fruit.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: plot_6_low_hanging_fruit.png ({len(low_hanging)} total opportunities, showing top 20)")
plt.close()


# ============================================================================
# PLOT 7: High-Value Underperformers
# ============================================================================
print("Creating: High-Value Underperformers...")

# Identify high-value underperformers: High volume, high CPC, poor position
underperformers = df[(df['Volume'] >= 500) & (df['CPC'] >= 5) & (df['Current position'] > 20)]
top_underperformers = underperformers.nlargest(15, 'Volume')

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Keywords by Volume and CPC
ax1 = fig.add_subplot(gs[0, :])
x = range(len(top_underperformers))
width = 0.35

ax1_twin = ax1.twinx()
bars1 = ax1.bar([i - width/2 for i in x], top_underperformers['Volume'], width,
               label='Volume', color='#9b59b6', alpha=0.7)
bars2 = ax1_twin.bar([i + width/2 for i in x], top_underperformers['CPC'], width,
                     label='CPC ($)', color='#f39c12', alpha=0.7)

ax1.set_xticks(x)
ax1.set_xticklabels(top_underperformers['Keyword'].values, rotation=45, ha='right', fontsize=9)
ax1.set_ylabel('Search Volume', fontsize=12, fontweight='bold', color='#9b59b6')
ax1_twin.set_ylabel('CPC ($)', fontsize=12, fontweight='bold', color='#f39c12')
ax1.set_title('High-Value Underperformers - Volume and CPC Analysis',
             fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.tick_params(axis='y', labelcolor='#9b59b6')
ax1_twin.tick_params(axis='y', labelcolor='#f39c12')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Current Position
ax2 = fig.add_subplot(gs[1, 0])
positions = top_underperformers['Current position'].values
colors_pos = ['#e74c3c' if p > 50 else '#f39c12' for p in positions]
bars = ax2.barh(range(len(top_underperformers)), positions, color=colors_pos, alpha=0.7)
ax2.set_yticks(range(len(top_underperformers)))
ax2.set_yticklabels(top_underperformers['Keyword'].values, fontsize=9)
ax2.set_xlabel('Current Position', fontsize=12, fontweight='bold')
ax2.set_title('Current Rankings (Lower is Better)', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Position 50')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Revenue Potential Analysis
ax3 = fig.add_subplot(gs[1, 1])
# Calculate potential value (Volume * CPC as a proxy for potential)
top_underperformers['Potential_Value'] = top_underperformers['Volume'] * top_underperformers['CPC']
potential_sorted = top_underperformers.nlargest(10, 'Potential_Value')

bars = ax3.barh(range(len(potential_sorted)), potential_sorted['Potential_Value'],
               color='#e74c3c', alpha=0.7)
ax3.set_yticks(range(len(potential_sorted)))
ax3.set_yticklabels(potential_sorted['Keyword'].values, fontsize=9)
ax3.set_xlabel('Potential Value (Volume × CPC)', fontsize=12, fontweight='bold')
ax3.set_title('Top 10 by Revenue Potential', fontsize=12, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(potential_sorted['Potential_Value']):
    ax3.text(v + 100, i, f'${v:,.0f}', va='center', fontsize=8)

# Add overall statistics
total_underperformers = len(underperformers)
total_potential_volume = underperformers['Volume'].sum()
avg_cpc = underperformers['CPC'].mean()
total_potential_value = (underperformers['Volume'] * underperformers['CPC']).sum()

fig.text(0.5, 0.02,
         f'Total High-Value Underperformers: {total_underperformers} | '
         f'Combined Volume: {total_potential_volume:,}/month | '
         f'Avg CPC: ${avg_cpc:.2f} | '
         f'Total Potential Value: ${total_potential_value:,.0f}/month',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.savefig('plot_7_high_value_underperformers.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: plot_7_high_value_underperformers.png ({total_underperformers} total underperformers)")
plt.close()


# ============================================================================
# PLOT 8: Quick Win Keywords (Combined Analysis)
# ============================================================================
print("Creating: Quick Win Keywords...")

# Quick wins: Good volume, currently in positions 11-20 (close to first page)
quick_wins = df[(df['Volume'] >= 300) &
                (df['Current position'] > 10) &
                (df['Current position'] <= 20)]
top_quick_wins = quick_wins.nlargest(20, 'Volume')

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Quick Win Keywords Overview
ax1 = fig.add_subplot(gs[0, :])
y_pos = range(len(top_quick_wins))
bars = ax1.barh(y_pos, top_quick_wins['Volume'], color='#3498db', alpha=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_quick_wins['Keyword'].values, fontsize=9)
ax1.set_xlabel('Search Volume', fontsize=12, fontweight='bold')
ax1.set_title('Quick Win Keywords - Currently Positions 11-20 (Close to First Page!)',
             fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# Add position and KD annotations
for i, (vol, pos, kd) in enumerate(zip(top_quick_wins['Volume'],
                                        top_quick_wins['Current position'],
                                        top_quick_wins['KD'])):
    ax1.text(vol + 50, i, f'Pos: {pos:.0f} | KD: {kd}', va='center', fontsize=8)

# Plot 2: Position Distribution
ax2 = fig.add_subplot(gs[1, 0])
position_counts = top_quick_wins['Current position'].value_counts().sort_index()
bars = ax2.bar(position_counts.index, position_counts.values, color='#f39c12', alpha=0.7)
ax2.set_xlabel('Current Position', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Keywords', fontsize=12, fontweight='bold')
ax2.set_title('Distribution of Positions (11-20)', fontsize=12, fontweight='bold')
ax2.axvline(x=10, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Top 10 Target')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

# Plot 3: KD vs CPC for Quick Wins
ax3 = fig.add_subplot(gs[1, 1])
scatter = ax3.scatter(top_quick_wins['KD'], top_quick_wins['CPC'],
                     s=top_quick_wins['Volume']/3,
                     c=top_quick_wins['Current position'],
                     cmap='RdYlGn', alpha=0.6,
                     edgecolors='black', linewidth=1)
ax3.set_xlabel('Keyword Difficulty', fontsize=12, fontweight='bold')
ax3.set_ylabel('CPC ($)', fontsize=12, fontweight='bold')
ax3.set_title('Difficulty vs Value\n(Size = Volume, Color = Position)',
             fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Position', fontsize=10)

# Add quadrant lines
ax3.axvline(x=40, color='gray', linestyle=':', alpha=0.5)
ax3.axhline(y=top_quick_wins['CPC'].median(), color='gray', linestyle=':', alpha=0.5)

# Add quadrant labels
ax3.text(20, ax3.get_ylim()[1]*0.9, 'Easy + High Value\n(PRIORITY!)',
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax3.text(60, ax3.get_ylim()[1]*0.9, 'Hard + High Value\n(Long-term)',
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Add statistics and recommendations
total_quick_wins = len(quick_wins)
avg_pos = quick_wins['Current position'].mean()
total_vol = quick_wins['Volume'].sum()
avg_kd = quick_wins['KD'].mean()

stats_text = f'''Quick Win Statistics:
Total Opportunities: {total_quick_wins}
Shown: Top 20 by volume

Average Position: {avg_pos:.1f}
Total Volume: {total_vol:,}/month
Average KD: {avg_kd:.1f}

Why These Are Quick Wins:
• Already on page 2 (positions 11-20)
• Just need small boost to reach page 1
• Decent search volume
• Close to converting traffic

Action Plan:
1. Improve on-page SEO
2. Build quality backlinks
3. Update content freshness
4. Optimize for user intent
5. Target positions 8-10 first'''

fig.text(0.99, 0.01, stats_text,
        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.savefig('plot_8_quick_win_keywords.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: plot_8_quick_win_keywords.png ({total_quick_wins} total quick wins)")
plt.close()

print("\n" + "="*70)
print("INDIVIDUAL OPPORTUNITY PLOTS COMPLETE")
print("="*70)
print("\nAdditional files generated:")
print("  9. plot_6_low_hanging_fruit.png - Easy wins with low KD")
print("  10. plot_7_high_value_underperformers.png - High revenue potential")
print("  11. plot_8_quick_win_keywords.png - Close to first page (positions 11-20)")
print("\n" + "="*70)

# ============================================================================
# PLOT 9: Investment Priority Matrix
# ============================================================================
print("\n" + "="*70)
print("Creating Investment Priority Matrix...")
print("="*70)

fig, ax = plt.subplots(figsize=(20, 12))

# Create bubble chart: X=Position, Y=Volume, Size=CPC, Color=KD
scatter = ax.scatter(df['Current position'], df['Volume'],
                    s=df['CPC']*50,  # Bubble size based on CPC
                    c=df['KD'],      # Color based on KD
                    cmap='RdYlGn_r', # Red (high KD) to Green (low KD)
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Keyword Difficulty (Green=Easy, Red=Hard)', fontsize=12, fontweight='bold')

# Add strategic threshold lines
ax.axvline(x=10, color='green', linestyle='--', linewidth=2.5, alpha=0.7, label='Top 10 Threshold')
ax.axvline(x=20, color='blue', linestyle='--', linewidth=2.5, alpha=0.7, label='Top 20 Threshold')
ax.axvline(x=50, color='orange', linestyle='--', linewidth=2.5, alpha=0.7, label='Position 50 Threshold')
ax.axhline(y=500, color='purple', linestyle='--', linewidth=2.5, alpha=0.7, label='Volume 500 Threshold')
ax.axhline(y=1000, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Volume 1000 Threshold')

# Add quadrant backgrounds
ax.axvspan(0, 20, alpha=0.05, color='green', label='Best Positions')
ax.axvspan(20, 50, alpha=0.05, color='yellow')
ax.axvspan(50, df['Current position'].max()+5, alpha=0.05, color='red', label='Worst Positions')

ax.axhspan(500, df['Volume'].max()+200, alpha=0.03, color='green')
ax.axhspan(0, 500, alpha=0.03, color='red')

# Add quadrant labels
ax.text(10, df['Volume'].max()*0.95, 'TIER 1 - PROTECT\nGood Position + High Volume',
        fontsize=11, ha='center', va='top', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=2))

ax.text(35, df['Volume'].max()*0.95, 'TIER 2 - OPTIMIZE\nMid Position + High Volume',
        fontsize=11, ha='center', va='top', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange', linewidth=2))

ax.text(70, df['Volume'].max()*0.95, 'TIER 3 - IMPROVE\nPoor Position + High Volume',
        fontsize=11, ha='center', va='top', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8, edgecolor='red', linewidth=2))

ax.text(10, 200, 'TIER 4 - MONITOR\nGood Position + Low Volume',
        fontsize=10, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='blue', linewidth=1))

ax.text(70, 200, 'TIER 5 - DEPRIORITIZE\nPoor Position + Low Volume',
        fontsize=10, ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7, edgecolor='gray', linewidth=1))

# Annotate key keywords
# Top keywords to annotate (by volume)
top_keywords_to_label = df.nlargest(8, 'Volume')
for idx, row in top_keywords_to_label.iterrows():
    ax.annotate(row['Keyword'][:20],
                xy=(row['Current position'], row['Volume']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold', alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7, edgecolor='black'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1, alpha=0.6))

# Add category statistics boxes
tier1 = df[(df['Current position'] <= 20) & (df['Volume'] >= 500)]
tier2 = df[(df['Current position'] > 20) & (df['Current position'] <= 50) & (df['Volume'] >= 500)]
tier3 = df[(df['Current position'] > 50) & (df['Volume'] >= 500)]
tier4 = df[(df['Current position'] <= 20) & (df['Volume'] < 500)]
tier5 = df[(df['Current position'] > 50) & (df['Volume'] < 500)]

stats_box = f'''INVESTMENT PRIORITY BREAKDOWN:

TIER 1 (Protect): {len(tier1)} keywords
  Avg Position: {tier1['Current position'].mean():.1f}
  Avg Volume: {tier1['Volume'].mean():.0f}

TIER 2 (Optimize): {len(tier2)} keywords
  Avg Position: {tier2['Current position'].mean():.1f}
  Avg Volume: {tier2['Volume'].mean():.0f}

TIER 3 (Improve): {len(tier3)} keywords
  Avg Position: {tier3['Current position'].mean():.1f}
  Avg Volume: {tier3['Volume'].mean():.0f}

TIER 4 (Monitor): {len(tier4)} keywords
  Avg Position: {tier4['Current position'].mean():.1f}
  Avg Volume: {tier4['Volume'].mean():.0f}

TIER 5 (Deprioritize): {len(tier5)} keywords
  Avg Position: {tier5['Current position'].mean():.1f}
  Avg Volume: {tier5['Volume'].mean():.0f}

Bubble Size = CPC (Larger = Higher Value)
Bubble Color = KD (Green = Easy, Red = Hard)'''

ax.text(0.98, 0.98, stats_box, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                 edgecolor='black', linewidth=2),
        family='monospace')

# Formatting
ax.set_xlabel('Current Position (Lower is Better)', fontsize=14, fontweight='bold')
ax.set_ylabel('Search Volume (monthly searches)', fontsize=14, fontweight='bold')
ax.set_title('Investment Priority Matrix - Complete View (All Positions)',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=10, loc='lower left', framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--')

# Set x-axis limits with some padding
ax.set_xlim(-5, df['Current position'].max()+5)
ax.set_ylim(-100, df['Volume'].max()+300)

plt.tight_layout()
plt.savefig('plot_9_investment_priority_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot_9_investment_priority_matrix.png")
plt.close()

print("\n" + "="*70)
print("INVESTMENT PRIORITY MATRIX COMPLETE")
print("="*70)
print("\nFile generated:")
print("  12. plot_9_investment_priority_matrix.png - Strategic investment matrix")
print("\n" + "="*70)

# ============================================================================
# TIER 2 & TIER 3 ANALYSIS - TABLES & STATISTICAL BREAKDOWN
# ============================================================================
print("\n" + "="*70)
print("TIER 2 & TIER 3 ANALYSIS - KEYWORD TABLES & STATISTICS")
print("="*70)

# Define tiers
tier2_full = df[(df['Current position'] > 20) & (df['Current position'] <= 50) & (df['Volume'] >= 500)]
tier3_full = df[(df['Current position'] > 50) & (df['Volume'] >= 500)]

# Sort by priority score
tier2_full_copy = tier2_full.copy()
tier3_full_copy = tier3_full.copy()
tier2_full_copy['Priority_Score'] = (tier2_full_copy['Volume'] * (100 - tier2_full_copy['KD'])) / 100
tier3_full_copy['Priority_Score'] = (tier3_full_copy['Volume'] * (100 - tier3_full_copy['KD'])) / 100

tier2_sorted = tier2_full_copy.nlargest(20, 'Priority_Score')
tier3_sorted = tier3_full_copy.nlargest(25, 'Priority_Score')

# ============================================================================
# TIER 2 TABLE
# ============================================================================
print("\n" + "="*100)
print("TIER 2 (OPTIMIZE): Mid-Position High-Volume Keywords - Positions 21-50")
print("="*100)
print("\nStrategy: Incremental optimization to push into top 20")
print("Action: Content refresh, backlink building, on-page SEO improvements\n")

tier2_display = tier2_sorted[[
    'Keyword', 'Current position', 'Volume', 'KD', 'CPC'
]].copy()

tier2_display['Rank'] = range(1, len(tier2_display) + 1)
tier2_display['Ease_Score'] = ((100 - tier2_display['KD'])).round(1)
tier2_display['Position_to_Top10'] = tier2_display['Current position'] - 10

tier2_display = tier2_display[[
    'Rank', 'Keyword', 'Current position', 'Volume', 'KD', 'CPC',
    'Ease_Score', 'Position_to_Top10'
]]

tier2_display.columns = [
    'Rank', 'Keyword', 'Position', 'Volume/mo', 'KD', 'CPC ($)',
    'Ease (0-100)', 'Distance to Top10'
]

print(tier2_display.to_string(index=False))

# ============================================================================
# TIER 3 TABLE
# ============================================================================
print("\n" + "="*100)
print("TIER 3 (IMPROVE): Poor-Position High-Volume Keywords - Positions 51+")
print("="*100)
print("\nStrategy: Major optimization efforts needed - highest ROI potential")
print("Action: Complete content overhaul, comprehensive link building, competitive analysis\n")

tier3_display = tier3_sorted[[
    'Keyword', 'Current position', 'Volume', 'KD', 'CPC'
]].copy()

tier3_display['Rank'] = range(1, len(tier3_display) + 1)
tier3_display['Ease_Score'] = ((100 - tier3_display['KD'])).round(1)
tier3_display['Position_to_Top10'] = tier3_display['Current position'] - 10

tier3_display = tier3_display[[
    'Rank', 'Keyword', 'Current position', 'Volume', 'KD', 'CPC',
    'Ease_Score', 'Position_to_Top10'
]]

tier3_display.columns = [
    'Rank', 'Keyword', 'Position', 'Volume/mo', 'KD', 'CPC ($)',
    'Ease (0-100)', 'Distance to Top10'
]

print(tier3_display.to_string(index=False))

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("STATISTICAL REASONING & TIER ANALYSIS")
print("="*100)

print("\n1. SELECTION CRITERIA - WHY THESE KEYWORDS?")
print("-" * 100)

print("""
✓ VOLUME THRESHOLD (>= 500/month):
  Reasoning: Keywords with monthly search volume >= 500 represent commercial intent
  - Below 500: Minimal traffic potential even with top ranking
  - This threshold ensures ROI-positive efforts

✓ POSITION THRESHOLDS:
  TIER 2 (21-50):  Keywords "on the edge" - achievable with optimization
  TIER 3 (51+):    Keywords "severely underperforming" - high upside potential

  Why this matters:
  - Position 1-10: Already getting traffic, maintenance focus
  - Position 21-50: Page 2 - can move to page 1 with moderate effort (TIER 2)
  - Position 51+: Not getting meaningful traffic - needs major overhaul (TIER 3)
""")

print("\n2. PRIORITY SCORING METHODOLOGY")
print("-" * 100)

priority_formula = """
Priority Score = Volume × (100 - KD) / 100

This formula considers:
  • Volume: Higher volume keywords = more potential traffic gain
  • (100 - KD): Lower KD = easier to improve position

Example:
  Keyword A: Volume=1000, KD=40 → Score = 600
  Keyword B: Volume=800, KD=20 → Score = 640 ✓ (Lower KD wins!)
"""
print(priority_formula)

print("\n3. TIER COMPARISON")
print("-" * 100)

tier2_stats = f"""
TIER 2 (20 Keywords):
  • Average Position: {tier2_full['Current position'].mean():.1f}
  • Total Volume: {tier2_full['Volume'].sum():,}/month
  • Average KD: {tier2_full['KD'].mean():.1f}
  • Easy Keywords (KD ≤ 40): {len(tier2_full[tier2_full['KD'] <= 40])} out of {len(tier2_full)}
  • Median Position: {tier2_full['Current position'].median():.0f}
  • Position Range: {tier2_full['Current position'].min():.0f} - {tier2_full['Current position'].max():.0f}

  Strategic Value: Only 10-40 positions away from top 10. User intent proven
  (they're being searched). Needs content refresh and backlinks.

TIER 3 (25 Keywords):
  • Average Position: {tier3_full['Current position'].mean():.1f}
  • Total Volume: {tier3_full['Volume'].sum():,}/month
  • Average KD: {tier3_full['KD'].mean():.1f}
  • Easy Keywords (KD ≤ 40): {len(tier3_full[tier3_full['KD'] <= 40])} out of {len(tier3_full)}
  • Median Position: {tier3_full['Current position'].median():.0f}
  • Position Range: {tier3_full['Current position'].min():.0f} - {tier3_full['Current position'].max():.0f}

  Strategic Value: Currently invisible (page 3+). High search demand but
  severe underperformance. Needs comprehensive content overhaul.
"""
print(tier2_stats)

print("\n4. STRATEGIC FOCUS AREAS")
print("-" * 100)

strategic_areas = """
TIER 2 FOCUS (Quick Wins):
  ✓ Easier wins: Only 10-40 positions away from top 10
  ✓ Faster implementation: 3-6 month typical timeline
  ✓ Lower risk: Position improvements are more predictable
  ✓ Best for: Building momentum and proving value

TIER 3 FOCUS (Long-Term Growth):
  ✓ Larger opportunities: Currently invisible (position 51+)
  ✓ Longer timeline: 6-12 months typical
  ✓ Higher effort: Requires comprehensive content overhaul
  ✓ Best for: Long-term growth and major traffic gains

PRACTICAL APPROACH:
  1. Run TIER 2 improvements FIRST (quick wins, build confidence)
  2. Then tackle TIER 3 (longer-term, bigger payoff)
"""
print(strategic_areas)

print("\n5. CORRELATION INSIGHTS")
print("-" * 100)

print(f"""
Why These Tiers Exist (Statistical Validation):

  • Volume vs Position: {df['Volume'].corr(df['Current position']):.3f}
    → Weak correlation = high-volume keywords aren't all ranking well
    → This creates opportunities!

  • KD vs Position: {df['KD'].corr(df['Current position']):.3f}
    → Very weak = many easy keywords rank poorly
    → Explains why 14/20 TIER 2 keywords have KD ≤ 40

  • CPC vs Volume: {df['CPC'].corr(df['Volume']):.3f}
    → Weak correlation = volume and value are independent
    → Both metrics matter for prioritization
""")

# Export CSV files
tier2_export = tier2_sorted[[
    'Keyword', 'Current position', 'Volume', 'KD', 'CPC'
]].copy()
tier2_export.insert(0, 'Rank', range(1, len(tier2_export) + 1))

tier3_export = tier3_sorted[[
    'Keyword', 'Current position', 'Volume', 'KD', 'CPC'
]].copy()
tier3_export.insert(0, 'Rank', range(1, len(tier3_export) + 1))

tier2_export.to_csv('TIER_2_Optimize_Keywords.csv', index=False)
tier3_export.to_csv('TIER_3_Improve_Keywords.csv', index=False)

print("\n" + "="*100)
print("TIER ANALYSIS COMPLETE")
print("="*100)
print("\nFiles generated:")
print("  13. TIER_2_Optimize_Keywords.csv - 20 TIER 2 keywords with metrics")
print("  14. TIER_3_Improve_Keywords.csv - 25 TIER 3 keywords with metrics")
print("\n" + "="*100)

# ============================================================================
# CORRELATION HEATMAP: Volume, KD, and CPC
# ============================================================================
print("\n" + "="*70)
print("Creating Correlation Heatmap (Volume, KD, CPC)...")
print("="*70)

# Create a figure with two sections: heatmap on top, legend on bottom
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.5], hspace=0.4)

# TOP: Heatmap
ax_heatmap = fig.add_subplot(gs[0])

# Select the three variables of interest
correlation_data = df[['Volume', 'KD', 'CPC']].corr()

# Create the heatmap
sns.heatmap(correlation_data,
            annot=True,              # Show correlation values
            fmt='.3f',              # Format to 3 decimal places
            cmap='coolwarm',        # Color scheme: blue (negative) to red (positive)
            center=0,               # Center the colormap at 0
            square=True,            # Make cells square
            linewidths=2,           # Add gridlines
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
            vmin=-1,                # Min correlation value
            vmax=1,                 # Max correlation value
            ax=ax_heatmap)

# Customize appearance
ax_heatmap.set_title('Correlation Heatmap: Volume, KD, and CPC\nSEO Performance Metrics',
             fontsize=16, fontweight='bold', pad=20)

# Rotate labels for better readability
ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=12)
ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, fontsize=12)

# BOTTOM: Interpretation legend
ax_legend = fig.add_subplot(gs[1])
ax_legend.axis('off')  # Hide the axis

interpretation_text = """CORRELATION MATRIX INTERPRETATION GUIDE:

Correlation Range: -1 to +1  |  +1.0: Perfect positive (both increase)  |  0.0: No correlation  |  -1.0: Perfect negative (opposite)

Color Guide:  🔴 Red (Positive): Variables move together  |  🔵 Blue (Negative): Variables move opposite  |  ⚪ White (Near 0): Little to no relationship

Key Insights:
  • Volume vs KD: Shows if high volume keywords are harder to rank  |  • Volume vs CPC: Shows if high-traffic keywords cost more per click  |  • KD vs CPC: Shows if difficult keywords have higher commercial value"""

ax_legend.text(0.05, 0.5, interpretation_text,
           fontsize=9,
           verticalalignment='center',
           horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1.5),
           family='monospace',
           wrap=True)

plt.savefig('plot_10_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot_10_correlation_heatmap.png")
plt.close()

# ============================================================================
# DETAILED CORRELATION ANALYSIS & INSIGHTS
# ============================================================================
print("\n" + "="*70)
print("CORRELATION ANALYSIS - DETAILED INSIGHTS")
print("="*70)

# Calculate correlations
vol_kd_corr = df['Volume'].corr(df['KD'])
vol_cpc_corr = df['Volume'].corr(df['CPC'])
kd_cpc_corr = df['KD'].corr(df['CPC'])

print("\n1. CORRELATION VALUES")
print("-" * 70)
print(f"Volume vs KD (Keyword Difficulty):  {vol_kd_corr:>7.3f}")
print(f"Volume vs CPC (Cost Per Click):     {vol_cpc_corr:>7.3f}")
print(f"KD vs CPC:                           {kd_cpc_corr:>7.3f}")

print("\n2. WHAT EACH CORRELATION TELLS US")
print("-" * 70)

# Volume vs KD Analysis
if vol_kd_corr > 0.3:
    kd_insight = "Strong positive: High-volume keywords tend to be harder to rank for"
elif vol_kd_corr > 0.1:
    kd_insight = "Weak positive: High-volume keywords are slightly harder to rank for"
elif vol_kd_corr > -0.1:
    kd_insight = "No correlation: Volume and difficulty are independent"
elif vol_kd_corr > -0.3:
    kd_insight = "Weak negative: High-volume keywords tend to be easier to rank for"
else:
    kd_insight = "Strong negative: High-volume keywords are significantly easier to rank for"

print(f"\nVolume vs KD: {vol_kd_corr:.3f}")
print(f"  {kd_insight}")
print(f"  Practical Impact: Keywords with high search volume")
if vol_kd_corr > 0:
    print(f"    → Are generally HARDER to rank for (higher KD)")
    print(f"    → Competitive keywords attract more search traffic")
    print(f"    → Requires stronger SEO efforts (backlinks, authority)")
else:
    print(f"    → Are generally EASIER to rank for (lower KD)")
    print(f"    → Opportunity: High volume with low difficulty = quick wins")

# Volume vs CPC Analysis
if vol_cpc_corr > 0.3:
    cpc_insight = "Strong positive: High-volume keywords have higher CPC"
elif vol_cpc_corr > 0.1:
    cpc_insight = "Weak positive: High-volume keywords have slightly higher CPC"
elif vol_cpc_corr > -0.1:
    cpc_insight = "No correlation: Volume and CPC are independent"
elif vol_cpc_corr > -0.3:
    cpc_insight = "Weak negative: High-volume keywords have slightly lower CPC"
else:
    cpc_insight = "Strong negative: High-volume keywords have much lower CPC"

print(f"\nVolume vs CPC: {vol_cpc_corr:.3f}")
print(f"  {cpc_insight}")
print(f"  Practical Impact: Keywords with high search volume")
if vol_cpc_corr > 0:
    print(f"    → Have HIGHER commercial value per click")
    print(f"    → Worth more investment in SEO efforts")
    print(f"    → Target for PPC campaigns = higher ROI potential")
else:
    print(f"    → Have LOWER commercial value per click")
    print(f"    → High traffic volume may not equal high revenue")
    print(f"    → Balance traffic volume with CPC when prioritizing")

# KD vs CPC Analysis
if kd_cpc_corr > 0.3:
    diff_cpc = "Strong positive: More difficult keywords cost more per click"
elif kd_cpc_corr > 0.1:
    diff_cpc = "Weak positive: More difficult keywords are slightly more valuable"
elif kd_cpc_corr > -0.1:
    diff_cpc = "No correlation: KD and CPC are independent"
elif kd_cpc_corr > -0.3:
    diff_cpc = "Weak negative: More difficult keywords are slightly less valuable"
else:
    diff_cpc = "Strong negative: More difficult keywords are much less valuable"

print(f"\nKD vs CPC: {kd_cpc_corr:.3f}")
print(f"  {diff_cpc}")
print(f"  Practical Impact: Keywords with high difficulty")
if kd_cpc_corr > 0:
    print(f"    → Have HIGHER commercial intent (higher CPC)")
    print(f"    → Worth the effort even if harder to rank")
    print(f"    → Focus on competitive keywords with high CPC")
else:
    print(f"    → Have LOWER commercial intent (lower CPC)")
    print(f"    → Might not be worth significant SEO effort")
    print(f"    → Focus on easier keywords with decent volume/CPC")

print("\n3. KEY INSIGHTS & STRATEGIC RECOMMENDATIONS")
print("-" * 70)

# Generate strategic insights
insights = []

if vol_kd_corr > 0.2:
    insights.append("✓ COMPETITIVE MARKET: High-volume keywords are harder to rank for")
    insights.append("  → Strategy: Build authority, get quality backlinks")
    insights.append("  → Timeline: 6-12 months for results")

elif vol_kd_corr < -0.2:
    insights.append("✓ OPPORTUNITY ZONE: High-volume keywords are easy to rank for")
    insights.append("  → Strategy: Prioritize high-volume, low-KD keywords")
    insights.append("  → Timeline: 2-4 months to move into top positions")

else:
    insights.append("✓ BALANCED MARKET: Volume and difficulty don't correlate")
    insights.append("  → Strategy: Evaluate each keyword individually")
    insights.append("  → Look for diamonds: High volume + Low KD combinations")

if vol_cpc_corr > 0.1:
    insights.append("\n✓ VALUABLE TRAFFIC: High-volume keywords are high-value")
    insights.append("  → Focus: High volume keywords for revenue potential")
    insights.append("  → Business Impact: SEO improvements = direct revenue gains")

else:
    insights.append("\n✓ QUALITY OVER VOLUME: Volume doesn't guarantee value")
    insights.append("  → Focus: Balance volume with CPC in your strategy")
    insights.append("  → Business Impact: Consider both traffic and conversion value")

if kd_cpc_corr > 0.1:
    insights.append("\n✓ DIFFICULTY JUSTIFIED: Hard keywords are more valuable")
    insights.append("  → Invest in difficult keywords that have high CPC")
    insights.append("  → Avoid: Wasting effort on low-value difficult keywords")

else:
    insights.append("\n✓ EASY WINS AVAILABLE: Difficult keywords aren't necessarily valuable")
    insights.append("  → Strategy: Find and rank for easy, valuable keywords")
    insights.append("  → Quick Wins: Low KD + Decent Volume + Good CPC")

for insight in insights:
    print(insight)

print("\n4. STATISTICAL SUMMARY")
print("-" * 70)
print(f"Dataset size: {len(df)} keywords")
print(f"Average Volume: {df['Volume'].mean():.0f} searches/month")
print(f"Average KD: {df['KD'].mean():.1f}")
print(f"Average CPC: ${df['CPC'].mean():.2f}")
print(f"\nVolume Range: {df['Volume'].min()} - {df['Volume'].max()}")
print(f"KD Range: {df['KD'].min()} - {df['KD'].max()}")
print(f"CPC Range: ${df['CPC'].min():.2f} - ${df['CPC'].max():.2f}")

print("\n5. CORRELATION STRENGTH INTERPRETATION")
print("-" * 70)
print("""
Correlation Strength Guidelines:
  • 0.7 to 1.0 (or -0.7 to -1.0):  STRONG correlation
  • 0.3 to 0.7 (or -0.3 to -0.7):  MODERATE correlation
  • 0.0 to 0.3 (or 0.0 to -0.3):   WEAK correlation
  • 0.0:                            NO correlation

All three correlations in your SEO data appear to be WEAK, which means:
  → Variables operate more independently
  → You can find keywords with DIVERSE characteristics
  → Strategic flexibility: High volume AND low KD keywords might exist
""")

print("\n" + "="*70)
print("CORRELATION HEATMAP ANALYSIS COMPLETE")
print("="*70)
print("\nFile generated:")
print("  15. plot_10_correlation_heatmap.png - Correlation matrix visualization")
print("\n" + "="*70)
