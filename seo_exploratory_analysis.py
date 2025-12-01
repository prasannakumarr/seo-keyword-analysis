import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the data
print("=" * 80)
print("SEO MARKETING PERFORMANCE - EXPLORATORY DATA ANALYSIS")
print("=" * 80)
print("\n")

df = pd.read_csv('SEO_analysis.csv')

# 1. BASIC DATA OVERVIEW
print("1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total number of keywords: {len(df)}")
print(f"Data shape: {df.shape}")
print("\n")

print("First few rows:")
print(df.head(10))
print("\n")

print("Data types:")
print(df.dtypes)
print("\n")

print("Missing values:")
print(df.isnull().sum())
print("\n")

# 2. STATISTICAL SUMMARY
print("\n2. STATISTICAL SUMMARY")
print("-" * 80)
print(df.describe())
print("\n")

# 3. CURRENT POSITION ANALYSIS
print("\n3. CURRENT POSITION ANALYSIS")
print("-" * 80)
print(f"Average position: {df['Current position'].mean():.2f}")
print(f"Median position: {df['Current position'].median():.0f}")
print(f"Best position: {df['Current position'].min()}")
print(f"Worst position: {df['Current position'].max()}")
print("\n")

# Position categories
top_3 = len(df[df['Current position'] <= 3])
top_10 = len(df[df['Current position'] <= 10])
top_20 = len(df[df['Current position'] <= 20])
beyond_50 = len(df[df['Current position'] > 50])

print("Position Distribution:")
print(f"  Top 3 positions: {top_3} keywords ({top_3/len(df)*100:.1f}%)")
print(f"  Top 10 positions: {top_10} keywords ({top_10/len(df)*100:.1f}%)")
print(f"  Top 20 positions: {top_20} keywords ({top_20/len(df)*100:.1f}%)")
print(f"  Beyond 50: {beyond_50} keywords ({beyond_50/len(df)*100:.1f}%)")
print("\n")

# 4. SEARCH VOLUME ANALYSIS
print("\n4. SEARCH VOLUME ANALYSIS")
print("-" * 80)
print(f"Total search volume: {df['Volume'].sum():,}")
print(f"Average search volume: {df['Volume'].mean():.0f}")
print(f"Median search volume: {df['Volume'].median():.0f}")
print("\n")

# High volume keywords
high_volume = df[df['Volume'] >= 1000].sort_values('Volume', ascending=False)
print(f"High-volume keywords (>= 1000 searches): {len(high_volume)}")
print("Top 10 by volume:")
print(high_volume[['Keyword', 'Volume', 'Current position', 'KD', 'CPC']].head(10))
print("\n")

# 5. KEYWORD DIFFICULTY ANALYSIS
print("\n5. KEYWORD DIFFICULTY (KD) ANALYSIS")
print("-" * 80)
print(f"Average KD: {df['KD'].mean():.1f}")
print(f"Median KD: {df['KD'].median():.0f}")
print("\n")

# KD categories
easy = len(df[df['KD'] <= 30])
medium = len(df[(df['KD'] > 30) & (df['KD'] <= 60)])
hard = len(df[df['KD'] > 60])

print("KD Distribution:")
print(f"  Easy (0-30): {easy} keywords ({easy/len(df)*100:.1f}%)")
print(f"  Medium (31-60): {medium} keywords ({medium/len(df)*100:.1f}%)")
print(f"  Hard (60+): {hard} keywords ({hard/len(df)*100:.1f}%)")
print("\n")

# 6. CPC ANALYSIS
print("\n6. COST PER CLICK (CPC) ANALYSIS")
print("-" * 80)
print(f"Average CPC: ${df['CPC'].mean():.2f}")
print(f"Median CPC: ${df['CPC'].median():.2f}")
print(f"Highest CPC: ${df['CPC'].max():.2f}")
print(f"Lowest CPC: ${df['CPC'].min():.2f}")
print("\n")

# High-value keywords (high CPC, good volume)
high_value = df[(df['CPC'] >= 10) & (df['Volume'] >= 200)].sort_values('CPC', ascending=False)
print(f"High-value keywords (CPC >= $10, Volume >= 200): {len(high_value)}")
print("Top 10 high-value keywords:")
print(high_value[['Keyword', 'Volume', 'CPC', 'Current position', 'KD']].head(10))
print("\n")

# 7. CORRELATION ANALYSIS
print("\n7. CORRELATION ANALYSIS")
print("-" * 80)
correlation_matrix = df[['Volume', 'KD', 'CPC', 'Current position']].corr()
print(correlation_matrix)
print("\n")

# Key insights from correlations
print("Key correlation insights:")
vol_pos_corr = correlation_matrix.loc['Volume', 'Current position']
kd_pos_corr = correlation_matrix.loc['KD', 'Current position']
cpc_vol_corr = correlation_matrix.loc['CPC', 'Volume']

print(f"  Volume vs Position: {vol_pos_corr:.3f}")
print(f"  KD vs Position: {kd_pos_corr:.3f}")
print(f"  CPC vs Volume: {cpc_vol_corr:.3f}")
print("\n")

# 8. OPPORTUNITY ANALYSIS
print("\n8. OPPORTUNITY ANALYSIS")
print("-" * 80)

# Low-hanging fruit: Easy KD, decent volume, poor position
low_hanging = df[(df['KD'] <= 40) & (df['Volume'] >= 200) & (df['Current position'] > 10)]
low_hanging_sorted = low_hanging.sort_values(['KD', 'Current position'])

print(f"Low-hanging fruit opportunities (Easy KD <= 40, Volume >= 200, Position > 10): {len(low_hanging)}")
print("Top 15 opportunities:")
print(low_hanging_sorted[['Keyword', 'Volume', 'KD', 'Current position', 'CPC']].head(15))
print("\n")

# High-value underperformers: Good volume, high CPC, poor position
underperformers = df[(df['Volume'] >= 500) & (df['CPC'] >= 5) & (df['Current position'] > 20)]
underperformers_sorted = underperformers.sort_values('Volume', ascending=False)

print(f"High-value underperformers (Volume >= 500, CPC >= $5, Position > 20): {len(underperformers)}")
print("Top opportunities:")
print(underperformers_sorted[['Keyword', 'Volume', 'CPC', 'Current position', 'KD']].head(10))
print("\n")

# 9. PERFORMANCE SCORE
print("\n9. PERFORMANCE SCORE CALCULATION")
print("-" * 80)

# Create a performance score: combination of position and potential value
df['Performance_Score'] = (
    (1 / df['Current position']) * 0.4 +  # Lower position is better
    (df['Volume'] / df['Volume'].max()) * 0.3 +  # Higher volume is better
    (df['CPC'] / df['CPC'].max()) * 0.2 +  # Higher CPC is better
    ((100 - df['KD']) / 100) * 0.1  # Lower KD is better
) * 100

print("Top 20 performing keywords (based on performance score):")
top_performers = df.sort_values('Performance_Score', ascending=False)
print(top_performers[['Keyword', 'Current position', 'Volume', 'CPC', 'KD', 'Performance_Score']].head(20))
print("\n")

print("Bottom 20 performing keywords:")
print(top_performers[['Keyword', 'Current position', 'Volume', 'CPC', 'KD', 'Performance_Score']].tail(20))
print("\n")

# 10. SUMMARY INSIGHTS
print("\n10. KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 80)

avg_pos = df['Current position'].mean()
total_vol = df['Volume'].sum()
avg_cpc = df['CPC'].mean()

print(f"""
OVERALL PERFORMANCE:
- Tracking {len(df)} keywords with {total_vol:,} total monthly search volume
- Average ranking position: {avg_pos:.1f} (needs improvement)
- Only {top_3/len(df)*100:.1f}% of keywords in top 3 positions
- {beyond_50/len(df)*100:.1f}% of keywords ranking beyond position 50

OPPORTUNITIES:
- {len(low_hanging)} low-hanging fruit keywords (easy wins)
- {len(underperformers)} high-value keywords severely underperforming
- Average CPC of ${avg_cpc:.2f} indicates valuable commercial intent

RECOMMENDATIONS:
1. Focus on the {len(low_hanging)} low-difficulty keywords ranking poorly
2. Investigate why high-volume keywords are underperforming
3. Prioritize keywords with both high volume AND high CPC
4. Consider content optimization for keywords in positions 11-20 (close to page 1)
5. Analyze top-performing competitors for keywords beyond position 50
""")

print("=" * 80)
print("Analysis complete! Check the visualization output next.")
print("=" * 80)

# Save the enhanced dataset
df.to_csv('SEO_analysis_enhanced.csv', index=False)
print("\nEnhanced dataset saved to: SEO_analysis_enhanced.csv")
