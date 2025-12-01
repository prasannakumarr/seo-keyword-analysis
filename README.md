# SEO Keyword Analysis

Comprehensive SEO marketing performance analysis for keyword tracking and optimization.

## Overview

This project analyzes 258 SEO keywords to identify optimization opportunities and track performance metrics.

## Dataset

- **Keywords tracked:** 258
- **Total search volume:** 114,250/month
- **Metrics analyzed:** Position, Volume, Keyword Difficulty (KD), Cost Per Click (CPC)

## Key Findings

- **Average position:** 43.2 (needs improvement)
- **Top 3 positions:** 4.7% of keywords
- **Beyond position 50:** 39.9% of keywords
- **Average CPC:** $7.66

## Opportunities

- **105 low-hanging fruit keywords** - Easy difficulty, poor position
- **23 high-value underperformers** - High volume & CPC, poor ranking

## Files

- `SEO_analysis.csv` - Original dataset
- `SEO_analysis_enhanced.csv` - Enhanced dataset with performance scores
- `seo_exploratory_analysis.py` - Comprehensive analysis script
- `seo_visualizations.py` - Visualization generation script
- `seo_dashboard.png` - Main 9-panel dashboard
- `seo_insights.png` - Additional insights visualizations

## Usage

Run the analysis:
```bash
python3 seo_exploratory_analysis.py
```

Generate visualizations:
```bash
python3 seo_visualizations.py
```

## Recommendations

1. Focus on 105 low-difficulty keywords ranking poorly
2. Investigate high-volume keyword underperformance
3. Prioritize keywords with both high volume AND high CPC
4. Optimize content for keywords in positions 11-20
5. Analyze top competitors for keywords beyond position 50
