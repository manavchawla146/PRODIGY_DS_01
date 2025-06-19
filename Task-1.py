import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style for charts
plt.style.use('ggplot')

# File paths
labor_file = "API_SL.TLF.TOTL.IN_DS2_en_csv_v2_73823.csv"
metadata_file = "Metadata_Country_API_SL.TLF.TOTL.IN_DS2_en_csv_v2_73823.csv"
output_dir = Path("charts")
output_dir.mkdir(exist_ok=True)

# Load data
labor_df = pd.read_csv(labor_file, skiprows=4)
metadata_df = pd.read_csv(metadata_file)

# Clean labor force data
labor_df.drop(columns=['Indicator Name', 'Indicator Code'], inplace=True, errors='ignore')
year_cols = [str(y) for y in range(1960, 2025)]
labor_df[year_cols] = labor_df[year_cols].apply(pd.to_numeric, errors='coerce')
labor_df.dropna(subset=year_cols, how='all', inplace=True)
labor_df[year_cols] = labor_df[year_cols].fillna(0)

# Clean metadata
metadata_df = metadata_df[['Country Code', 'Region', 'IncomeGroup', 'TableName']]
metadata_df.fillna({'Region': 'Unknown', 'IncomeGroup': 'Unknown'}, inplace=True)

# Merge
df = pd.merge(labor_df, metadata_df, on='Country Code', how='inner')

# Filter for 1990–2024
year_cols_filtered = [str(y) for y in range(1990, 2025)]
df_filtered = df[['Country Name', 'Country Code', 'Region', 'IncomeGroup', 'TableName'] + year_cols_filtered]

# Define decades
decades = {
    '1990s': [str(y) for y in range(1990, 2000)],
    '2000s': [str(y) for y in range(2000, 2010)],
    '2010s': [str(y) for y in range(2010, 2020)],
    '2020s': [str(y) for y in range(2020, 2025)],
}

# --- Categorical Charts ---

# 1. Bar Chart: Average Labor Force by Region
region_avg = df_filtered[df_filtered['Region'] != 'Unknown'].groupby('Region')[year_cols_filtered].mean().mean(axis=1).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(region_avg.index, region_avg / 1e6)
plt.title('Average Labor Force by Region (1990–2024)')
plt.xlabel('Region')
plt.ylabel('Average Labor Force (Millions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'bar_avg_labor_region.png')
plt.close()

# 2. Bar Chart: Top 10 Countries in 2024
top_10_2024 = df_filtered[['TableName', '2024']].sort_values('2024', ascending=False).head(10)
plt.figure(figsize=(12, 6))
plt.barh(top_10_2024['TableName'], top_10_2024['2024'] / 1e6)
plt.title('Top 10 Countries by Labor Force in 2024')
plt.xlabel('Labor Force (Millions)')
plt.tight_layout()
plt.savefig(output_dir / 'bar_top_10_countries_2024.png')
plt.close()

# 3. Radar Chart by Region & Decade
region_decade_data = {}
for decade, years in decades.items():
    region_sum = df_filtered[df_filtered['Region'] != 'Unknown'].groupby('Region')[years].sum().mean(axis=1)
    region_decade_data[decade] = region_sum

regions = list(region_decade_data['1990s'].index)
angles = np.linspace(0, 2 * np.pi, len(regions), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
for decade, data in region_decade_data.items():
    values = (data / 1e6).tolist() + [(data / 1e6).tolist()[0]]
    ax.plot(angles, values, label=decade)
    ax.fill(angles, values, alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(regions)
plt.title('Labor Force Across Regions by Decade')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.tight_layout()
plt.savefig(output_dir / 'radar_labor_regions_decades.png')
plt.close()

# 4. Stacked Bar Chart by Income Group & Decade
income_decade_data = {}
for decade, years in decades.items():
    income_sum = df_filtered[df_filtered['IncomeGroup'] != 'Unknown'].groupby('IncomeGroup')[years].sum().mean(axis=1)
    income_decade_data[decade] = income_sum

income_decade_df = pd.DataFrame(income_decade_data).T
plt.figure(figsize=(10, 6))
bottom = np.zeros(len(income_decade_df))
for income_group in income_decade_df.columns:
    plt.bar(income_decade_df.index, income_decade_df[income_group] / 1e6, bottom=bottom, label=income_group)
    bottom += income_decade_df[income_group] / 1e6
plt.title('Labor Force by Income Group Over Decades')
plt.xlabel('Decade')
plt.ylabel('Labor Force (Millions)')
plt.legend(title='Income Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_dir / 'stacked_bar_income_decades.png')
plt.close()

# 5. Pie Charts: Income Group Share in 1990 vs 2024
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for year, ax in zip(['1990', '2024'], [ax1, ax2]):
    income_sum = df_filtered[df_filtered['IncomeGroup'] != 'Unknown'].groupby('IncomeGroup')[year].sum()
    ax.pie(income_sum / 1e6, labels=income_sum.index, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Labor Force by Income Group ({year})')
fig.suptitle('Labor Force Contribution by Income Group: 1990 vs. 2024')
plt.tight_layout()
plt.savefig(output_dir / 'pie_income_1990_2024.png')
plt.close()

# --- Continuous Charts ---

# 6. Line Chart: China and India
china_india = df_filtered[df_filtered['Country Code'].isin(['CHN', 'IND'])]
plt.figure(figsize=(12, 6))
for _, row in china_india.iterrows():
    plt.plot(year_cols_filtered, row[year_cols_filtered] / 1e6, label=row['TableName'])
plt.title('Labor Force Growth: China & India (1990–2024)')
plt.xlabel('Year')
plt.ylabel('Labor Force (Millions)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'line_china_india.png')
plt.close()

# 7. Area Chart: High Income Trend
high_income = df_filtered[df_filtered['IncomeGroup'] == 'High income']
high_sum = high_income[year_cols_filtered].sum()
plt.figure(figsize=(12, 6))
plt.fill_between(year_cols_filtered, high_sum / 1e6, alpha=0.4)
plt.plot(year_cols_filtered, high_sum / 1e6, marker='o')
plt.title('Labor Force Trends: High Income (1990–2024)')
plt.xlabel('Year')
plt.ylabel('Labor Force (Millions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'area_high_income.png')
plt.close()

# 8. Scatter Chart: 1990 vs 2024
scatter_data = df_filtered[['TableName', '1990', '2024']].dropna()
plt.figure(figsize=(10, 6))
plt.scatter(scatter_data['1990'] / 1e6, scatter_data['2024'] / 1e6)
plt.title('Labor Force 2024 vs. 1990')
plt.xlabel('1990 (Millions)')
plt.ylabel('2024 (Millions)')
plt.tight_layout()
plt.savefig(output_dir / 'scatter_1990_vs_2024.png')
plt.close()

# 9. Line Chart: Low Income Countries
low_income = df_filtered[df_filtered['IncomeGroup'] == 'Low income']
low_sum = low_income[year_cols_filtered].sum()
plt.figure(figsize=(12, 6))
plt.plot(year_cols_filtered, low_sum / 1e6, marker='o')
plt.title('Labor Force Growth: Low Income Countries (1990–2024)')
plt.xlabel('Year')
plt.ylabel('Labor Force (Millions)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'line_low_income.png')
plt.close()

print("✅ All cleaned, processed, and visualized. Charts saved in the 'charts' folder.")
 