# -*- coding: utf-8 -*-
"""
问题一：秸秆焚烧高峰期特征分析
分析黑龙江省2010-2019年秸秆焚烧的时空分布特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import seaborn as sns

# Set English display (using default fonts)
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False

# 读取数据
print("=" * 60)
print("问题一：秸秆焚烧高峰期特征分析")
print("=" * 60)

# 读取MODIS火点数据
modis_df = pd.read_csv('已数据处理/modis_processed.csv', encoding='utf-8-sig')
modis_df['acq_date'] = pd.to_datetime(modis_df['acq_date'])

print(f"\n数据总量：{len(modis_df)}条火点记录")
print(f"时间跨度：{modis_df['acq_date'].min()} 至 {modis_df['acq_date'].max()}")

# 1. 按年份统计火点数量
print("\n" + "=" * 60)
print("1. 年度火点数量统计")
print("=" * 60)

yearly_count = modis_df.groupby('year').size().reset_index(name='count')
print(yearly_count)

# 2. 按月份统计火点数量（识别季节性规律）
print("\n" + "=" * 60)
print("2. 月度火点数量统计")
print("=" * 60)

monthly_count = modis_df.groupby('month').size().reset_index(name='count')
monthly_count['percentage'] = (monthly_count['count'] / len(modis_df) * 100).round(2)
print(monthly_count)

# 找出焚烧高峰月份
peak_months = monthly_count.nlargest(3, 'count')
print("\n焚烧高峰月份TOP3：")
print(peak_months)

# 3. 按旬统计（上旬、中旬、下旬）
print("\n" + "=" * 60)
print("3. 旬度统计分析")
print("=" * 60)

ten_day_count = modis_df.groupby(['month', 'ten_day']).size().reset_index(name='count')
ten_day_count = ten_day_count.sort_values(['month', 'count'], ascending=[True, False])
# 只输出简要信息
print(f"共有{len(ten_day_count)}个月-旬组合")
print(f"最高单旬火点数：{ten_day_count['count'].max()}")
top5_ten_day = ten_day_count.nlargest(5, 'count')
print("火点数最多的5个旬：")

# 4. FRP（火辐射功率）统计分析
print("\n" + "=" * 60)
print("4. 火辐射功率（FRP）统计")
print("=" * 60)

frp_stats = modis_df.groupby('year')['frp'].agg(['mean', 'std', 'min', 'max', 'median'])
print(frp_stats)

# 5. 按月份分析FRP平均值
monthly_frp = modis_df.groupby('month')['frp'].mean().reset_index()
monthly_frp.columns = ['month', 'avg_frp']
print("\n月度平均FRP：")
print(monthly_frp)

# ============== 可视化分析 ==============

# Figure 1: Annual Fire Spot Trend
plt.figure(figsize=(12, 6))
plt.plot(yearly_count['year'], yearly_count['count'], marker='o', linewidth=2, markersize=8, color='#d62728')
plt.fill_between(yearly_count['year'], yearly_count['count'], alpha=0.3, color='#d62728')
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Number of Fire Spots', fontsize=14, fontweight='bold')
plt.title('Annual Variation of Straw Burning Fire Spots in Heilongjiang (2010-2019)', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(yearly_count['year'], fontsize=12)
plt.yticks(fontsize=12)
for i, row in yearly_count.iterrows():
    plt.text(row['year'], row['count']+500, str(row['count']),
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('问题一/年度火点数量变化趋势.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n已保存：年度火点数量变化趋势.png")

# Figure 2: Monthly Fire Spot Distribution
plt.figure(figsize=(12, 6))
colors = ['#1f77b4' if x not in [3, 4, 10, 11] else '#ff7f0e' for x in monthly_count['month']]
bars = plt.bar(monthly_count['month'], monthly_count['count'], color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Month', fontsize=14, fontweight='bold')
plt.ylabel('Number of Fire Spots', fontsize=14, fontweight='bold')
plt.title('Monthly Distribution of Straw Burning in Heilongjiang (2010-2019)', fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(1, 13), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3, axis='y', linestyle='--')

# Add value labels
for i, (bar, row) in enumerate(zip(bars, monthly_count.itertuples())):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height+500,
             f'{int(height)}\n({row.percentage}%)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#ff7f0e', edgecolor='black', label='Peak Seasons (Spring & Autumn)'),
                   Patch(facecolor='#1f77b4', edgecolor='black', label='Non-peak Period')]
plt.legend(handles=legend_elements, loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('问题一/月度火点数量分布.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存：月度火点数量分布.png")

# Figure 3: Temporal Distribution Heatmap
pivot_data = modis_df.groupby(['year', 'month']).size().reset_index(name='count')
heatmap_data = pivot_data.pivot(index='month', columns='year', values='count')

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='g',
            cbar_kws={'label': 'Number of Fire Spots'}, linewidths=0.5, linecolor='gray')
plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Month', fontsize=14, fontweight='bold')
plt.title('Temporal Distribution Heatmap of Straw Burning in Heilongjiang (2010-2019)', fontsize=16, fontweight='bold', pad=20)
plt.yticks(rotation=0, fontsize=11)
plt.xticks(fontsize=11)
plt.tight_layout()
plt.savefig('问题一/时间分布热力图.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存：时间分布热力图.png")

# ============== 输出统计结果 ==============
print("\n" + "=" * 60)
print("关键发现总结")
print("=" * 60)

print(f"\n1. 焚烧高峰期特征：")
print(f"   - 主要集中在春季（3-5月）和秋季（9-11月）")
print(f"   - 春季火点数：{monthly_count[monthly_count['month'].isin([3,4,5])]['count'].sum()}")
print(f"   - 秋季火点数：{monthly_count[monthly_count['month'].isin([9,10,11])]['count'].sum()}")
print(f"   - 两季合计占比：{monthly_count[monthly_count['month'].isin([3,4,5,9,10,11])]['percentage'].sum():.2f}%")

print(f"\n2. 年度变化趋势：")
print(f"   - 最高年份：{yearly_count.loc[yearly_count['count'].idxmax(), 'year']}")
print(f"   - 最高火点数：{yearly_count['count'].max()}")
print(f"   - 最低年份：{yearly_count.loc[yearly_count['count'].idxmin(), 'year']}")
print(f"   - 最低火点数：{yearly_count['count'].min()}")

print(f"\n3. 火辐射功率特征：")
print(f"   - 平均FRP：{modis_df['frp'].mean():.2f} MW")
print(f"   - 最大FRP：{modis_df['frp'].max():.2f} MW")
print(f"   - 中位数FRP：{modis_df['frp'].median():.2f} MW")

spring_frp = modis_df[modis_df['month'].isin([3,4,5])]['frp'].mean()
autumn_frp = modis_df[modis_df['month'].isin([9,10,11])]['frp'].mean()
print(f"   - 春季平均FRP：{spring_frp:.2f} MW")
print(f"   - 秋季平均FRP：{autumn_frp:.2f} MW")

print("\n问题一分析完成！")
print("=" * 60)
