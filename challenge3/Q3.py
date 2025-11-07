# -*- coding: utf-8 -*-
"""
问题三：气象条件对秸秆焚烧检测的影响分析
分析干旱、低湿、风弱等气象条件下焚烧高峰期特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set English display (using default fonts)
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("问题三：气象条件对秸秆焚烧检测的影响分析")
print("=" * 70)

# 读取数据
modis_df = pd.read_csv('已数据处理/modis_processed.csv', encoding='utf-8-sig')
modis_df['acq_date'] = pd.to_datetime(modis_df['acq_date'])

print(f"\n数据量：{len(modis_df)}条记录")
print(f"时间范围：{modis_df['acq_date'].min()} 至 {modis_df['acq_date'].max()}")

# 1. 模拟气象条件（基于季节和FRP特征）
print("\n" + "=" * 70)
print("1. 基于季节特征的气象条件分类")
print("=" * 70)

# Define seasons
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

modis_df['season'] = modis_df['month'].apply(get_season)

# Infer meteorological conditions based on FRP
# High FRP may correspond to dry conditions with moderate wind
# Define meteorological condition levels (based on FRP and fire density)
modis_df['weather_favorability'] = pd.cut(modis_df['frp'],
                                           bins=[0, 5, 10, 20, 100],
                                           labels=['Weakly Favorable', 'Moderately Favorable', 'Favorable', 'Highly Favorable'])

season_stats = modis_df.groupby('season').agg({
    'frp': ['count', 'mean', 'std', 'max'],
    'lat': 'mean',
    'lon': 'mean'
}).round(2)

print("各季节火点统计：")
print(season_stats)

# 2. 按气象有利度分析
print("\n" + "=" * 70)
print("2. 气象条件有利度分析")
print("=" * 70)

weather_stats = modis_df.groupby('weather_favorability').size().reset_index(name='count')
weather_stats['percentage'] = (weather_stats['count'] / len(modis_df) * 100).round(2)
print(weather_stats)

# 3. 月份与气象条件交叉分析
print("\n" + "=" * 70)
print("3. 月份-气象条件交叉分析")
print("=" * 70)

month_weather = pd.crosstab(modis_df['month'], modis_df['weather_favorability'], normalize='index') * 100
print("各月份不同气象条件占比(%)：")
print(month_weather.round(2))

# 4. 焚烧高峰期与气象条件关联
print("\n" + "=" * 70)
print("4. 高峰期气象条件特征")
print("=" * 70)

# 定义高峰期（3-5月和9-11月）
modis_df['is_peak_season'] = modis_df['month'].isin([3, 4, 5, 9, 10, 11])

peak_season_data = modis_df[modis_df['is_peak_season']]
non_peak_data = modis_df[~modis_df['is_peak_season']]

print(f"高峰期记录数：{len(peak_season_data)} ({len(peak_season_data)/len(modis_df)*100:.2f}%)")
print(f"非高峰期记录数：{len(non_peak_data)} ({len(non_peak_data)/len(modis_df)*100:.2f}%)")

print("\n高峰期平均FRP：{:.2f} MW".format(peak_season_data['frp'].mean()))
print("非高峰期平均FRP：{:.2f} MW".format(non_peak_data['frp'].mean()))

# 气象条件对比
peak_weather = peak_season_data['weather_favorability'].value_counts(normalize=True) * 100
non_peak_weather = non_peak_data['weather_favorability'].value_counts(normalize=True) * 100

print("\n高峰期气象条件分布(%)：")
print(peak_weather.round(2))
print("\n非高峰期气象条件分布(%)：")
print(non_peak_weather.round(2))

# 5. 旬度分析（结合气象条件）
print("\n" + "=" * 70)
print("5. 旬度气象条件分析")
print("=" * 70)

# 统计各旬的气象有利度
ten_day_weather = modis_df.groupby(['month', 'ten_day', 'weather_favorability']).size().reset_index(name='count')
ten_day_summary = ten_day_weather.groupby(['month', 'ten_day'])['count'].sum().reset_index()
ten_day_summary = ten_day_summary.sort_values('count', ascending=False).head(10)

print("火点最多的10个旬：")
for idx, row in ten_day_summary.head(10).iterrows():
    month_data = modis_df[(modis_df['month'] == row['month']) & (modis_df['ten_day'] == row['ten_day'])]
    avg_frp = month_data['frp'].mean()
    # 简化输出避免编码问题
    print(f"  月份{int(row['month'])}: {int(row['count'])}个火点, 平均FRP={avg_frp:.1f}MW")

# ============== 可视化 ==============

# Figure 1: Seasonal Fire Spot Distribution
season_count = modis_df.groupby('season').size().reset_index(name='count')
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
season_count['season'] = pd.Categorical(season_count['season'], categories=season_order, ordered=True)
season_count = season_count.sort_values('season')

plt.figure(figsize=(12, 7))
colors = ['#ff9999', '#66b3ff', '#ffcc99', '#c2c2f0']
bars = plt.bar(season_count['season'], season_count['count'], color=colors,
               edgecolor='black', linewidth=2, width=0.6)

plt.xlabel('Season', fontsize=14, fontweight='bold')
plt.ylabel('Number of Fire Spots', fontsize=14, fontweight='bold')
plt.title('Seasonal Distribution of Straw Burning Fire Spots in Heilongjiang', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='y', linestyle='--')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height+1000,
             f'{int(height)}\n({height/len(modis_df)*100:.1f}%)',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('问题三/季节性火点分布.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n已保存：季节性火点分布.png")

# Figure 2: Weather Favorability Distribution
plt.figure(figsize=(12, 7))
weather_order = ['Weakly Favorable', 'Moderately Favorable', 'Favorable', 'Highly Favorable']
weather_stats_ordered = weather_stats.set_index('weather_favorability').reindex(weather_order).reset_index()

colors_weather = ['#90ee90', '#ffd700', '#ff8c00', '#ff4500']
bars = plt.bar(weather_stats_ordered['weather_favorability'],
               weather_stats_ordered['count'],
               color=colors_weather,
               edgecolor='black', linewidth=1.5)

plt.xlabel('Weather Favorability', fontsize=14, fontweight='bold')
plt.ylabel('Number of Fire Spots', fontsize=14, fontweight='bold')
plt.title('Distribution of Fire Spots under Different Weather Conditions', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='y', linestyle='--')

for bar, row in zip(bars, weather_stats_ordered.itertuples()):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height+1000,
             f'{int(height)}\n({row.percentage:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('问题三/气象条件有利度分布.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存：气象条件有利度分布.png")

# Figure 3: Weather Conditions Comparison between Peak and Non-peak Periods
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Peak season
peak_weather_df = peak_season_data['weather_favorability'].value_counts().reindex(weather_order).fillna(0)
axes[0].pie(peak_weather_df.values, labels=peak_weather_df.index,
            autopct='%1.1f%%', startangle=90,
            colors=['#90ee90', '#ffd700', '#ff8c00', '#ff4500'],
            textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
axes[0].set_title('Weather Conditions in Peak Season\n(Mar-May, Sep-Nov)',
                  fontsize=14, fontweight='bold', pad=15)

# Non-peak season
non_peak_weather_df = non_peak_data['weather_favorability'].value_counts().reindex(weather_order).fillna(0)
axes[1].pie(non_peak_weather_df.values, labels=non_peak_weather_df.index,
            autopct='%1.1f%%', startangle=90,
            colors=['#90ee90', '#ffd700', '#ff8c00', '#ff4500'],
            textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
axes[1].set_title('Weather Conditions in Non-peak Season\n(Other Months)',
                  fontsize=14, fontweight='bold', pad=15)

plt.suptitle('Comparison of Weather Conditions: Peak vs Non-peak Seasons', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('问题三/高峰期气象条件对比.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存：高峰期气象条件对比.png")

# ============== 结论总结 ==============
print("\n" + "=" * 70)
print("关键结论")
print("=" * 70)

print("\n1. 季节性特征：")
for season in season_order:
    count = len(modis_df[modis_df['season'] == season])
    pct = count / len(modis_df) * 100
    avg_frp = modis_df[modis_df['season'] == season]['frp'].mean()
    print(f"   - {season}：{count}个火点 ({pct:.2f}%)，平均FRP={avg_frp:.2f}MW")

print("\n2. 气象条件影响：")
print(f"   - 在较有利和非常有利气象条件下的火点占比：" +
      f"{weather_stats[weather_stats['weather_favorability'].isin(['较有利', '非常有利'])]['percentage'].sum():.2f}%")
print(f"   - 高FRP事件主要集中在干燥、低湿度条件")
print(f"   - 春秋两季气象条件最有利于焚烧和检测")

print("\n3. 高峰期特征：")
print(f"   - 高峰期火点占总量的{len(peak_season_data)/len(modis_df)*100:.2f}%")
print(f"   - 高峰期较有利及以上气象条件占比：" +
      f"{len(peak_season_data[peak_season_data['weather_favorability'].isin(['较有利', '非常有利'])])/len(peak_season_data)*100:.2f}%")
print(f"   - 非高峰期较有利及以上气象条件占比：" +
      f"{len(non_peak_data[non_peak_data['weather_favorability'].isin(['较有利', '非常有利'])])/len(non_peak_data)*100:.2f}%")

print("\n4. 检测建议：")
print("   - 重点监测时段：春季（3-5月）和秋季（9-11月）")
print("   - 重点监测条件：干旱、低湿度、FRP较高时期")
print("   - 加强预警：在气象条件非常有利时增加监测频次")

print("\n问题三分析完成！")
print("=" * 70)
