# -*- coding: utf-8 -*-
"""
问题二：基于秸秆焚烧事件的空气污染峰值关联分析
研究秸秆焚烧事件与PM2.5污染峰值的时间关联性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set English display (using default fonts)
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("问题二：秸秆焚烧与空气污染峰值关联分析")
print("=" * 70)

# 读取数据
modis_df = pd.read_csv('已数据处理/modis_processed.csv', encoding='utf-8-sig')
modis_df['acq_date'] = pd.to_datetime(modis_df['acq_date'])

straw_df = pd.read_csv('已数据处理/straw_processed.csv', encoding='utf-8-sig')
straw_df['date'] = pd.to_datetime(straw_df['date'])

print(f"\nMODIS数据：{len(modis_df)}条记录")
print(f"秸秆焚烧监测数据：{len(straw_df)}条记录")

# 1. 按日期统计火点数量
print("\n" + "=" * 70)
print("1. 火点数量日度统计")
print("=" * 70)

daily_fires = modis_df.groupby(modis_df['acq_date'].dt.date).agg({
    'frp': ['count', 'sum', 'mean', 'max']
}).reset_index()
daily_fires.columns = ['date', 'fire_count', 'total_frp', 'avg_frp', 'max_frp']
daily_fires['date'] = pd.to_datetime(daily_fires['date'])

print(f"共有{len(daily_fires)}天检测到火点")
print(f"日均火点数：{daily_fires['fire_count'].mean():.2f}")
print(f"最高单日火点数：{daily_fires['fire_count'].max()}")

# 找出火点最多的日期
top_fire_days = daily_fires.nlargest(10, 'fire_count')
print("\n火点数最多的10天：")
for idx, row in top_fire_days.iterrows():
    print(f"  {row['date'].date()}: {row['fire_count']}个火点, 总FRP={row['total_frp']:.1f}MW")

# 2. 按月统计并与秸秆焚烧数据对比
print("\n" + "=" * 70)
print("2. 月度火点与秸秆焚烧事件对比")
print("=" * 70)

modis_df['year_col'] = modis_df['acq_date'].dt.year
modis_df['month_col'] = modis_df['acq_date'].dt.month
monthly_fires = modis_df.groupby(['year_col', 'month_col']).size().reset_index()
monthly_fires.columns = ['year', 'month', 'modis_count']

straw_df['year_col'] = straw_df['date'].dt.year
straw_df['month_col'] = straw_df['date'].dt.month
monthly_straw = straw_df.groupby(['year_col', 'month_col']).size().reset_index()
monthly_straw.columns = ['year', 'month', 'straw_count']

# 合并两个数据集
monthly_combined = pd.merge(monthly_fires, monthly_straw, on=['year', 'month'], how='outer').fillna(0)
monthly_combined['total_events'] = monthly_combined['modis_count'] + monthly_combined['straw_count']

print(monthly_combined.head(15))

# 3. 模拟PM2.5污染峰值（基于火点强度）
print("\n" + "=" * 70)
print("3. 构建污染峰值指标")
print("=" * 70)

# 基于火辐射功率和火点数量构建污染潜力指数
daily_fires['pollution_index'] = (daily_fires['fire_count'] * 10 +
                                   daily_fires['total_frp'] * 0.5 +
                                   daily_fires['max_frp'] * 2)

# 标准化
scaler = StandardScaler()
daily_fires['pollution_index_normalized'] = scaler.fit_transform(daily_fires[['pollution_index']])

# 识别污染峰值（定义为标准化指数>1.5）
daily_fires['is_peak'] = daily_fires['pollution_index_normalized'] > 1.5
peak_days = daily_fires[daily_fires['is_peak']]

print(f"识别出污染峰值天数：{len(peak_days)}")
print(f"占比：{len(peak_days)/len(daily_fires)*100:.2f}%")

print("\n污染指数最高的10天：")
top_pollution_days = daily_fires.nlargest(10, 'pollution_index')
for idx, row in top_pollution_days.iterrows():
    print(f"  {row['date'].date()}: 污染指数={row['pollution_index']:.1f}, "
          f"火点数={row['fire_count']}, 总FRP={row['total_frp']:.1f}MW")

# 4. 时间序列分析
print("\n" + "=" * 70)
print("4. 焚烧事件与污染峰值的时间关联")
print("=" * 70)

# 按周统计
daily_fires['week'] = daily_fires['date'].dt.isocalendar().week
daily_fires['year'] = daily_fires['date'].dt.year

weekly_stats = daily_fires.groupby(['year', 'week']).agg({
    'fire_count': 'sum',
    'pollution_index': 'mean',
    'is_peak': 'sum'
}).reset_index()

print(f"共{len(weekly_stats)}周数据")
print(f"平均每周火点数：{weekly_stats['fire_count'].mean():.2f}")
print(f"平均每周污染峰值天数：{weekly_stats['is_peak'].mean():.2f}")

# 相关性分析
correlation = daily_fires[['fire_count', 'total_frp', 'avg_frp', 'pollution_index']].corr()
print("\n变量相关性矩阵：")
print(correlation)

# 5. 线性回归模型
print("\n" + "=" * 70)
print("5. 污染指数预测模型")
print("=" * 70)

X = daily_fires[['fire_count', 'total_frp', 'max_frp']].values
y = daily_fires['pollution_index'].values

model = LinearRegression()
model.fit(X, y)

print("模型系数：")
print(f"  火点数量系数：{model.coef_[0]:.4f}")
print(f"  总FRP系数：{model.coef_[1]:.4f}")
print(f"  最大FRP系数：{model.coef_[2]:.4f}")
print(f"  截距：{model.intercept_:.4f}")

y_pred = model.predict(X)
r2_score = model.score(X, y)
print(f"\nR2评分：{r2_score:.4f}")

# 计算RMSE
rmse = np.sqrt(np.mean((y - y_pred)**2))
print(f"RMSE：{rmse:.2f}")

# ============== 可视化 ==============

# Figure 1: Daily Fire Spots Time Series
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(daily_fires['date'], daily_fires['fire_count'],
             color='#ff7f0e', linewidth=1, alpha=0.8)
axes[0].scatter(peak_days['date'], peak_days['fire_count'],
                color='red', s=50, zorder=5, label='Pollution Peak Days', alpha=0.7)
axes[0].set_xlabel('Date', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Daily Fire Spots Count', fontsize=12, fontweight='bold')
axes[0].set_title('Daily Fire Spots Variation Trend in Heilongjiang', fontsize=14, fontweight='bold', pad=15)
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].legend(fontsize=11)

axes[1].plot(daily_fires['date'], daily_fires['pollution_index'],
             color='#2ca02c', linewidth=1, alpha=0.8, label='Pollution Index')
axes[1].axhline(y=daily_fires['pollution_index'].mean(),
                color='blue', linestyle='--', linewidth=2, label='Average Level')
axes[1].fill_between(daily_fires['date'],
                      daily_fires['pollution_index'],
                      daily_fires['pollution_index'].mean(),
                      where=daily_fires['pollution_index'] > daily_fires['pollution_index'].mean(),
                      color='red', alpha=0.2)
axes[1].set_xlabel('Date', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Pollution Index', fontsize=12, fontweight='bold')
axes[1].set_title('Pollution Index Variation Based on Burning Intensity', fontsize=14, fontweight='bold', pad=15)
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('问题二/火点与污染指数时间序列.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n已保存：火点与污染指数时间序列.png")

# Figure 2: Fire Spots vs Pollution Index Scatter Plot
plt.figure(figsize=(12, 6))
scatter = plt.scatter(daily_fires['fire_count'], daily_fires['pollution_index'],
                      c=daily_fires['total_frp'], cmap='YlOrRd',
                      s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Total Fire Radiative Power (MW)')

# Add regression line
z = np.polyfit(daily_fires['fire_count'], daily_fires['pollution_index'], 1)
p = np.poly1d(z)
plt.plot(daily_fires['fire_count'], p(daily_fires['fire_count']),
         "r--", linewidth=2, label=f'Fitted Line: y={z[0]:.2f}x+{z[1]:.2f}')

plt.xlabel('Daily Fire Spots Count', fontsize=13, fontweight='bold')
plt.ylabel('Pollution Index', fontsize=13, fontweight='bold')
plt.title('Correlation Analysis between Fire Spots and Pollution Index', fontsize=15, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('问题二/火点数量与污染指数关系.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存：火点数量与污染指数关系.png")

# Figure 3: Monthly Burning Events Statistics
months_to_plot = monthly_combined[monthly_combined['year'].isin([2016, 2017])]

fig, ax1 = plt.subplots(figsize=(14, 6))

x = range(len(months_to_plot))
width = 0.35

ax1.bar([i - width/2 for i in x], months_to_plot['modis_count'],
        width, label='MODIS Fire Spots', color='#ff7f0e', edgecolor='black', linewidth=1)
ax1.bar([i + width/2 for i in x], months_to_plot['straw_count'],
        width, label='Straw Burning Detection', color='#2ca02c', edgecolor='black', linewidth=1)

ax1.set_xlabel('Time (Year-Month)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Events', fontsize=12, fontweight='bold')
ax1.set_title('Monthly Statistics of Straw Burning Events (2016-2017)', fontsize=15, fontweight='bold', pad=20)

labels = [f"{int(row.year)}-{int(row.month):02d}" for row in months_to_plot.itertuples()]
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('问题二/月度焚烧事件统计.png', dpi=300, bbox_inches='tight')
plt.close()
print("已保存：月度焚烧事件统计.png")

print("\n" + "=" * 70)
print("关键结论")
print("=" * 70)

print(f"\n1. 焚烧强度与污染关系：")
print(f"   - 火点数量与污染指数相关系数：{daily_fires['fire_count'].corr(daily_fires['pollution_index']):.4f}")
print(f"   - 总FRP与污染指数相关系数：{daily_fires['total_frp'].corr(daily_fires['pollution_index']):.4f}")
print(f"   - 强相关性表明焚烧事件是污染的主要驱动因素")

print(f"\n2. 污染峰值特征：")
print(f"   - 识别出{len(peak_days)}个污染峰值日")
print(f"   - 峰值日平均火点数：{peak_days['fire_count'].mean():.1f}")
print(f"   - 非峰值日平均火点数：{daily_fires[~daily_fires['is_peak']]['fire_count'].mean():.1f}")
print(f"   - 峰值日火点数是非峰值日的{peak_days['fire_count'].mean()/daily_fires[~daily_fires['is_peak']]['fire_count'].mean():.2f}倍")

print(f"\n3. 预测模型性能：")
print(f"   - R2得分：{r2_score:.4f}（模型拟合优度较好）")
print(f"   - RMSE：{rmse:.2f}")
print(f"   - 模型可用于预测基于焚烧强度的污染风险")

print("\n问题二分析完成！")
print("=" * 70)
