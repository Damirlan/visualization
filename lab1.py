import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Чтобы нормально отображались русские подписи (обычно DejaVu Sans есть по умолчанию)
plt.rcParams["font.family"] = "DejaVu Sans"

# ===== 1. Загрузка данных =====
file_path = "agg_202511011103.xlsx"
sheet_name = "agg_202511011103"

df = pd.read_excel(file_path, sheet_name=sheet_name)

# Преобразуем месяц в datetime и потом в период "год-месяц"
df["month"] = pd.to_datetime(df["month"])
df["month_period"] = df["month"].dt.to_period("M")

# ===== 2. Агрегация по месяцам =====
month_agg = (
    df.groupby("month_period")
    .agg(
        sales_qty=("sales_qty", "sum"),
        returns_qty=("returns_qty", "sum"),
        cancellations_qty=("cancellations_qty", "sum"),
        retail_amount=("retail_amount", "sum"),
        wb_commission=("wb_commission", "sum"),
        acquiring_fee=("acquiring_fee", "sum"),
        pvz_fee=("pvz_fee", "sum"),
        logistics_direct=("logistics_direct", "sum"),
        logistics_reverse=("logistics_reverse", "sum"),
        total_cost=("total_cost", "sum"),
        net_profit=("net_profit", "sum"),
    )
    .sort_index()
)

# Считаем количество заказов и коэффициенты отмен/возвратов
month_agg["orders"] = (
    month_agg["sales_qty"]
    + month_agg["returns_qty"]
    + month_agg["cancellations_qty"]
)

month_agg["cancel_rate"] = (
    month_agg["cancellations_qty"] / month_agg["orders"]
).replace([np.inf, -np.inf], np.nan)

month_agg["return_rate"] = (
    month_agg["returns_qty"] / month_agg["orders"]
).replace([np.inf, -np.inf], np.nan)

# Маржа = прибыль / выручка
month_agg["profit_margin"] = (
    month_agg["net_profit"] / month_agg["retail_amount"]
).replace([np.inf, -np.inf], np.nan)

# Структура себестоимости
cost_components = [
    "wb_commission",
    "acquiring_fee",
    "pvz_fee",
    "logistics_direct",
    "logistics_reverse",
]
month_agg["other_costs"] = (
    month_agg["total_cost"] - month_agg[cost_components].sum(axis=1)
)

# Преобразуем PeriodIndex в что-то удобное для оси X
x = month_agg.index.to_timestamp()  # для оси времени
x_labels = [p.strftime("%Y-%m") for p in month_agg.index]

# ===== 3. График: продажи и прибыль по месяцам =====
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(x, month_agg["sales_qty"], marker="o", label="Продажи, шт")
ax1.set_xlabel("Месяц")
ax1.set_ylabel("Продажи, шт")
ax1.grid(axis="y", alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(
    x,
    month_agg["net_profit"],
    marker="o",
    linestyle="--",
    label="Чистая прибыль, ₽",
)
ax2.set_ylabel("Чистая прибыль, ₽")

ax1.set_title("Динамика продаж и чистой прибыли по месяцам")

# Общая легенда для двух осей
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

fig.autofmt_xdate()

# ===== 4. График: доля отмен и возвратов =====
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    x,
    month_agg["cancel_rate"] * 100,
    marker="o",
    label="Доля отмен, %",
)
ax.plot(
    x,
    month_agg["return_rate"] * 100,
    marker="o",
    linestyle="--",
    label="Доля возвратов, %",
)

ax.set_title("Отмены и возвраты по месяцам")
ax.set_xlabel("Месяц")
ax.set_ylabel("Доля, %")
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.autofmt_xdate()

# ===== 5. График: структура себестоимости (стековый бар) =====
fig, ax = plt.subplots(figsize=(10, 6))

stack_cols = [
    "wb_commission",
    "acquiring_fee",
    "pvz_fee",
    "logistics_direct",
    "logistics_reverse",
    "other_costs",
]

bottom = np.zeros(len(month_agg))
for col in stack_cols:
    ax.bar(x_labels, month_agg[col], bottom=bottom, label=col)
    bottom += month_agg[col].values

ax.set_title("Структура себестоимости по месяцам")
ax.set_xlabel("Месяц")
ax.set_ylabel("Сумма затрат, ₽")
ax.legend(title="Компоненты затрат", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=45)
plt.tight_layout()

# ===== 6. График: связь цены и объёма продаж по товарам =====
# Возьмём, например, последние 3 месяца, чтобы не захламлять график
last_periods = sorted(df["month_period"].unique())[-3:]
mask = df["month_period"].isin(last_periods)
df_last = df[mask].copy()

fig, ax = plt.subplots(figsize=(8, 6))

for period, sub in df_last.groupby("month_period"):
    ax.scatter(
        sub["avg_ppvz_spp_prc"],
        sub["sales_qty"],
        alpha=0.7,
        label=str(period),
    )

ax.set_title("Связь средней цены и объёма продаж (последние 3 месяца)")
ax.set_xlabel("Средняя цена продажи, ₽")
ax.set_ylabel("Продажи, шт")
ax.grid(alpha=0.3)
ax.legend(title="Месяц")

# Можно сделать логарифмическую шкалу по количеству, если есть сильный размах:
# ax.set_yscale("log")

# ===== 7. График: топ-10 товаров по прибыли =====
product_agg = (
    df.groupby(["nm_id", "product_title"])
    .agg(
        sales_qty=("sales_qty", "sum"),
        net_profit=("net_profit", "sum"),
        retail_amount=("retail_amount", "sum"),
    )
    .reset_index()
)

top_products = (
    product_agg.sort_values("net_profit", ascending=False).head(10)
)

fig, ax = plt.subplots(figsize=(10, 6))
# Чтобы подписи не были слишком длинными, немного их подрежем
labels = top_products["product_title"].str.slice(0, 50) + "…"
ax.barh(labels, top_products["net_profit"])
ax.invert_yaxis()  # чтобы самый прибыльный был сверху

ax.set_title("Топ-10 товаров по чистой прибыли")
ax.set_xlabel("Чистая прибыль, ₽")
plt.tight_layout()

# ===== 8. Показать все графики =====
plt.show()
