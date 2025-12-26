import sys
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QLabel,
    QComboBox,
    QSlider,
    QSpinBox,
    QPushButton,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, нужно для 3D

import matplotlib.pyplot as plt

# Настройка шрифта для русских подписей
plt.rcParams["font.family"] = "DejaVu Sans"

FILE_PATH = "agg_202511011103.xlsx"
SHEET_NAME = "agg_202511011103"


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Аналитика Wildberries (пример с Qt + matplotlib)")
        self.resize(1400, 900)

        # Таймер для анимации
        self.timer = QTimer(self)
        self.timer.setInterval(800)
        self.timer.timeout.connect(self.update_animation)
        self.animation_running = False
        self.animation_step = 0

        # Загрузим и подготовим данные
        self.load_data()

        # UI
        self.init_ui()

    def load_data(self):
        """Загрузка и агрегация данных."""
        self.df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

        # month как период (год-месяц)
        self.df["month"] = pd.to_datetime(self.df["month"])
        self.df["month_period"] = self.df["month"].dt.to_period("M")

        # Агрегация по месяцам
        self.month_agg = (
            self.df.groupby("month_period")
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

        self.month_agg["orders"] = (
            self.month_agg["sales_qty"]
            + self.month_agg["returns_qty"]
            + self.month_agg["cancellations_qty"]
        )

        self.month_agg["cancel_rate"] = (
            self.month_agg["cancellations_qty"] / self.month_agg["orders"]
        ).replace([np.inf, -np.inf], np.nan)

        self.month_agg["return_rate"] = (
            self.month_agg["returns_qty"] / self.month_agg["orders"]
        ).replace([np.inf, -np.inf], np.nan)

        self.month_agg["profit_margin"] = (
            self.month_agg["net_profit"] / self.month_agg["retail_amount"]
        ).replace([np.inf, -np.inf], np.nan)

        # Структура затрат
        cost_components = [
            "wb_commission",
            "acquiring_fee",
            "pvz_fee",
            "logistics_direct",
            "logistics_reverse",
        ]
        self.cost_components = cost_components
        self.month_agg["other_costs"] = (
            self.month_agg["total_cost"] - self.month_agg[cost_components].sum(axis=1)
        )

        # Агрегация по товарам
        self.product_agg = (
            self.df.groupby(["nm_id", "product_title"])
            .agg(
                sales_qty=("sales_qty", "sum"),
                net_profit=("net_profit", "sum"),
                retail_amount=("retail_amount", "sum"),
                avg_price=("avg_ppvz_spp_prc", "mean"),
            )
            .reset_index()
        )

        # Список месяцев для комбобоксов
        self.month_list = list(self.month_agg.index)
        self.month_list_str = [p.strftime("%Y-%m") for p in self.month_list]

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # вкладки
        self.tabs.addTab(self.create_overview_tab(), "Обзор")
        self.tabs.addTab(self.create_cancellations_tab(), "Отмены/возвраты")
        self.tabs.addTab(self.create_costs_tab(), "Затраты")
        self.tabs.addTab(self.create_scatter_tab(), "Scatter (цена/продажи)")
        self.tabs.addTab(self.create_top_products_tab(), "Топ товаров")
        self.tabs.addTab(self.create_pie_tab(), "Круговые диаграммы")
        self.tabs.addTab(self.create_3d_anim_tab(), "3D + анимация")

    # ---------------- ОТДЕЛЬНЫЕ ВКЛАДКИ ----------------

    def create_overview_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        layout.addLayout(controls)

        self.chk_sales = QCheckBox("Показывать продажи, шт")
        self.chk_sales.setChecked(True)
        self.chk_profit = QCheckBox("Показывать чистую прибыль, ₽")
        self.chk_profit.setChecked(True)

        self.chk_sales.stateChanged.connect(self.update_overview_plot)
        self.chk_profit.stateChanged.connect(self.update_overview_plot)

        controls.addWidget(self.chk_sales)
        controls.addWidget(self.chk_profit)
        controls.addStretch()

        self.overview_canvas = MplCanvas(tab)
        self.overview_toolbar = NavigationToolbar(self.overview_canvas, tab)

        layout.addWidget(self.overview_toolbar)
        layout.addWidget(self.overview_canvas)

        self.update_overview_plot()

        return tab

    def update_overview_plot(self):
        self.overview_canvas.fig.clear()
        ax1 = self.overview_canvas.fig.add_subplot(111)

        x = self.month_agg.index.to_timestamp()
        ax2 = ax1.twinx()

        if self.chk_sales.isChecked():
            ax1.plot(x, self.month_agg["sales_qty"], marker="o", label="Продажи, шт")

        if self.chk_profit.isChecked():
            ax2.plot(
                x,
                self.month_agg["net_profit"],
                marker="o",
                linestyle="--",
                label="Чистая прибыль, ₽",
            )

        ax1.set_title("Продажи и чистая прибыль по месяцам")
        ax1.set_xlabel("Месяц")
        ax1.set_ylabel("Продажи, шт")
        ax2.set_ylabel("Чистая прибыль, ₽")
        ax1.grid(alpha=0.3)

        # легенда
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        self.overview_canvas.fig.autofmt_xdate()
        self.overview_canvas.draw_idle()

    def create_cancellations_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.canc_canvas = MplCanvas(tab)
        self.canc_toolbar = NavigationToolbar(self.canc_canvas, tab)

        layout.addWidget(self.canc_toolbar)
        layout.addWidget(self.canc_canvas)

        self.update_cancellations_plot()

        return tab

    def update_cancellations_plot(self):
        self.canc_canvas.fig.clear()
        ax = self.canc_canvas.fig.add_subplot(111)

        x = self.month_agg.index.to_timestamp()

        ax.plot(
            x,
            self.month_agg["cancel_rate"] * 100,
            marker="o",
            label="Доля отмен, %",
        )
        ax.plot(
            x,
            self.month_agg["return_rate"] * 100,
            marker="o",
            linestyle="--",
            label="Доля возвратов, %",
        )

        ax.set_title("Отмены и возвраты по месяцам")
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Доля, %")
        ax.grid(alpha=0.3)
        ax.legend()

        self.canc_canvas.fig.autofmt_xdate()
        self.canc_canvas.draw_idle()

    def create_costs_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.costs_canvas = MplCanvas(tab)
        self.costs_toolbar = NavigationToolbar(self.costs_canvas, tab)

        layout.addWidget(self.costs_toolbar)
        layout.addWidget(self.costs_canvas)

        self.update_costs_plot()

        return tab

    def update_costs_plot(self):
        self.costs_canvas.fig.clear()
        ax = self.costs_canvas.fig.add_subplot(111)

        labels = [p.strftime("%Y-%m") for p in self.month_agg.index]
        bottom = np.zeros(len(self.month_agg))

        stack_cols = self.cost_components + ["other_costs"]

        for col in stack_cols:
            ax.bar(labels, self.month_agg[col], bottom=bottom, label=col)
            bottom += self.month_agg[col].values

        ax.set_title("Структура затрат по месяцам")
        ax.set_xlabel("Месяц")
        ax.set_ylabel("Сумма затрат, ₽")
        ax.legend(title="Компоненты затрат", bbox_to_anchor=(1.05, 1), loc="upper left")
        self.costs_canvas.fig.autofmt_xdate()
        self.costs_canvas.fig.tight_layout()
        self.costs_canvas.draw_idle()

    def create_scatter_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        layout.addLayout(controls)

        controls.addWidget(QLabel("Месяц:"))
        self.scatter_month_combo = QComboBox()
        self.scatter_month_combo.addItem("Все")
        for s in self.month_list_str:
            self.scatter_month_combo.addItem(s)
        self.scatter_month_combo.currentIndexChanged.connect(self.update_scatter_plot)
        controls.addWidget(self.scatter_month_combo)

        controls.addSpacing(20)
        self.scatter_slider_label = QLabel("Минимум продаж: 0")
        controls.addWidget(self.scatter_slider_label)

        self.scatter_slider = QSlider(Qt.Horizontal)
        max_sales = int(self.df["sales_qty"].max()) if not self.df["sales_qty"].isna().all() else 100
        self.scatter_slider.setRange(0, max_sales)
        self.scatter_slider.setValue(0)
        self.scatter_slider.valueChanged.connect(self.update_scatter_plot)
        controls.addWidget(self.scatter_slider)

        controls.addStretch()

        self.scatter_canvas = MplCanvas(tab)
        self.scatter_toolbar = NavigationToolbar(self.scatter_canvas, tab)

        layout.addWidget(self.scatter_toolbar)
        layout.addWidget(self.scatter_canvas)

        self.update_scatter_plot()

        return tab

    def update_scatter_plot(self):
        self.scatter_canvas.fig.clear()
        ax = self.scatter_canvas.fig.add_subplot(111)

        min_sales = self.scatter_slider.value()
        self.scatter_slider_label.setText(f"Минимум продаж: {min_sales}")

        month_text = self.scatter_month_combo.currentText()

        df = self.df.copy()
        if month_text != "Все":
            period = pd.Period(month_text)
            df = df[df["month_period"] == period]

        df = df[df["sales_qty"] >= min_sales]

        if df.empty:
            ax.text(0.5, 0.5, "Нет данных для выбранных фильтров", ha="center", va="center")
        else:
            ax.scatter(df["avg_ppvz_spp_prc"], df["sales_qty"], alpha=0.7)
            ax.set_xlabel("Средняя цена продажи, ₽")
            ax.set_ylabel("Продажи, шт")
            ax.set_title("Связь цены и объёма продаж")
            ax.grid(alpha=0.3)

        self.scatter_canvas.fig.tight_layout()
        self.scatter_canvas.draw_idle()

    def create_top_products_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        layout.addLayout(controls)

        controls.addWidget(QLabel("Количество товаров в ТОП:"))
        self.top_spin = QSpinBox()
        self.top_spin.setRange(3, 50)
        self.top_spin.setValue(10)
        self.top_spin.valueChanged.connect(self.update_top_products_plot)
        controls.addWidget(self.top_spin)

        controls.addStretch()

        self.top_canvas = MplCanvas(tab)
        self.top_toolbar = NavigationToolbar(self.top_canvas, tab)

        layout.addWidget(self.top_toolbar)
        layout.addWidget(self.top_canvas)

        self.update_top_products_plot()

        return tab

    def update_top_products_plot(self):
        self.top_canvas.fig.clear()
        ax = self.top_canvas.fig.add_subplot(111)

        n = self.top_spin.value()
        top_products = (
            self.product_agg.sort_values("net_profit", ascending=False).head(n)
        )

        if top_products.empty:
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        else:
            labels = top_products["product_title"].str.slice(0, 40) + "…"
            ax.barh(labels, top_products["net_profit"])
            ax.invert_yaxis()
            ax.set_xlabel("Чистая прибыль, ₽")
            ax.set_title(f"ТОП-{n} товаров по чистой прибыли")

        self.top_canvas.fig.tight_layout()
        self.top_canvas.draw_idle()

    def create_pie_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        layout.addLayout(controls)

        controls.addWidget(QLabel("Месяц:"))
        self.pie_month_combo = QComboBox()
        for s in self.month_list_str:
            self.pie_month_combo.addItem(s)
        self.pie_month_combo.currentIndexChanged.connect(self.update_pie_plot)
        controls.addWidget(self.pie_month_combo)

        controls.addSpacing(20)
        controls.addWidget(QLabel("Количество товаров в круговой по прибыли:"))

        self.pie_top_spin = QSpinBox()
        self.pie_top_spin.setRange(3, 20)
        self.pie_top_spin.setValue(8)
        self.pie_top_spin.valueChanged.connect(self.update_pie_plot)
        controls.addWidget(self.pie_top_spin)

        controls.addStretch()

        self.pie_canvas = MplCanvas(tab)
        self.pie_toolbar = NavigationToolbar(self.pie_canvas, tab)

        layout.addWidget(self.pie_toolbar)
        layout.addWidget(self.pie_canvas)

        self.update_pie_plot()

        return tab

    def update_pie_plot(self):
        self.pie_canvas.fig.clear()

        if not self.month_list:
            return

        month_text = self.pie_month_combo.currentText()
        period = pd.Period(month_text)

        # Сделаем фигуру пошире конкретно под пай-чарты
        self.pie_canvas.fig.set_size_inches(8, 4.5, forward=True)
        self.pie_canvas.fig.subplots_adjust(wspace=0.4)

        # Левая круговая: структура затрат за месяц
        ax1 = self.pie_canvas.fig.add_subplot(121, aspect="equal")
        row = self.month_agg.loc[period]

        cost_vals = [row[c] for c in self.cost_components + ["other_costs"]]
        cost_labels = self.cost_components + ["other_costs"]

        if sum(cost_vals) > 0:
            wedges1, texts1, autotexts1 = ax1.pie(
                cost_vals,
                labels=cost_labels,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 9},
            )
            ax1.set_title(f"Структура затрат {month_text}", fontsize=11)
        else:
            ax1.text(0.5, 0.5, "Нет данных", ha="center", va="center")
            ax1.set_aspect("equal")

        # Правая круговая: распределение прибыли по товарам за месяц
        ax2 = self.pie_canvas.fig.add_subplot(122, aspect="equal")
        n = self.pie_top_spin.value()

        df_month = self.df[self.df["month_period"] == period]
        profit_by_product = (
            df_month.groupby("product_title")["net_profit"]
            .sum()
            .sort_values(ascending=False)
        )

        if profit_by_product.empty or profit_by_product.sum() == 0:
            ax2.text(0.5, 0.5, "Нет данных по товарам", ha="center", va="center")
            ax2.set_aspect("equal")
        else:
            top = profit_by_product.head(n)
            rest = profit_by_product.iloc[n:].sum()
            if rest > 0:
                top = top.append(pd.Series({"Прочие": rest}))

            # подписи сильно не растягиваем, но делаем покрупнее
            labels = top.index.str.slice(0, 25)
            wedges2, texts2, autotexts2 = ax2.pie(
                top.values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 9},
            )
            ax2.set_title(f"Прибыль по товарам {month_text}", fontsize=11)

        # НЕ вызываем слишком агрессивный tight_layout, чтобы не сжимал круги
        self.pie_canvas.draw_idle()

    def create_3d_anim_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        layout.addLayout(controls)

        self.btn_start_anim = QPushButton("Старт анимации")
        self.btn_stop_anim = QPushButton("Стоп анимации")

        self.btn_start_anim.clicked.connect(self.start_animation)
        self.btn_stop_anim.clicked.connect(self.stop_animation)

        controls.addWidget(self.btn_start_anim)
        controls.addWidget(self.btn_stop_anim)
        controls.addStretch()

        self.anim_canvas = MplCanvas(tab)
        self.anim_toolbar = NavigationToolbar(self.anim_canvas, tab)
        layout.addWidget(self.anim_toolbar)
        layout.addWidget(self.anim_canvas)

        # 3D график + начальный кадр анимации
        self.update_3d_and_animation(initial=True)

        return tab

    def update_3d_and_animation(self, initial=False):
        self.anim_canvas.fig.clear()
        # Сетка 2x1: сверху 3D, снизу — анимируемый 2D
        ax3d = self.anim_canvas.fig.add_subplot(211, projection="3d")
        ax2d = self.anim_canvas.fig.add_subplot(212)

        # 3D scatter по товарам: (avg_price, sales_qty, net_profit)
        data = self.product_agg.copy()
        if data.empty:
            ax3d.text(0.5, 0.5, 0.5, "Нет данных", ha="center", va="center")
        else:
            x = data["avg_price"]
            y = data["sales_qty"]
            z = data["net_profit"]
            ax3d.scatter(x, y, z, alpha=0.6)

            ax3d.set_xlabel("Средняя цена, ₽")
            ax3d.set_ylabel("Продажи, шт")
            ax3d.set_zlabel("Чистая прибыль, ₽")
            ax3d.set_title("3D: цена / продажи / прибыль по товарам")

        # Анимируемая линия: динамика продаж по месяцам
        self.anim_x = self.month_agg.index.to_timestamp()
        self.anim_y = self.month_agg["sales_qty"].values if len(self.month_agg) > 0 else np.array([])

        if len(self.anim_x) == 0:
            ax2d.text(0.5, 0.5, "Нет данных для анимации", ha="center", va="center")
            self.anim_line = None
        else:
            ax2d.set_title("Анимация: накопленная динамика продаж по месяцам")
            ax2d.set_xlabel("Месяц")
            ax2d.set_ylabel("Продажи, шт")
            ax2d.grid(alpha=0.3)
            # Заранее задаём пределы
            ax2d.set_xlim(self.anim_x[0], self.anim_x[-1])
            ax2d.set_ylim(0, max(self.anim_y) * 1.1)
            (self.anim_line,) = ax2d.plot([], [], marker="o")

        self.anim_canvas.fig.tight_layout()
        self.anim_canvas.draw_idle()

        if initial:
            self.animation_step = 0

    def start_animation(self):
        if len(self.month_agg) == 0:
            return
        self.animation_running = True
        if not self.timer.isActive():
            self.timer.start()

    def stop_animation(self):
        self.animation_running = False

    def update_animation(self):
        if not self.animation_running:
            return
        if self.anim_line is None or len(self.anim_x) == 0:
            return

        self.animation_step = (self.animation_step + 1) % len(self.anim_x)
        x_data = self.anim_x[: self.animation_step + 1]
        y_data = self.anim_y[: self.animation_step + 1]
        self.anim_line.set_data(x_data, y_data)
        self.anim_canvas.draw_idle()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
