import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, RadioButtons

from sklearn.datasets import make_moons, make_circles, make_blobs


# ========== 1. Нейросеть с несколькими скрытыми слоями ==========

class MultiLayerNN:
    def __init__(self, layer_sizes, lr=0.1, seed=0):
        """
        layer_sizes: список размеров слоёв, например [2, 4, 4, 3, 1]
        """
        self.layer_sizes = layer_sizes
        self.lr = lr
        rng = np.random.RandomState(seed)

        self.W = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
            # Инициализация Ксавье
            limit = np.sqrt(6 / (fan_in + fan_out))
            W_i = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            b_i = np.zeros((1, fan_out))
            self.W.append(W_i)
            self.b.append(b_i)

    def forward(self, X):
        """
        Возвращает:
        z_list: линейные комбинации (без активации)
        a_list: активации (a0 = X, aL = выход)
        """
        a = X
        a_list = [a]
        z_list = []
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            z_list.append(z)
            if i == len(self.W) - 1:
                # последний слой — сигмоида
                a = 1.0 / (1.0 + np.exp(-z))
            else:
                # скрытые слои — tanh
                a = np.tanh(z)
            a_list.append(a)
        return z_list, a_list

    def loss(self, y_true, y_pred):
        eps = 1e-8
        return - (y_true * np.log(y_pred + eps) +
                  (1 - y_true) * np.log(1 - y_pred + eps))

    def train_step(self, x, y):
        """
        Один шаг обучения (SGD на одном объекте).
        Возвращает (loss, (z_list, a_list))
        """
        # Прямой проход
        z_list, a_list = self.forward(x)
        y_pred = a_list[-1]

        # Потери
        L = self.loss(y, y_pred)  # shape (1,1)

        # Обратное распространение
        # delta_L = dL/dz_L для выхода(sigmoid + BCE) = y_pred - y
        delta = y_pred - y  # (1, out_dim)

        dW_list = [None] * len(self.W)
        db_list = [None] * len(self.b)

        for i in reversed(range(len(self.W))):
            a_prev = a_list[i]  # (1, fan_in)
            dW = a_prev.T @ delta
            db = delta

            dW_list[i] = dW
            db_list[i] = db

            if i > 0:
                # для скрытого слоя: delta_prev = (delta @ W_i.T) * f'(z_prev)
                da_prev = delta @ self.W[i].T
                z_prev = z_list[i-1]
                # tanh'
                delta = da_prev * (1 - np.tanh(z_prev) ** 2)

        # Обновление весов
        for i in range(len(self.W)):
            self.W[i] -= self.lr * dW_list[i]
            self.b[i] -= self.lr * db_list[i]

        return float(L.squeeze()), (z_list, a_list)


# ========== 2. Генерация различных датасетов (2 признака) ==========

def make_dataset(kind="moons", n_samples=200, noise=0.15, seed=0):
    if kind == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif kind == "circles":
        X, y = make_circles(
            n_samples=n_samples, noise=noise, factor=0.4, random_state=seed
        )
    elif kind == "blobs":
        X, y = make_blobs(
            n_samples=n_samples, centers=2, cluster_std=1.2, random_state=seed
        )
        # приведём к 0/1
        y = (y > 0).astype(int)
    else:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    # Стандартизация, чтобы значения были ± нормальные
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = y.reshape(-1, 1)
    return X, y


# ========== 3. Глобальные переменные для анимации ==========

DATASET_KIND = "moons"
LAYER_SIZES = [2, 4, 4, 3, 1]
LEARNING_RATE = 0.15

X, y = make_dataset(DATASET_KIND)
n_samples = X.shape[0]

net = MultiLayerNN(LAYER_SIZES, lr=LEARNING_RATE, seed=1)

# сетка для решающей границы
def make_grid(X, num=100, pad=1.0):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, num),
        np.linspace(y_min, y_max, num),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid, (x_min, x_max, y_min, y_max)

xx, yy, grid_points, limits = make_grid(X)
x_min, x_max, y_min, y_max = limits

loss_history = []

current_index = 0  # текущий объект
phase = 0         # фаза прохождения по слоям
num_phases = len(LAYER_SIZES) - 1  # количество связей между слоями

last_forward = None  # (z_list, a_list) для текущего примера
last_sample = None   # (x, t)


# ========== 4. Фигура, оси, схема сети ==========

plt.style.use("default")
fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.8], height_ratios=[3, 1])

ax_net = fig.add_subplot(gs[:, 0])    # слева — сеть
ax_data = fig.add_subplot(gs[0, 1])   # справа сверху — данные
ax_loss = fig.add_subplot(gs[1, 1])   # справа снизу — loss

fig.suptitle("Анимация обучения многослойной нейросети", fontsize=14)


# Позиции нейронов по слоям
def build_layer_positions(layer_sizes):
    positions = []
    x_spacing = 1.2
    for i, size in enumerate(layer_sizes):
        x = i * x_spacing
        if size == 1:
            ys = np.array([0.0])
        else:
            ys = np.linspace(0.8, -0.8, size)
        layer_pos = np.column_stack([np.full(size, x), ys])
        positions.append(layer_pos)
    return positions

positions = build_layer_positions(LAYER_SIZES)

def weight_to_linewidth(w, base=0.4, scale=4.5):
    return base + scale * np.abs(w)


# ========== 5. Функции перерисовки ==========

def reset_dataset(kind):
    """
    Смена датасета: пересоздаём X, y, сеть и сетку.
    """
    global DATASET_KIND, X, y, n_samples, net, xx, yy, grid_points
    global x_min, x_max, y_min, y_max, loss_history, current_index, last_forward, last_sample

    DATASET_KIND = kind
    X, y = make_dataset(kind, n_samples=220, noise=0.18, seed=0)
    n_samples = X.shape[0]

    net = MultiLayerNN(LAYER_SIZES, lr=LEARNING_RATE, seed=1)

    xx, yy, grid_points, limits = make_grid(X)
    x_min, x_max, y_min, y_max = limits

    loss_history.clear()
    current_index = 0
    last_forward = None
    last_sample = None


def draw_network(ax, net, positions, forward_result, phase, sample):
    """
    Рисует схему сети, веса и активации.
    phase — номер слоя, через связи которого визуально "идут" данные.
    """
    ax.clear()
    ax.set_title("Проход данных по слоям и изменение весов", fontsize=11)
    ax.set_xlim(-0.5, len(positions) * 1.2 - 0.7)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")

    if forward_result is None or sample is None:
        return

    z_list, a_list = forward_result
    x, t = sample

    cmap = plt.cm.viridis

    # Рисуем связи
    for l in range(len(net.W)):
        W = net.W[l]
        layer_from = positions[l]
        layer_to = positions[l+1]
        is_active_layer = (l == phase)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                w = W[i, j]
                lw = weight_to_linewidth(w)
                color = "green" if w >= 0 else "red"
                alpha = 0.8 if is_active_layer else 0.25
                ax.plot(
                    [layer_from[i, 0], layer_to[j, 0]],
                    [layer_from[i, 1], layer_to[j, 1]],
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                )

    # Рисуем нейроны
    for l, layer_pos in enumerate(positions):
        acts = a_list[l][0]  # shape (layer_size,)
        if l == len(positions) - 1:
            vals = np.clip(acts, 0, 1)             # выход (0..1)
        else:
            vals = (np.clip(acts, -2, 2) + 2) / 4  # ~[-2,2] -> [0,1]
        colors = cmap(vals)

        s = 900 if l == phase or (l == phase + 1) else 700

        ax.scatter(
            layer_pos[:, 0], layer_pos[:, 1],
            s=s, c=colors, edgecolors="k", zorder=3
        )

    # Подписи слоёв
    ax.text(positions[0][0, 0], 1.0, "Входной слой", fontsize=9, ha="center")
    for i in range(1, len(positions) - 1):
        ax.text(positions[i][0, 0], 1.0, f"Скрытый {i}", fontsize=9, ha="center")
    ax.text(positions[-1][0, 0], 1.0, "Выход", fontsize=9, ha="center")

    # Подписи входных значений и выхода
    ax.text(
        positions[0][0, 0] - 0.1, positions[0][0, 1],
        f"x1={x[0,0]:.2f}", ha="right", va="center", fontsize=8
    )
    ax.text(
        positions[0][1, 0] - 0.1, positions[0][1, 1],
        f"x2={x[0,1]:.2f}", ha="right", va="center", fontsize=8
    )
    ax.text(
        positions[-1][0, 0] + 0.1, positions[-1][0, 1],
        f"ŷ={a_list[-1][0,0]:.2f}", ha="left", va="center", fontsize=9
    )

    ax.text(
        0.5, -1.02,
        "Цвет и толщина линий показывают знак и величину весов (зелёный — +, красный — −).\n"
        "Активация нейронов отображается оттенком цвета.",
        fontsize=7.5, ha="center"
    )


def draw_data_and_boundary(ax, net, X, y, xx, yy, grid_points, sample):
    ax.clear()
    ax.set_title(f"Данные («{DATASET_KIND}») и решающая граница", fontsize=11)

    _, a_grid = net.forward(grid_points)
    Z = a_grid[-1].reshape(xx.shape)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

    ax.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.6)
    ax.scatter(
        X[:, 0], X[:, 1],
        c=y[:, 0], cmap="bwr", edgecolors="k", s=40, alpha=0.85
    )

    if sample is not None:
        x, t = sample
        ax.scatter(
            x[0, 0], x[0, 1],
            s=130, facecolors="none", edgecolors="yellow", linewidths=2.0,
            label="Текущий объект"
        )
        ax.legend(loc="upper left", fontsize=7)


def draw_loss(ax, loss_history, current_loss):
    ax.clear()
    ax.set_title("Функция потерь", fontsize=11)
    ax.set_xlabel("Шаг обучения")
    ax.set_ylabel("Loss")

    if loss_history:
        steps = np.arange(1, len(loss_history) + 1)
        ax.plot(steps, loss_history, marker="o", markersize=3)
    ax.grid(alpha=0.3)

    ax.text(
        0.98, 0.95,
        f"Текущий loss: {current_loss:.3f}",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )


# ========== 6. Функция обновления кадра для анимации ==========

def update(frame):
    global current_index, phase, last_forward, last_sample

    # ФАЗЫ:
    # 0..num_phases-1 — визуальное прохождение через связи,
    # на phase == 0 делаем новый шаг обучения (новый объект).
    if phase == 0:
        x = X[current_index:current_index+1]
        t = y[current_index:current_index+1]
        current_index = (current_index + 1) % n_samples

        L, forward_result = net.train_step(x, t)
        loss_history.append(L)
        last_forward = forward_result
        last_sample = (x, t)
        current_loss = L
    else:
        current_loss = loss_history[-1] if loss_history else 0.0

    draw_network(ax_net, net, positions, last_forward, phase, last_sample)
    draw_data_and_boundary(ax_data, net, X, y, xx, yy, grid_points, last_sample)
    draw_loss(ax_loss, loss_history, current_loss)

    phase = (phase + 1) % num_phases

    fig.canvas.draw_idle()
    return []


# ========== 7. Элементы управления: кнопки, слайдер, переключатель датасета ==========

ax_play = plt.axes([0.10, 0.02, 0.08, 0.05])
ax_pause = plt.axes([0.20, 0.02, 0.08, 0.05])
btn_play = Button(ax_play, "Старт")
btn_pause = Button(ax_pause, "Пауза")

ax_speed = plt.axes([0.35, 0.025, 0.35, 0.03])
speed_slider = Slider(
    ax_speed, "Скорость", valmin=50, valmax=800, valinit=150, valstep=10,
)

ax_radio = plt.axes([0.02, 0.02, 0.06, 0.16])
radio = RadioButtons(ax_radio, ("moons", "circles", "blobs"))

anim = animation.FuncAnimation(
    fig,
    update,
    frames=1000,
    interval=150,
    blit=False,
    repeat=True,
)


def on_play(event):
    anim.event_source.start()

def on_pause(event):
    anim.event_source.stop()

def on_speed_change(val):
    anim.event_source.interval = val  # меньше интервал — быстрее

def on_dataset_change(label):
    reset_dataset(label)
    update(0)
    fig.canvas.draw_idle()


btn_play.on_clicked(on_play)
btn_pause.on_clicked(on_pause)
speed_slider.on_changed(on_speed_change)
radio.on_clicked(on_dataset_change)

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.show()
