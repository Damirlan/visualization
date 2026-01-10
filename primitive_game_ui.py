import random
import sys

from PyQt5.QtCore import (
    Qt,
    QTimer,
    QPropertyAnimation,
    pyqtProperty,
    QEasingCurve,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class AnimatedBar(QWidget):
    def __init__(self, label, color_start, color_end, max_value=100, unit="", parent=None):
        super().__init__(parent)
        self.label = label
        self.color_start = QColor(color_start)
        self.color_end = QColor(color_end)
        self.max_value = max_value
        self.unit = unit
        self._value = max_value
        self._anim = None
        self.setMinimumHeight(26)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_max(self, value):
        self.max_value = max(1, value)
        self.set_value(min(self._value, self.max_value), animate=False)

    def set_value(self, value, animate=True):
        value = max(0, min(self.max_value, value))
        if not animate:
            self._value = value
            self.update()
            return
        if self._anim is not None:
            self._anim.stop()
        self._anim = QPropertyAnimation(self, b"value")
        self._anim.setDuration(420)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.setStartValue(self._value)
        self._anim.setEndValue(value)
        self._anim.start()

    def get_value(self):
        return self._value

    def set_value_property(self, value):
        self._value = value
        self.update()

    value = pyqtProperty(float, fget=get_value, fset=set_value_property)

    def paintEvent(self, event):  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(2, 2, -2, -2)
        radius = rect.height() / 2

        painter.setPen(QPen(QColor("#1c1f2a"), 1))
        painter.setBrush(QColor("#0f121a"))
        painter.drawRoundedRect(rect, radius, radius)

        ratio = 0 if self.max_value == 0 else self._value / self.max_value
        fill_width = max(0, int(rect.width() * ratio))
        if fill_width > 0:
            fill_rect = rect.adjusted(0, 0, -(rect.width() - fill_width), 0)
            gradient = QLinearGradient(fill_rect.topLeft(), fill_rect.topRight())
            gradient.setColorAt(0.0, self.color_start)
            gradient.setColorAt(1.0, self.color_end)
            painter.setBrush(gradient)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(fill_rect, radius, radius)

        text = f"{self.label}: {int(self._value)}/{int(self.max_value)}{self.unit}"
        painter.setPen(QColor("#e6eef9"))
        painter.setFont(QFont("Avenir", 10, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, text)


class EnemyImage(QLabel):
    clicked = pyqtSignal()

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.original = QPixmap(image_path)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("enemyImage")
        self._update_pixmap()

    def _update_pixmap(self):
        if self.original.isNull():
            self.setText("ENEMY")
            return
        scaled = self.original.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        self._update_pixmap()
        super().resizeEvent(event)

    def mousePressEvent(self, event):  # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class StatCard(QFrame):
    def __init__(self, title, subtitle, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("cardSubtitle")

        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)


class GameWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neon Pocket Quest")
        self.resize(1100, 720)

        self.hp_max = 100
        self.energy_max = 80
        self.level = 1
        self.xp = 0
        self.xp_to_level = 100
        self.money = 120
        self.defeated = False
        self.enemy_max = 120
        self.enemy_hp = self.enemy_max
        self.hp = self.hp_max
        self.energy = self.energy_max
        self.heal_ready = True
        self.heal_cooldown = 8
        self.heal_remaining = 0

        self.central = QWidget()
        self.setCentralWidget(self.central)
        main_layout = QVBoxLayout(self.central)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(18)

        header = QHBoxLayout()
        title = QLabel("NEON POCKET QUEST")
        title.setObjectName("mainTitle")
        subtitle = QLabel("Click the enemy to attack, manage energy, and heal on cooldown")
        subtitle.setObjectName("mainSubtitle")
        header.addWidget(title)
        header.addStretch()
        main_layout.addLayout(header)
        main_layout.addWidget(subtitle)

        body = QGridLayout()
        body.setHorizontalSpacing(18)
        body.setVerticalSpacing(18)
        main_layout.addLayout(body)

        self.stats_panel = QFrame()
        self.stats_panel.setObjectName("panel")
        stats_layout = QVBoxLayout(self.stats_panel)
        stats_layout.setContentsMargins(18, 18, 18, 18)
        stats_layout.setSpacing(12)

        self.hp_bar = AnimatedBar("HEALTH", "#ff6774", "#ff2b5f", self.hp_max)
        self.energy_bar = AnimatedBar("ENERGY", "#7ad7ff", "#2fa1ff", self.energy_max)
        self.xp_bar = AnimatedBar("LEVEL PROGRESS", "#ffe08a", "#ffb347", self.xp_to_level)

        stats_layout.addWidget(self.hp_bar)
        stats_layout.addWidget(self.energy_bar)
        stats_layout.addWidget(self.xp_bar)

        level_card = StatCard("LEVEL", "1")
        self.level_value = level_card.subtitle
        stats_layout.addWidget(level_card)

        money_card = StatCard("CREDITS", str(self.money))
        self.money_value = money_card.subtitle
        stats_layout.addWidget(money_card)

        status_card = StatCard("STATE", "Ready")
        self.state_value = status_card.subtitle
        stats_layout.addWidget(status_card)

        self.log = QLabel("Awakened in the neon arena.")
        self.log.setWordWrap(True)
        self.log.setObjectName("logText")
        stats_layout.addWidget(self.log)

        body.addWidget(self.stats_panel, 0, 0, 2, 1)

        self.arena_panel = QFrame()
        self.arena_panel.setObjectName("panel")
        arena_layout = QVBoxLayout(self.arena_panel)
        arena_layout.setContentsMargins(18, 18, 18, 18)
        arena_layout.setSpacing(12)

        self.enemy_image = EnemyImage("images.jpeg")
        arena_layout.addWidget(self.enemy_image)

        self.enemy_bar = AnimatedBar("ENEMY", "#c79eff", "#6e44ff", self.enemy_max)
        arena_layout.addWidget(self.enemy_bar)

        body.addWidget(self.arena_panel, 0, 1, 1, 1)

        self.actions_panel = QFrame()
        self.actions_panel.setObjectName("panel")
        actions_layout = QVBoxLayout(self.actions_panel)
        actions_layout.setContentsMargins(18, 18, 18, 18)
        actions_layout.setSpacing(12)

        action_title = QLabel("ACTIONS")
        action_title.setObjectName("cardTitle")
        actions_layout.addWidget(action_title)

        click_hint = QLabel("Click the enemy to strike.")
        click_hint.setObjectName("cardSubtitle")

        self.btn_special = QPushButton("Overcharge (25 energy)")
        self.btn_rest = QPushButton("Rest")
        self.btn_heal = QPushButton("Heal +25 HP")
        self.btn_reset = QPushButton("Reset")

        actions_layout.addWidget(click_hint)
        actions_layout.addWidget(self.btn_special)
        actions_layout.addWidget(self.btn_rest)
        actions_layout.addWidget(self.btn_heal)
        actions_layout.addWidget(self.btn_reset)
        actions_layout.addStretch()

        body.addWidget(self.actions_panel, 1, 1, 1, 1)

        self.enemy_image.clicked.connect(self.attack)
        self.btn_special.clicked.connect(self.special)
        self.btn_rest.clicked.connect(self.rest)
        self.btn_heal.clicked.connect(self.heal)
        self.btn_reset.clicked.connect(self.reset_game)

        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(900)
        self.tick_timer.timeout.connect(self.passive_tick)
        self.tick_timer.start()

        self.enemy_timer = QTimer(self)
        self.enemy_timer.setInterval(1200)
        self.enemy_timer.timeout.connect(self.enemy_attack)
        self.enemy_timer.start()

        self.heal_timer = QTimer(self)
        self.heal_timer.setInterval(1000)
        self.heal_timer.timeout.connect(self.update_heal_cooldown)

        self.apply_styles()
        self.refresh_ui(animate=False)

    def apply_styles(self):
        self.setStyleSheet(
            """
            QWidget {
                background: #0b0e14;
                color: #e6eef9;
                font-family: 'Avenir';
            }
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0b0e14, stop:0.6 #101828, stop:1 #0b0e14);
            }
            #panel {
                background: #121826;
                border: 1px solid #1f2a44;
                border-radius: 18px;
            }
            #card {
                background: #0f1422;
                border: 1px solid #1f2a44;
                border-radius: 14px;
            }
            #cardTitle {
                font-size: 14px;
                letter-spacing: 2px;
                color: #7db1ff;
            }
            #cardSubtitle {
                font-size: 22px;
                font-weight: 600;
            }
            #mainTitle {
                font-size: 28px;
                font-weight: 700;
                letter-spacing: 4px;
            }
            #mainSubtitle {
                font-size: 14px;
                color: #9fb1c9;
            }
            #logText {
                font-size: 13px;
                color: #c6d4ea;
            }
            #enemyImage {
                background: #0f1422;
                border: 1px solid #2a3654;
                border-radius: 16px;
                padding: 10px;
            }
            QPushButton {
                background: #1a2440;
                border: 1px solid #2c3d62;
                border-radius: 12px;
                padding: 10px 14px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #233256;
                border-color: #385083;
            }
            QPushButton:pressed {
                background: #0f1422;
            }
            """
        )

    def refresh_ui(self, animate=True):
        self.hp_bar.set_max(self.hp_max)
        self.energy_bar.set_max(self.energy_max)
        self.xp_bar.set_max(self.xp_to_level)
        self.enemy_bar.set_max(self.enemy_max)

        self.hp_bar.set_value(self.hp, animate=animate)
        self.energy_bar.set_value(self.energy, animate=animate)
        self.xp_bar.set_value(self.xp, animate=animate)
        self.enemy_bar.set_value(self.enemy_hp, animate=animate)

        self.level_value.setText(str(self.level))
        self.money_value.setText(str(self.money))

        if self.hp <= 0:
            self.state_value.setText("Defeated")
            self.btn_special.setEnabled(False)
            self.btn_rest.setEnabled(False)
            self.btn_heal.setEnabled(False)
            self.enemy_image.setEnabled(False)
        else:
            self.state_value.setText("Engaged")
            self.btn_special.setEnabled(True)
            self.btn_rest.setEnabled(True)
            self.btn_heal.setEnabled(self.heal_ready)
            self.enemy_image.setEnabled(True)

    def log_event(self, message):
        self.log.setText(message)

    def attack(self):
        if self.hp <= 0:
            return
        if self.energy < 10:
            self.log_event("Not enough energy. Try to rest.")
            return
        self.energy -= 10
        damage = random.randint(8, 18)
        self.enemy_hp = max(0, self.enemy_hp - damage)
        self.log_event(f"Strike landed for {damage}.")
        self.handle_enemy_state()
        self.refresh_ui()

    def special(self):
        if self.energy < 25:
            self.log_event("Overcharge needs 25 energy.")
            return
        self.energy -= 25
        damage = random.randint(20, 34)
        self.enemy_hp = max(0, self.enemy_hp - damage)
        self.log_event(f"Overcharge blast for {damage}.")
        self.handle_enemy_state()
        self.refresh_ui()

    def rest(self):
        self.energy = min(self.energy_max, self.energy + 20)
        self.hp = min(self.hp_max, self.hp + 6)
        self.log_event("Restored some energy and health.")
        self.refresh_ui()

    def handle_enemy_state(self):
        if self.enemy_hp <= 0:
            self.enemy_max += 20
            self.enemy_hp = self.enemy_max
            gained_xp = random.randint(30, 50)
            earned = random.randint(35, 60)
            self.xp += gained_xp
            self.money += earned
            self.level += 1
            self.hp_max += 8
            self.energy_max += 4
            self.xp_to_level = int(self.xp_to_level * 1.2)
            self.xp = 0
            self.hp = self.hp_max
            self.energy = self.energy_max
            self.log_event(f"Enemy down. +{earned} credits and level up.")

    def passive_tick(self):
        if self.hp <= 0:
            return
        self.energy = min(self.energy_max, self.energy + 2)
        if self.hp < self.hp_max and random.random() < 0.4:
            self.hp = min(self.hp_max, self.hp + 1)
        self.refresh_ui()

    def enemy_attack(self):
        if self.hp <= 0 or self.enemy_hp <= 0:
            return
        if random.random() < 0.55:
            damage = random.randint(5, 12)
            self.hp = max(0, self.hp - damage)
            self.log_event(f"Enemy hits for {damage}.")
            if self.hp <= 0 and not self.defeated:
                self.defeated = True
                loss = min(self.money, 40)
                self.money -= loss
                self.log_event(f"Defeated. Lost {loss} credits.")
                self.enemy_timer.stop()
            self.refresh_ui()

    def heal(self):
        if not self.heal_ready or self.hp <= 0:
            return
        self.hp = min(self.hp_max, self.hp + 25)
        self.heal_ready = False
        self.heal_remaining = self.heal_cooldown
        self.btn_heal.setEnabled(False)
        self.btn_heal.setText(f"Heal ready in {self.heal_remaining}s")
        self.heal_timer.start()
        self.log_event("Heal activated.")
        self.refresh_ui()

    def update_heal_cooldown(self):
        if self.heal_remaining <= 0:
            self.heal_ready = True
            self.heal_timer.stop()
            self.btn_heal.setText("Heal +25 HP")
            self.refresh_ui()
            return
        self.heal_remaining -= 1
        self.btn_heal.setText(f"Heal ready in {self.heal_remaining}s")

    def reset_game(self):
        self.hp_max = 100
        self.energy_max = 80
        self.level = 1
        self.xp = 0
        self.xp_to_level = 100
        self.money = 120
        self.enemy_max = 120
        self.enemy_hp = self.enemy_max
        self.hp = self.hp_max
        self.energy = self.energy_max
        self.defeated = False
        self.heal_ready = True
        self.heal_remaining = 0
        self.btn_heal.setText("Heal +25 HP")
        self.enemy_timer.start()
        self.log_event("Systems reset. Back to level 1.")
        self.refresh_ui()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameWindow()
    window.show()
    sys.exit(app.exec_())
