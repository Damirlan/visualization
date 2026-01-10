import math
import sys

from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLCDNumber,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class GaugeWidget(QWidget):
    def __init__(self, title, unit, min_value, max_value, major_step, parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.min_value = min_value
        self.max_value = max_value
        self.major_step = major_step
        self.value = min_value

        self.start_angle = -120
        self.span_angle = 240

    def set_value(self, value):
        self.value = max(self.min_value, min(self.max_value, value))
        self.update()

    def _value_to_angle(self, value):
        ratio = (value - self.min_value) / (self.max_value - self.min_value)
        return self.start_angle + self.span_angle * ratio

    def paintEvent(self, event):  # noqa: N802 - PyQt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(10, 10, -10, -10)
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2

        painter.setPen(QPen(QColor("#2b2b2b"), 4))
        painter.setBrush(QColor("#111111"))
        painter.drawEllipse(center, int(radius), int(radius))

        self._draw_ticks(painter, center, radius)
        self._draw_arc(painter, center, radius)
        self._draw_needle(painter, center, radius)
        self._draw_labels(painter, center, radius)

    def _draw_ticks(self, painter, center, radius):
        painter.setPen(QPen(QColor("#d9d9d9"), 2))
        major_count = int((self.max_value - self.min_value) / self.major_step)

        for i in range(major_count + 1):
            value = self.min_value + i * self.major_step
            angle_deg = self._value_to_angle(value)
            angle_rad = math.radians(angle_deg)

            inner = radius * 0.78
            outer = radius * 0.9
            x1 = center.x() + inner * math.cos(angle_rad)
            y1 = center.y() + inner * math.sin(angle_rad)
            x2 = center.x() + outer * math.cos(angle_rad)
            y2 = center.y() + outer * math.sin(angle_rad)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            label_radius = radius * 0.62
            lx = center.x() + label_radius * math.cos(angle_rad)
            ly = center.y() + label_radius * math.sin(angle_rad)

            painter.setFont(QFont("Helvetica", 10, QFont.Bold))
            text = str(int(value))
            metrics = painter.fontMetrics()
            painter.drawText(
                int(lx - metrics.horizontalAdvance(text) / 2),
                int(ly + metrics.ascent() / 2),
                text,
            )

        painter.setPen(QPen(QColor("#8a8a8a"), 1))
        minor_step = self.major_step / 2
        minor_count = int((self.max_value - self.min_value) / minor_step)
        for i in range(minor_count + 1):
            value = self.min_value + i * minor_step
            if value % self.major_step == 0:
                continue
            angle_deg = self._value_to_angle(value)
            angle_rad = math.radians(angle_deg)
            inner = radius * 0.82
            outer = radius * 0.9
            x1 = center.x() + inner * math.cos(angle_rad)
            y1 = center.y() + inner * math.sin(angle_rad)
            x2 = center.x() + outer * math.cos(angle_rad)
            y2 = center.y() + outer * math.sin(angle_rad)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

    def _draw_arc(self, painter, center, radius):
        painter.setPen(QPen(QColor("#e54b4b"), 6))
        angle = self._value_to_angle(self.value)
        span = angle - self.start_angle
        rect = (center.x() - radius * 0.9, center.y() - radius * 0.9, radius * 1.8, radius * 1.8)
        painter.drawArc(
            int(rect[0]),
            int(rect[1]),
            int(rect[2]),
            int(rect[3]),
            int((90 - self.start_angle) * 16),
            int(-span * 16),
        )

    def _draw_needle(self, painter, center, radius):
        angle_deg = self._value_to_angle(self.value)
        angle_rad = math.radians(angle_deg)

        painter.setPen(QPen(QColor("#f5f5f5"), 3))
        needle_len = radius * 0.7
        x = center.x() + needle_len * math.cos(angle_rad)
        y = center.y() + needle_len * math.sin(angle_rad)
        painter.drawLine(center, QPoint(int(x), int(y)))

        painter.setBrush(QColor("#f5f5f5"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, int(radius * 0.05), int(radius * 0.05))

    def _draw_labels(self, painter, center, radius):
        painter.setPen(QPen(QColor("#f0f0f0"), 1))
        painter.setFont(QFont("Helvetica", 12, QFont.Bold))
        title_metrics = painter.fontMetrics()
        painter.drawText(
            int(center.x() - title_metrics.horizontalAdvance(self.title) / 2),
            int(center.y() + radius * 0.25),
            self.title,
        )

        painter.setFont(QFont("Helvetica", 10))
        unit_metrics = painter.fontMetrics()
        painter.drawText(
            int(center.x() - unit_metrics.horizontalAdvance(self.unit) / 2),
            int(center.y() + radius * 0.38),
            self.unit,
        )


class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Приборная панель")
        self.resize(900, 520)

        self.speed = 0.0
        self.rpm = 800.0
        self.accelerating = False

        self.speed_gauge = GaugeWidget("Скорость", "км/ч", 0, 240, 20)
        self.rpm_gauge = GaugeWidget("Обороты", "x1000", 0, 8, 1)

        self.speed_display = QLCDNumber()
        self.speed_display.setDigitCount(3)
        self.speed_display.display(0)

        self.rpm_display = QLCDNumber()
        self.rpm_display.setDigitCount(4)
        self.rpm_display.display(0)

        self.gas_button = QPushButton("Газ")
        self.gas_button.setCheckable(True)
        self.gas_button.pressed.connect(self._start_accel)
        self.gas_button.released.connect(self._stop_accel)

        info_row = QHBoxLayout()
        info_row.addWidget(QLabel("Скорость"))
        info_row.addWidget(self.speed_display)
        info_row.addSpacing(20)
        info_row.addWidget(QLabel("RPM"))
        info_row.addWidget(self.rpm_display)
        info_row.addStretch()
        info_row.addWidget(self.gas_button)

        gauges_row = QHBoxLayout()
        gauges_row.addWidget(self.speed_gauge)
        gauges_row.addWidget(self.rpm_gauge)

        layout = QVBoxLayout()
        layout.addLayout(gauges_row)
        layout.addSpacing(10)
        layout.addLayout(info_row)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._update_state)
        self.timer.start()

    def _start_accel(self):
        self.accelerating = True

    def _stop_accel(self):
        self.accelerating = False

    def _update_state(self):
        accel_rate = 1.2
        decel_rate = 0.7

        if self.accelerating:
            self.speed += accel_rate
        else:
            self.speed -= decel_rate

        self.speed = max(0.0, min(240.0, self.speed))

        base_rpm = 800 + self.speed * 20
        boost = 800 if self.accelerating else 0
        self.rpm = max(800.0, min(8000.0, base_rpm + boost))

        self.speed_gauge.set_value(self.speed)
        self.rpm_gauge.set_value(self.rpm / 1000)

        self.speed_display.display(int(self.speed))
        self.rpm_display.display(int(self.rpm))


def main():
    app = QApplication(sys.argv)
    window = DashboardWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
