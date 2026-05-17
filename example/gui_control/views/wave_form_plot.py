from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import sys
import random
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class WaveformPlot(QWidget):
    def __init__(self, num_lines=3, labels=None, title="Waveform Plot"):
        super().__init__()
        self.num_lines = num_lines
        self.labels = labels if labels else [f"Line {i+1}" for i in range(num_lines)]
        self.data = [[] for _ in range(self.num_lines)]  # Initialize data list
        self.max_points = 100  # Max display points
        self.setWindowTitle(title)
        # Set layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Initialize matplotlib figure
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Initialize plot lines
        self.lines = [self.ax.plot([], [], label=self.labels[i])[0] for i in range(self.num_lines)]
        self.ax.set_xlim(0, self.max_points)
        self.ax.set_ylim(0, 300)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.legend(loc='upper left')  # Fix legend position to upper left

    def update_data(self, new_data):
        for i in range(self.num_lines):
            self.data[i].append(new_data[i])
            if len(self.data[i]) > self.max_points:
                self.data[i].pop(0)
        self._update_plot()

    def _update_plot(self):
        """Internal method: Update plot"""
        for i, line in enumerate(self.lines):
            line.set_data(range(len(self.data[i])), self.data[i])

        self.ax.set_xlim(0, self.max_points)
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Example: Create a waveform plot with 3 lines
    waveform_plot = WaveformPlot(num_lines=3)
    waveform_plot.resize(800, 400)
    waveform_plot.show()

    # Simulate data update
    timer = QTimer()

    def update():
        import random
        values = [random.randint(0, 300) for _ in range(3)]  # Generate random data
        waveform_plot.update_data(values)

    timer.timeout.connect(update)
    timer.start(100)  # Update every 100 ms

    sys.exit(app.exec_())