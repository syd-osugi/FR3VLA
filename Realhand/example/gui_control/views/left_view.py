from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
)
from PyQt5.QtCore import Qt,pyqtSignal

class LeftView(QWidget):
    # Define a signal to be emitted when the slider value changes
    slider_value_changed = pyqtSignal(dict)
    def __init__(self, joint_name=[],init_pos=[]):
        super().__init__()
        self.is_open = True
        self.joint_name = joint_name
        self.init_pos = init_pos
        self.init_view()

    def init_view(self):
        main_layout = QVBoxLayout(self)
        # Store sliders and labels
        self.sliders = []
        self.labels = []
        # Create sliders
        for i in range(len(self.joint_name)):
            # Horizontal layout for each slider and label
            slider_layout = QHBoxLayout()
            # Label displays the slider value
            label = QLabel(f"{self.joint_name[i]}: 255", self)
            label.setFixedWidth(110)  # Set fixed width
            self.labels.append(label)
            slider_layout.addWidget(label)

            # Slider
            slider = QSlider(Qt.Horizontal, self)
            slider.setRange(0, 255)
            slider.setValue(self.init_pos[i])
            slider.setFixedHeight(15)
            slider.valueChanged.connect(lambda value, index=i: self.update_label(index, value))
            self.sliders.append(slider)
            slider_layout.addWidget(slider)
            main_layout.addLayout(slider_layout)
        # Create Open/Close button
        self.toggle_button = QPushButton("Opened", self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self.toggle_button_clicked)
        main_layout.addWidget(self.toggle_button)

    def update_label(self, index, value):
        self.labels[index].setText(f"{self.joint_name[index]}: {value}")
        slider_values = {}
        sliders = self.findChildren(QSlider)
        for i, slider in enumerate(sliders):
            slider_values[i] = slider.value()
        # Emit signal, passing the current value of the slider
        self.slider_value_changed.emit(slider_values)
        
        
    def set_slider_values(self, values):
        for i, value in enumerate(values):
            if i < len(self.sliders):
                self.sliders[i].setValue(value)
                
    def get_slider_values(self):
        """Get values of all sliders"""
        return [slider.value() for slider in self.sliders]
    def handle_button_click(self, text):
        print(f"Button clicked with text: {text}")
        # Handle button click event here

    def toggle_button_clicked(self):
        if self.toggle_button.isChecked():
            self.toggle_button.setText("Closed")
            self.is_open = False
            # Handle open state
        else:
            self.toggle_button.setText("Opened")
            self.is_open = True
            # Handle closed state