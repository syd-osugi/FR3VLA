import sys,os,time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLineEdit, QPushButton, QWidget, QGridLayout, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(target_dir)
from RealHand.utils.load_write_yaml import LoadWriteYaml
class RightView(QMainWindow):
    add_button_handle = pyqtSignal(str)  # Define a signal
    handle_button_click = pyqtSignal(str)  # Define a signal
    def __init__(self,hand_joint="L20", hand_type="left"):
        super().__init__()
        self.hand_joint = hand_joint
        self.hand_type = hand_type
        self.buttons = []
        self.yaml = LoadWriteYaml() # Initialize configuration file
        self.all_action = None
        self.all_action = self.yaml.load_action_yaml(hand_type=self.hand_type,hand_joint=self.hand_joint)
        self.setWindowTitle("Button Grid Layout")
        self.setGeometry(100, 100, 600, 400)
        self.init_ui()
        self.init_buttons()

    def init_ui(self):
        # Main window container
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Main layout
        self.main_layout = QVBoxLayout(main_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Input field and Add button
        self.input_field = QLineEdit()
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_button_to_list)

        # Input field and button layout
        self.top_layout = QVBoxLayout()
        self.top_layout.setSpacing(10)
        self.top_layout.addWidget(self.input_field)
        self.top_layout.addWidget(self.add_button)

        # Add to main layout
        self.main_layout.addLayout(self.top_layout)
        # Create a scroll area for button list
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(10, 10, 10, 10)
        self.scroll_layout.setSpacing(10)
        self.scroll_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # Align Left and Top
        self.scroll_area.setWidget(self.scroll_widget)

        # Add scroll area to main layout
        self.main_layout.addWidget(self.scroll_area)

        # Grid layout parameters
        self.row = 0  # Current row
        self.column = 0  # Current column
        self.BUTTONS_PER_ROW = 2  # Buttons per row limit
    def init_buttons(self):
        if self.all_action == None:
            return
        for item in self.all_action:
            button = QPushButton(item["ACTION_NAME"])
            button.setFixedWidth(100)  # Set button width as needed
            button.setFixedHeight(30)  # Set button height as needed
            #button.clicked.connect(lambda checked, action_pos=item["ACTION_POS"]: self.button_clicked.emit(action_pos))
            button.clicked.connect(lambda checked, text=item["ACTION_NAME"]: self.handle_button_click.emit(text))
            # Add button to grid layout
            self.scroll_layout.addWidget(button, self.row, self.column, alignment=Qt.AlignLeft | Qt.AlignTop)

            # Update row/column position
            self.column += 1
            if self.column >= self.BUTTONS_PER_ROW:  # Wrap to next line when exceeding buttons per row
                self.column = 0
                self.row += 1

    def add_button_to_list(self):
        text = self.input_field.text().strip()
        if text:
            
            button = QPushButton(text)
            button.setFixedWidth(100)  # Set button width as needed
            button.setFixedHeight(30)  # Set button height as needed
            button.clicked.connect(lambda checked, text=text: self.handle_button_click.emit(text))
            # Add button to grid layout
            self.scroll_layout.addWidget(button, self.row, self.column, alignment=Qt.AlignLeft | Qt.AlignTop)

            # Update row/column position
            self.column += 1
            if self.column >= self.BUTTONS_PER_ROW:  # Wrap to next line when exceeding buttons per row
                self.column = 0
            self.input_field.clear()  # Clear input field
            self.buttons.append(button)
            self.add_button_handle.emit(text)  # Emit signal
            

    def clear_scroll_layout(self):
        """Clear all widgets in scroll_layout"""
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()