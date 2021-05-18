from PyQt5.QtWidgets import QMainWindow

from designer.HelpPage import Ui_MainWindow

class HelpPageImpl(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(HelpPageImpl, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Help Page")

