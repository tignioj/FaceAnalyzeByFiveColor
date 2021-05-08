import sys
from GUI import *
from PyQt5.QtWidgets import QApplication, QMainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainGUI()
    myWin.show()
    sys.exit(app.exec_())