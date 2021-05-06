from utils.MyDateUtils import getTodayYearMonthDayHourMinSec


class LogUtils:
    @staticmethod
    def log(label, msg, obj=""):
        print("[" + getTodayYearMonthDayHourMinSec() + "]" + label, msg, obj)

    @staticmethod
    def error(label, msg, obj=""):
        CRED = '\033[91m'
        CEND = '\033[0m'
        print(CRED + "[" + getTodayYearMonthDayHourMinSec() + "]" + CEND + label, msg, obj)
