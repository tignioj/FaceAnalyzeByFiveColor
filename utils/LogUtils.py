from utils.MyDateUtils import getTodayYearMonthDayHourMinSec


class LogUtils:
    @staticmethod
    def log(label, msg, obj=""):
        print("[" + getTodayYearMonthDayHourMinSec() + "]" + label, msg, obj)
