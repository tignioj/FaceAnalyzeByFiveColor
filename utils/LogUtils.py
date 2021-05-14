from utils.MyDateUtils import getTodayYearMonthDayHourMinSec

EDIT_TEXT_TYPE_UPDATE = 0
EDIT_TEXT_TYPE_APPEND = 1
EDIT_TEXT_TYPE_CLEAN = 2
EDIT_TEXT_TYPE_ERROR = 3


class LogUtils:
    @staticmethod
    def log(label, msg, obj="", progress=None):
        info = "[" + getTodayYearMonthDayHourMinSec() + "]" + label + " " + msg
        print("progressSignal:", LogUtils.progressSignal)
        if LogUtils.progressSignal is not None:
            if progress is not None:
                LogUtils.progressSignal.emit({'text': msg, 'type': EDIT_TEXT_TYPE_APPEND, 'progress': progress})
            else:
                LogUtils.progressSignal.emit({'text': msg, 'type': EDIT_TEXT_TYPE_APPEND})
        print(info, obj)

    @staticmethod
    def error(label, msg, obj=""):
        CRED = '\033[91m'
        CEND = '\033[0m'
        info = CRED + "[" + getTodayYearMonthDayHourMinSec() + "]" + CEND + label + " " + msg
        if LogUtils.progressSignal is not None:
            LogUtils.progressSignal.emit({'text': msg, 'type': EDIT_TEXT_TYPE_ERROR})
        print(info, obj)

    progressSignal = None



    @staticmethod
    def enableProgressSignal(progressSignal):
        LogUtils.progressSignal = progressSignal

    @staticmethod
    def disableProgressSignal():
        LogUtils.progressSignal = None
