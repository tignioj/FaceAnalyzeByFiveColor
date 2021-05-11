from datetime import datetime


def getTodayDateTime():
    "返回格式为：'2021-05-03-02:15:13'"
    return datetime.today().strftime('%Y-%m-%d-%H:%M:%S')


def getTodayYearMonthDay():
    "返回格式为：'20210503'"
    return datetime.today().strftime('%Y%m%d')


def getTodayYearMonthDayHourMinSec():
    "返回格式为：'20210503_02_11_13'"
    return datetime.today().strftime('%Y%m%d_%H_%M_%S')


print("MyDateUtils", getTodayDateTime())
