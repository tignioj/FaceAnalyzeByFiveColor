import os

import pdfkit
from docx import Document

from utils import HistogramTools
from utils.LogUtils import LogUtils
from utils.SkinUtils import *
from utils.SkinUtils import SkinUtils
from utils.DistanceUtils import DistanceUtils
from utils.HistogramTools import getDistanceByDifferentColorSpace
from entity.ReportEntity import ReportEntity
from utils.MyDateUtils import *
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn
from numpy import doc
from win32com.client import gencache, constants
from core.const_var import *
import cv2


class ReportService:
    @staticmethod
    def wordCreate(face):
        generatePath = OUTPUT_PATH + "\\" + face.name + "\\" + getTodayYearMonthDayHourMinSec()
        if not os.path.isdir(generatePath):
            os.makedirs(generatePath)

        cv2.imwrite(generatePath + "\\" + face.name + "_face.jpg", face.facePart)
        doc = Document()
        # 标题
        doc.add_heading()
        p = doc.add_paragraph()
        title = p.add_run("面容诊断报告")
        title.font.roiName = u'微软雅黑'
        title._element.rPr.rFonts.set(qn('w:eastAsia'), u'微软雅黑')
        title.font.size = Pt(24)
        paragraph_format = p.paragraph_format
        paragraph_format.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

        def setFontWord(string):
            run = doc.add_paragraph()
            title = run.add_run(string)
            title.font.roiName = u'微软雅黑'
            title._element.rPr.rFonts.set(qn('w:eastAsia'), u'微软雅黑')
            paragraph_format = run.paragraph_format
            paragraph_format.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

        # 内容
        setFontWord("姓名： %s" % face.name)
        setFontWord("性别： %s" % face.gender)
        doc.add_picture(face.imgPath, width=Inches(2))
        doc.save(generatePath + "\\" + face.name + "_report.docx")
        ReportService._word2pdf(generatePath, generatePath + "\\" + face.name + "_report.docx", )

    @staticmethod
    def _word2pdf(wordFolder, wordPath):
        wordfile = wordPath
        pdffile = wordFolder + "\\report.pdf"
        word = gencache.EnsureDispatch('Word.Application')
        doc = word.Documents.Open(wordfile, ReadOnly=1)
        doc.ExportAsFixedFormat(pdffile,
                                constants.wdExportFormatPDF,
                                Item=constants.wdExportDocumentWithMarkup,
                                CreateBookmarks=constants.wdExportCreateHeadingBookmarks)
        word.Quit(constants.wdDoNotSaveChanges)

    @staticmethod
    def generateReports(detectedFaces):
        LogUtils.log("reportService", "start generate report from faces...")
        reports = []
        for face in detectedFaces:
            report = ReportEntity()
            report.gender = face.gender
            report.username = face.name
            report.imgPath = face.imgPath
            report.facePart = face.facePart
            report.roiDict = face.landMarkROIDict
            report.roiRGBDict = {}
            report.roiHSVDict = {}
            report.roiYCrCbDict = {}
            report.roiHistograms = {}
            report.roiLabDict = {}
            report.roiColorResults = {}
            # fig = plt.figure(figsize=(5, 5))
            fig = plt.figure()
            LogUtils.log("reportService", "正在获取ROI不同颜色空间报告...")
            items = report.roiDict.items()
            currentProgress = 50
            maxProgress = 100
            step = int((maxProgress-currentProgress) / len(items))
            for (name, roi) in items:
                nameCN = FACIAL_LANDMARKS_NAME_DICT[name]
                LogUtils.log("ReportService", "正在绘制" + nameCN + "的颜色空间分布曲线", progress=currentProgress)
                currentProgress += step
                fig.clear()
                report.roiRGBDict[name] = (SkinUtils.skinHistogram(fig, roi.img, COLOR_MODE_RGB))
                fig.clear()
                report.roiHSVDict[name] = (SkinUtils.skinHistogram(fig, roi.img, COLOR_MODE_HSV))
                fig.clear()
                report.roiLabDict[name] = (SkinUtils.skinHistogram(fig, roi.img, COLOR_MODE_Lab))
                fig.clear()
                report.roiYCrCbDict[name] = (SkinUtils.skinHistogram(fig, roi.img, COLOR_MODE_YCrCb))
                fig.clear()
                LogUtils.log("ReportService", "查询" + nameCN+ "的肤色中...")
                report.roiHistograms[name], report.roiColorResults[name] = (
                    getDistanceByDifferentColorSpace(fig, roi.img, name))

            report.drawImg = face.drawImg
            LogUtils.log("reportService", "获取ROI不同颜色空间报告完成！")
            # report.faceColor = SkinUtils.roiTotalColorDetect(report.rois)
            # report.skinResult = SkinUtils.getResultByColor(report.rois)
            reports.append(report)

        LogUtils.log("reportService", "reports generate finished!")
        return reports
