import os

import pdfkit
from docx import Document
from utils.SkinUtils import *
from utils.SkinUtils import SkinUtils
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
        reports = []
        for face in detectedFaces:
            report = ReportEntity()
            report.gender = face.gender
            report.username = face.name
            report.imgPath = face.imgPath
            report.facePart = face.facePart
            report.rois = face.landMarkROI
            report.roisRGB = []
            report.roisHSV = []
            report.roisYCrCb = []
            report.roisLab = []
            fig = plt.figure()
            for roi in report.rois:
                fig.clear()
                report.roisRGB.append(SkinUtils.skinHistogram(fig, roi.img, COLOR_MODE_RGB))
                fig.clear()
                report.roisHSV.append(SkinUtils.skinHistogram(fig, roi.img, COLOR_MODE_HSV))
                fig.clear()
                report.roisLab.append(SkinUtils.skinHistogram(fig, roi.img, COLOR_MODE_Lab))
                fig.clear()
                report.roisYCrCb.append(SkinUtils.skinHistogram(fig, roi.img, COLOR_MODE_YCrCb))
            report.drawImg = face.drawImg
            # report.faceColor = SkinUtils.roiTotalColorDetect(report.rois)
            # report.skinResult = SkinUtils.getResultByColor(report.rois)

            reports.append(report)

        return reports
