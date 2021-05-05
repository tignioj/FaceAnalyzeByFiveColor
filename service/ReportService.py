import pdfkit
from docx import Document
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn
from numpy import doc
from win32com.client import gencache, constants
from core.const_var import *
import cv2


class ReportService:
    def __init__(self, username, gender, faceColor, skinResult, glossResult, image):
        self.username = username
        self.gender = gender
        self.faceColor = faceColor
        self.skinResult = skinResult
        self.glossResult = glossResult
        self.image = image

        self.imgPath = OUTPUT_PATH + '\\DiagnoseResult.jpg'

    def getImageStream(self):
        success, encoded_image = cv2.imencode('.jpg', self.image)
        return encoded_image.tobytes()



    def wordCreate(self):
        cv2.imwrite(self.imgPath, self.image)

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
        setFontWord("姓名： %s" % self.username)
        setFontWord("性别： %s" % self.gender)
        setFontWord("面容健康诊断：")
        setFontWord("    肤色状态：%s" % self.faceColor)
        setFontWord("    光泽状态：%s" % self.glossResult)
        doc.add_picture(self.imgPath, width=Inches(2))
        setFontWord("面容健康小贴士：")
        setFontWord("    (1)%s" % self.skinResult)
        setFontWord("    (2)%s" % self.glossResult)
        doc.save(OUTPUT_PATH + '\\FaceDiagnoseResults.docx')

    def word2pdf(self):
        wordfile = OUTPUT_PATH + "\\FaceDiagnoseResults.docx"
        pdffile = OUTPUT_PATH + "\\FaceDiagnoseResults.pdf"
        word = gencache.EnsureDispatch('Word.Application')
        doc = word.Documents.Open(wordfile, ReadOnly=1)
        doc.ExportAsFixedFormat(pdffile,
                                constants.wdExportFormatPDF,
                                Item=constants.wdExportDocumentWithMarkup,
                                CreateBookmarks=constants.wdExportCreateHeadingBookmarks)
        word.Quit(constants.wdDoNotSaveChanges)

    @staticmethod
    def CreateDetectResults(face_color, face_gloss):
        if face_color == '正常': resultOfSkin = "“漂亮的皮囊”get，请继续保持！"
        if face_color == '白': resultOfSkin = "“漂亮的皮囊”get，请继续保持！"
        if face_color == '黄': resultOfSkin = "“漂亮的皮囊”get，请继续保持！"
        if face_color == '赤': resultOfSkin = "“漂亮的皮囊”get，请继续保持！"
        if face_color == '青': resultOfSkin = "“漂亮的皮囊”get，请继续保持！"
        if face_color == '黑': resultOfSkin = "最近皮肤有点差哦，请注意多休息，可以使用一些护肤品！"

        if face_gloss == '有光泽': resultOfGloss = '如果不是皮肤太油，那就是皮肤太好，羡慕~'
        if face_gloss == '无光泽': resultOfGloss = '面色红润有光泽，你不想要吗，用一些保湿的护肤品吧！'
        if face_gloss == '少光泽': resultOfGloss = '皮肤有点光泽，用一点护肤品就更好了。'

        return resultOfSkin, resultOfGloss

    def txt2pdf(self, txt_file, pdf_file):
        # 将字符串生成pdf文件
        config = pdfkit.configuration(wkhtmltopdf=r"D:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
        options = {'encoding': 'utf-8'}
        pdfkit.from_file(txt_file, pdf_file, configuration=config, options=options)

    def pdfCreate(self, filename, name, sex, face_color, face_gloss, resultOfSkin, resultOfGloss):
        with open(filename, 'w') as f:
            # f.write("姓名: %s <br>" %'孙悦')
            # f.write("性别: %s <br>" %'男')
            # f.write("面部诊断结果: %s <br>" %'正常')
            f.write("姓名: %s <br>" % name)
            f.write("性别: %s <br>" % sex)
            f.write("您的面部诊断结果: <br>")
            f.write("&emsp;&emsp; 肤色诊断结果: %s <br>" % face_color)
            f.write("&emsp;&emsp; 皮肤光泽诊断结果: %s <br>" % face_gloss)
            f.write("诊断结果分析: <br>&emsp;&emsp; %s <br>" % resultOfSkin)
            f.write("&emsp;&emsp; %s <br>" % resultOfGloss)

        with open(filename, 'r') as f:
            self.txt2pdf(f, 'DetectResults.pdf')
