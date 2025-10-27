from fpdf import FPDF
from roicat import util, helpers
from datetime import datetime
import numpy as np
import re
from pathlib import Path
from photon_mosaic_roicat.roicat_helpers import tracking_helpers


class PDFReport(FPDF):
    # https://qiita.com/nassy20/items/a716b05ac34b7154012f
    def add_reportdate(self):  # noqa
        base_x = -70
        base_y = 10
        self.set_xy(base_x, base_y)
        self.cell(txt=f"PDF generated at : {format(datetime.now(), "%Y-%m-%d %H:%M:%S")}")

    def add_reporttitle(self, txt_title='This is title'):  # noqa
        base_x = 95
        base_y = 30
        self.set_font(size=20)
        self.set_xy(95, 30)
        self.set_xy(base_x, base_y)
        self.cell(txt=txt_title, align="C")

    def add_reporttxt(self, txt: str, base_x = 10, base_y = 70):  # noqa
        self.set_xy(base_x, base_y)
        self.cell(txt=txt)


def generate_roicat_report(dir_save:str = '', saving_pdf_location:str='roicat_report.pdf'):
    results_all = util.RichFile_ROICaT(path=dir_save + '/roicat.tracking.results_all.richfile').load()
    results_clusters = helpers.json_load(filepath=dir_save + '/roicat.tracking.results_clusters.json')
    FOVs = tracking_helpers.generate_roicat_FOVs(results_all)

    subject, paths_s2p = extract_neuroblueprint_from_roicat(results_all)

    print('Generating PDF report for ROICaT results...')
    pdf = PDFReport()
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_font("helvetica", style="B", size=10)

    pdf.add_page()
    pdf.add_reportdate()
    pdf.add_reporttitle('ROICaT Report')

    pdf.add_reporttxt(f"subject: {subject}", base_y = 70)
    pdf.add_reporttxt(f"n_sessions: {results_all["ROIs"]["n_sessions"]}", base_y = 75)
    pdf.add_reporttxt(f'Number of clusters: {len(np.unique(results_clusters["labels"]))}', base_y = 80)
    pdf.add_reporttxt(f'Number of discarded ROIs: {(np.array(results_clusters["labels"])==-1).sum()}', base_y = 85)

    for i, ses in enumerate(paths_s2p):
        path_to_roiimage = Path(dir_save) / 'visualization' / 'ROIs_aligned' / f'ROIs_aligned_{i}.png'

        pdf.add_page()
        pdf.add_reporttxt(f"Session: {ses.name}", base_y = 10)
        pdf.add_reporttxt("Channel 1:", base_y = 15)
        pdf.image(ses / 'funcimg' / 'suite2p' / 'plane0' / 'meanImg.png', h=40, w=40, x=10, y=20) 

        pdf.add_reporttxt("Channel 2:", base_y = 65) # add 45
        pdf.image(ses / 'funcimg' / 'suite2p' / 'plane0' / 'meanImg_chan2.png', h=40, w=40, x=10, y=70) 

        pdf.add_reporttxt("ROICaT cluster:", base_y = 115)
        pdf.image(path_to_roiimage, h=40, w=40, x=10, y=120) 

    pdf.output(Path(dir_save) / saving_pdf_location)

    return True # is_pdfmade

def extract_neuroblueprint_from_roicat(results_all: dict):
    paths_stat = results_all["input_data"]["paths_stat"]
    subject = get_subjectid(paths_stat[0])

    paths_s2p = []
    for path_stat in paths_stat:
        path_s2p = Path(path_stat).parent.parent.parent.parent
        paths_s2p.append(path_s2p)

    return subject, paths_s2p

def get_subjectid(path:str):
    m = re.search(r'(?<=/)sub-[A-Za-z0-9]+(?=/)', path)
    subject = m.group(0) if m else None
    return subject