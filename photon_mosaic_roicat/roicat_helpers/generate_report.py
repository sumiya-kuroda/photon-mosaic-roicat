from fpdf import FPDF
from roicat import util, helpers
from datetime import datetime
import numpy as np
import re
from pathlib import Path
from photon_mosaic_roicat.roicat_helpers import tracking_helpers

def generate_roicat_report(dir_save:str = '', saving_pdf_location:str='roicat_report.pdf'):
    results_all = util.RichFile_ROICaT(path=dir_save + '/roicat.tracking.results_all.richfile').load()
    results_clusters = helpers.json_load(filepath=dir_save + '/roicat.tracking.results_clusters.json')
    FOVs = tracking_helpers.generate_roicat_FOVs(results_all)

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    subject, paths_s2p = extract_neuroblueprint_from_roicat(results_all)

    pdf = FPDF()
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("helvetica", style="B", size=16)
    pdf.set_x(pdf.l_margin)
    pdf.cell(40, 10, "ROICaT report")
    # pdf.cell(0, 10, f"Today's date and time: {formatted_time}", ln=True)
    pdf.cell(0, 5, f"PDF generated at {formatted_time}")
    pdf.cell(0, 5, f"subject: {subject}")
    pdf.cell(0, 5, f"n_sessions: {results_all["ROIs"]["n_sessions"]}")
    pdf.cell(0, 5, f'Number of clusters: {len(np.unique(results_clusters["labels"]))}')
    pdf.cell(0, 5, f'Number of discarded ROIs: {(np.array(results_clusters["labels"])==-1).sum()}')

    # for i, ses in enumerate(paths_s2p):
    #     path_to_roiimage = Path(dir_save) / 'visualization' / 'ROIs_aligned' / f'ROIs_aligned_{i}.png'

    #     pdf.add_page()
    #     pdf.set_x(pdf.l_margin)
    #     pdf.cell(0, 5, f"Session: {ses.name}")
    #     pdf.cell(0, 5, "Channel 1:")
    #     pdf.image(ses / 'meanImg.png', h=pdf.eph/4, w=pdf.epw/4) 
    #     pdf.cell(0, 5, "Channel 2:")
    #     pdf.image(ses / 'meanImg_chan2.png', h=pdf.eph/4, w=pdf.epw/4) 
    #     pdf.cell(0, 5, "ROICaT cluster:")
    #     pdf.image(path_to_roiimage, h=pdf.eph/4, w=pdf.epw/4) 

    pdf.output(Path(dir_save) / saving_pdf_location)

    return True # is_pdfmade

def extract_neuroblueprint_from_roicat(results_all: dict):
    paths_stat = results_all["input_data"]["paths_stat"]
    subject = get_subjectid(paths_stat[0])

    paths_s2p = []
    for path_stat in paths_stat:
        path_s2p = Path(path_stat).parent
        paths_s2p.append(path_s2p)

    return subject, paths_s2p

def get_subjectid(path:str):
    m = re.search(r'(?<=/)sub-[A-Za-z0-9]+(?=/)', path)
    subject = m.group(0) if m else None
    return subject