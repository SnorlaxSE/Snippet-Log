import pydicom as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb


def calculate_suv(path):
    a = pd.read_file(path)
    SV = a.pixel_array
    pdb.set_trace()
    w = float(a[(0x0010, 0x1030)].value)  # (0010,1030) 患者体重 Patient's Weight [78] 
    h = float(a[(0x0010, 0x1020)].value)  # (0010,1020) 患者身高 Patient's Size [1.65] 
    t0 = float(a[(0x0009, 0x1039)].value) - float(a[(0x0008, 0x0022)].value) * 1e6  
    # (0x0009, 0x1039) 放射性药物给药时间t0 
    # (0008,0022) 图像采集时间 Acquisition Date [20170929] [20170929]
    t1 = float(a[(0x0008, 0x0032)].value)  #  (0008,0032) Acquisition Time [102808] [100308.0000] 采集时刻 t1
    t = float(a[(0x0018, 0x1242)].value)  # (0018,1242) Actual Frame Duration [300000] [1800000]  显像时间 t
    act = float(a[(0x0009, 0x1038)].value)  # (0009,1038) Private Data [183.52]  Activity 给药剂量 (MBq)
    T12 = float(a[(0x0009, 0x103f)].value)  # (0009,103F) Private Data [6588.0] 核素半衰期 T1/2
    rescale = float(a[(0x0028, 0x1053)].value)  # (0028,1053) Rescale Slope [0.138469] [4.89859e-05] 标准曲线斜率


    SUV = (SV * rescale) / (1000 / act * np.exp(-0.693/T12*(t0-t1)) * w)


    return SUV


calculate_suv(r'/Users/Captain/Desktop/ADNI/068_S_4424/ADNI_AC/2017-09-29_10_03_08.0/ADNI_068_S_4424_PT_ADNI_AC__br_raw_20171011153106882_236_S618643_I915803.dcm')
