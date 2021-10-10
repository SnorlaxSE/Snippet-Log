# Prepare: xxxx_UR_metadata (folder) ; BAI - PET NMRC FDG Summaries [ADNI1,GO,2,3].csv
# Input: Image UID 
# Output: 1) 提取相关UID的全部信息，make a new csv
#         2) ...
# 
import os
import csv
import pdb 
import pandas as pd
import numpy as np 
import  xml.dom.minidom


folder_name = 'FDG_PET_UR_metadata'
ur_metadata_path = './{}'.format(folder_name)
analysis_path = './BAI-PET_NMRC_FDG_Summaries.csv'
output_path = './{}_FDG-NMRC.csv'.format(folder_name)


def get_PTID_dateAcquired_subjectAge(xml_filepath):
    #打开xml文档
    dom = xml.dom.minidom.parse(xml_filepath)
    #得到文档元素对象
    root = dom.documentElement
    PTID_list = root.getElementsByTagName('subjectIdentifier')
    dateAcquired_list = root.getElementsByTagName('dateAcquired')
    subjectAge_list = root.getElementsByTagName('subjectAge')
    researchGroup_list = root.getElementsByTagName('researchGroup')
    PTID = PTID_list[0].firstChild.data
    dateAcquired = dateAcquired_list[0].firstChild.data
    subjectAge = subjectAge_list[0].firstChild.data
    researchGroup = researchGroup_list[0].firstChild.data
    # pdb.set_trace()
    return PTID, dateAcquired, subjectAge, researchGroup


def extract_imageuid(meta_filename):
    # ex. ADNI_002_S_0295_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution_S111104_I240519.xml
    filename_prefix = meta_filename.split('.')[0]
    uid = filename_prefix.split('_')[-1][1:]
    ptid = '_'.join(filename_prefix.split('_')[1:4])
    return uid, ptid

def read_csv(filename, iuid):
    info_list = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            # pdb.set_trace()
            if iuid in row:
                info_list.append(row)
    return info_list

item_list = []
for root, dirs, files in os.walk(ur_metadata_path):
    print(len(files))
    for filename in files:
        # Get Image UID 
        filepath = os.path.join(root, filename)
        PTID, dateAcquired, subjectAge, researchGroup = get_PTID_dateAcquired_subjectAge(filepath)

        uid, ptid = extract_imageuid(filename)

        # Get relevant Item
        info_list = read_csv(analysis_path, uid)
        if info_list:
            for info_item in info_list:
                info_item.insert(1, ptid)
                info_item.insert(2, subjectAge)
                info_item.insert(3, dateAcquired)
                info_item.insert(4, researchGroup)
                item_list.append(info_item)
                # pdb.set_trace()
        else:
            print("{} have no relevant items.".format(filename))
    break

print(len(item_list))
## save as csv
# Get csv header
col_name = []
with open(analysis_path) as f:
    reader = csv.reader(f)
    # pdb.set_trace()
    for row in reader:
        col_name = row
        break
# pdb.set_trace()
col_name.insert(1, 'PTID')
col_name.insert(2, 'subjectAge')
col_name.insert(3, 'dateAcquired')
col_name.insert(4, 'researchGroup')

df = pd.DataFrame(columns=col_name, data=item_list)
df.to_csv(output_path, encoding='utf-8', index=False)
print("all done")


# # check csv row_num
# check_path = r'/Users/Captain/Desktop/metadata_analysis/MCI_PET_UR_metadata_FDG-Analysis.csv'
# csv_reader = csv.reader(open(check_path, encoding='utf-8')) # 有的文件是utf-8编码
# print(np.array(list(csv_reader)).shape[0])
