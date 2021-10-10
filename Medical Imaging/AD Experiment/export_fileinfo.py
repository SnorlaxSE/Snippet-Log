import os
import pdb
import pandas as pd

import argparse
 
    
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-c", "--classes", type=str, help="AD/MCI/CN")
parser.add_argument("-m", "--modality", type=str, help="MRI_HT/CSF_PET/GM_PET/WM_PET etc.")
args = parser.parse_args()

modality = args.modality
classes = args.classes
classes_dataset = 'HCI_MRI_{}_total_skull_MNI_HT_SEG_{}'.format(classes, modality)
output_path = 'HCI_MRI_{}_total_skull_MNI_HT_SEG_{}.csv'.format(classes, modality)
output_txt = 'HCI_MRI_{}_total_skull_MNI_HT_SEG_{}.txt'.format(classes, modality)


# txt
log = open(output_txt, mode='a+')
for filename in os.listdir(classes_dataset):
    if 'nii' in filename:
        # AD_003_S_1257_85.3_20070521_UR_N3_CSF_UR.nii.gz
        log.write("{}\n".format(filename))

        
def record_csv():
    info_list = []
    for filename in os.listdir(classes_dataset):
        if 'nii' in filename:
            # AD_003_S_1257_85.3_20070521_UR_N3_CSF_UR.nii.gz
            ptid = '_'.join(filename.split('_')[1:4])
            age = filename.split('_')[4]
            date = filename.split('_')[5]
            info_list.append([ptid, age, date])

    print(len(info_list))

    col_name = ['PTID', 'AGE', 'DATE']


    df = pd.DataFrame(columns=col_name, data=info_list)
    df.to_csv(output_path, encoding='utf-8', index=False)
    print("all done")