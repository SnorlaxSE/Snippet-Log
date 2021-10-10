import os 
import pdb
import argparse
 

def extract_info(filename):
    # AD_114_S_0979_87.4_20080224_N3_MNI_ht_norm.nii.gz

    ptid = '_'.join(filename.split('_')[1:4])
    age = filename.split('_')[4]
    return ptid, age


def find_relevant_pet(pet_list, mri_file):

    mri_ptid, mri_age = extract_info(mri_file)

    conform_dicts = {}
    for pet_file in pet_list:
        pet_ptid, pet_age = extract_info(pet_file)
        if pet_ptid == mri_ptid:
            diff_age = round(abs(float(pet_age)-float(mri_age)), 2)
            conform_dicts[diff_age] = pet_file

    conform_dicts=sorted(conform_dicts.items(),key=lambda x:x[0])
    # print(conform_dicts)
    if conform_dicts:
        return conform_dicts[0][1]
    else:
        print(mri_file)
        pdb.set_trace()

def list2txt(list, txt_file):
    txt = open(txt_file, mode='a+')
    list.sort()
    for item in list:
        txt.write("{}\n".format(item))


def txt2list(txt_file):
    
    list = []
    with open(txt_file, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            # print(line)
            list.append(line)
    
    return list

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-c", "--classes", type=str, help="AD/MCI/CN")
    parser.add_argument("-mmri", "--modalitymri", type=str, help="MRI_HT/CSF_PET/GM_PET/WM_PET etc.")
    parser.add_argument("-mpet", "--modalitypet", type=str, help="MRI_HT/CSF_PET/GM_PET/WM_PET etc.")
    args = parser.parse_args()

    modality_mri = args.modalitymri
    modality_pet = args.modalitypet
    classes = args.classes
    output_txt_mri = 'HCI_MRI_{}_total_skull_MNI_HT_SEG_{}.txt'.format(classes, modality_mri)
    output_txt_pet = 'HCI_MRI_{}_total_skull_MNI_HT_SEG_{}.txt'.format(classes, modality_pet)
    redu_txt = 'HCI_MRI_{}_total_skull_MNI_HT_SEG_{}_{}.txt'.format(classes, modality_mri, modality_pet)

    mri_list = txt2list(output_txt_mri)
    pet_list = txt2list(output_txt_pet)
    print("mri_list: ", len(mri_list))
    print("pet_list: ", len(pet_list))

    rp_list = []
    for mri_file in mri_list:
        relevant_pet = find_relevant_pet(pet_list, mri_file)   
        rp_list.append(relevant_pet)
    
    print("rp_list: ", len(rp_list))

    for rp_file in rp_list:
        try:
            # pdb.set_trace()
            pet_list.remove(rp_file)
        except:
            pdb.set_trace()
            continue
    
    print("pet_list: ", pet_list)
    print("pet_list: ", len(pet_list))

    list2txt(pet_list, redu_txt)