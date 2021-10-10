def list2txt(list, txt_file):
    txt = open(txt_file, mode='a+')
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