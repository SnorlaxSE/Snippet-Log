from multiprocessing import Pool

def foo(a, target_txt):
    with open(a, "r") as f:
        contents = f.readlines()
        with open(target_txt, "a") as f2:
            f2.writelines(contents)

p = Pool(processes=4)

txt_list = [f"./{a+1}.txt" for a in range(10)]  
target_txt = "./all.txt"
# print(txt_list)

p.starmap(foo, zip(txt_list, [target_txt]*10))
p.close()
p.join()  # start work
