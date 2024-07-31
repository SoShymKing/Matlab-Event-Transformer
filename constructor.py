# -*- coding:utf-8 -*-
import json
from collections import OrderedDict
import pickle
import numpy as np
import sparse
import sys

print(sys.argv[0])
evnt_file = sys.argv[1]
acc_f = evnt_file + "_acc_test.txt"
evnt_f = evnt_file + ".txt"           # back6_test.txt 
out_f = evnt_file+"_out.pckl"
evncam = open("./raw_bag/"+evnt_f, 'r')
# accbrk = open("./raw_acc/"+acc_f,'r')

fps=1000                            #fixed
chunk_nums_of_files = 20     # >0
out_file_name  = evnt_file + "_"
w,h = map(int,evncam.readline().split())
print(w)
print(h)

files=0
while(True):
    output_arr = []
    output_acc_arr = []
    output_brk_arr = []
    evn_input=str()
    for i in range(0,chunk_nums_of_files):
#         acc_input = accbrk.readline()
#         if not acc_input: break
#         acc,brk = map(float,acc_input.split())
#         output_acc_arr.append(acc)
#         output_brk_arr.append(brk)
        evn_input_p = evncam.readline()
        if not evn_input_p: break 
        past_t,past_r,past_c,past_v = evn_input_p.split()
        past_t = float(past_t)
        past_r = int(past_r)
        past_c = int(past_c)
        past_v = int(past_v)
        tmp=np.zeros((h,w,2),np.uint8)
        tmp[past_c][past_r][past_v]=1
        while(True):
            evn_input = evncam.readline()
            if not evn_input: break
            t,r,c,v = evn_input.split()
            t=float(t)
            r=int(r)
            c=int(c)
            v=int(v)
            if past_t==t:
                tmp[c][r][v]=1
            if not past_t==t:           # 0, 1 일때 1->2 일때
                # print(past_t)
                break
        # print(tmp.shape)
        output_arr.append(tmp)
        if not evn_input: break 
    if not evn_input: break  
    output_np = np.array(output_arr)
#     output_pdl_np = np.array([output_acc_arr,output_brk_arr])
    output_coo = sparse.COO.from_numpy(output_np)
    # print(out_file_name+str(files)+".pckl")
    # print(output_np.shape)
    # print(output_coo.shape)
    # print(output_coo.coords) 
    # print(output_pdl_np.shape)
    # print(output_pdl_np[0])
    with open("./output/train/"+evnt_file + "/"+out_file_name+str(files)+".pckl","wb") as fw:
        pickle.dump(output_coo, fw)
#     with open("./output/pdl/"+out_file_name+"pedal_"+str(files)+".pckl","wb") as fw:
#         pickle.dump(output_pdl_np, fw)

    # 파일 쓰기
    files = files+1
    # if not evn_input: break  

# with open("./output/back6_0.pckl","rb") as fr:
#     tmp_data = pickle.load(fr)
# print(tmp_data.coords)

evncam.close()
# accbrk.close()
