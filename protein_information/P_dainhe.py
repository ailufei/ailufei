import subprocess
import re

def run_apbs(input_file):
    # 运行APBS并将标准输出重定向到临时文件
    with open("temp_output.txt", "w") as output_file:
        subprocess.run(["apbs", input_file], stdout=output_file)


def extract_total_energy_from_output(output_file):
    with open(output_file, "r") as file:
        lines = file.readlines()
    total_energy = None
    for line in lines:
        if "Total electrostatic energy" in line:
            total_energy = line.split("=")[-1].strip()
            sum= re.findall(r"\d+\.?\d*", total_energy)
            break

    return sum[0]


def total_energy(pro):
    pdb_file = '/home/ntu/Documents/MY_Module/protein_information/PDB/KIBA_PDB/' + str(pro) + '.pdb'
    # 转换 PDB 文件为 PQR 文件
    pqr_file = "/home/ntu/Documents/MY_Module/protein_information/Total_electrostatic_energy/1XYZ.pqr"
    pdb2pqr_command = f"pdb2pqr --ff=amber --whitespace {pdb_file} {pqr_file}"

    subprocess.run(pdb2pqr_command, shell=True)

    # 创建 APBS 输入文件
    input_file = "/home/ntu/Documents/MY_Module/protein_information/Total_electrostatic_energy/input.in"
    with open(input_file, "w") as f:
        f.write("read\n")
        f.write("    mol pqr {}\n".format(pqr_file))
        f.write("end\n\n")
        f.write("elec name com\n")
        f.write("    mg-auto\n")
        f.write("    dime 65 65 65\n")
        f.write("    cglen 81.6 81.6 81.6\n")
        f.write("    fglen 81.6 81.6 81.6\n")
        f.write("    fgcent 41.621 32.573 46.6415\n")
        f.write("    cgcent 41.621 32.573 46.6415\n")
        f.write("    mol 1\n")
        f.write("    npbe\n")
        f.write("    bcfl mdh\n")
        f.write("    srfm smol\n")
        f.write("    chgm spl4\n")
        f.write("    pdie 2.0\n")
        f.write("    sdens 10\n")
        f.write("    sdie 78.54\n")
        f.write("    srad 1.4\n")
        f.write("    temp 300.0\n")
        f.write("    mol 1\n")
        f.write("    calcenergy total\n")

        f.write("    write pot flat /home/ntu/Documents/MY_Module/protein_information/Total_electrostatic_energy/output\n")
        f.write("    end\n")
        f.write("    swin 0.3\n")


        f.write("    ion 1 charge 1 radius 2.0\n")
        f.write("    ion 2 charge -1 radius 2.0\n")


        f.write("    calcforce  no\n")
        f.write("    calcenergy comps\n")
        f.write("end\n")

        f.write("    print elecEnergy com end\n")
    # 运行 APBS 命令
    #apbs_command = f"apbs {input_file}"
    #result=subprocess.run(apbs_command,shell=True)
        # 运行APBS

    run_apbs(input_file)

# 从输出文件中提取总的静电势能值
    total_energy = extract_total_energy_from_output("/home/ntu/Documents/MY_Module/protein_information/Total_electrostatic_energy/temp_output.txt")

        #if total_energy is not None:
            #print("总静电势能:", total_energy)
        #else:
           # print("未找到总的静电势能。请检查APBS的输出和提取代码是否正确。")

    return total_energy

'''
import json
from collections import OrderedDict
path='/home/ntu/Documents/MY_Module/fusion/data_kiba/'
proteins = json.load(open(path + "proteins.txt"), object_pairs_hook=OrderedDict)
XT = []
XT_key=[]
for t in proteins.keys():
    XT.append(proteins[t])
    XT_key.append(t)

    a=total_energy(t)
    print(t)
    print(a)
'''