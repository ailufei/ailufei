from Bio.SeqUtils.ProtParam import ProteinAnalysis
from protein_curvature import CURVE
from P_dainhe import total_energy
def count_atoms(chformula):
    atoms = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K','L','M','N','P','Q','R','S','T','V','W','Y']
    atoms_count = {atom: 0 for atom in atoms}
    for atom in atoms_count.keys():
        count = chformula.count(atom)
        atoms_count[atom] = count
    return list(atoms_count.values())


def get_protein_information(proteins):
    # 定义蛋白质序列
    i=0
    for t in proteins.keys():
        i=i+1
        print(t)
        protein = ProteinAnalysis(proteins[t])         # 创建蛋白质分析对象
        hydrophobicity = protein.gravy()                    # 计算蛋白质氨基酸的疏水性指标
        molecular_weight = protein.molecular_weight()        # 计算蛋白质的分子量
        isoelectric_point = protein.isoelectric_point()   # 计算蛋白质的等电点

        pro_info_list = [molecular_weight,isoelectric_point,hydrophobicity,float(CURVE(t)),total_energy(t)]

        pro_reduse=count_atoms(proteins[t])#[分子量,等电点,疏水性指标]
        pro_info_list=pro_info_list+pro_reduse
        print(pro_info_list)
        # 打印结果
        #print("蛋白质分子量：", molecular_weight)
        #print("蛋白质等电点：", isoelectric_point)
    return pro_info_list


