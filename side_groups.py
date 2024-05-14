#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time, os, pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import collections
import itertools
import rdkit
print(rdkit.__version__)
from rdkit import rdBase,Chem
from rdkit.Chem import AllChem,Draw, Descriptors
from collections import defaultdict##########let me seem
#from rdkit.Chem import AllChem
#from rdkit.Chem import Descriptors
#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

#from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import rdqueries
#from rdkit import Chem
#from rdkit.Chem.Draw import IPythonConsole
#from rdkit.Chem import Draw
#IPythonConsole.ipython_useSVG=True  #< set this to False if you want PNGs instead of SVGs
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms
RDLogger.DisableLog('rdApp.*')
from config import dir_smi_log

# In[18]:


operations_backbone = {

     'Biphenyl addition': '[cH&r6,CH&r5:1]>>[#6:1](-[c]1[ch]ccc[ch]1)',  #1
     #'reverse_Biphenyl addition': '[c&r6,C&r5:1](-[c]1[ch]ccc[ch]1)>>[*:1]',   #2

     #'Linkage doublebond': '[ch&r6,CH1&r5:1]>>[*:1](-C=C-c1ccccc1)', #13
     #'reverse_Linkage doublebond': '[c&r6,C&r5:1](-[C!R]=[C!R]-c1ccccc1)>>[ch&r6,CH1&r5:1]', #14
     'Linkage triplebond': '[ch&r6,CH1&r5:1]>>[*:1](-C#C-c1ccccc1)', #15
     #'reverse_Linkage triplebond': '[c&r6,C&r5:1](-C#C-c1ccccc1)>>[ch&r6,CH1&r5:1]', #16
     #'Phenylamine linkage': '[ch&r6,CH1&r5:1]>>[*:1](-N-c1ccccc1)', #33
     #'reverse_Phenylamine linkage': '[c&r6,C&r5:1](-[N!R]-c1ccccc1)>>[*:1]', #34
     #'Phenol linkage': '[ch&r6,CH1&r5:1]>>[*:1](-O-c1ccccc1)',
       '6-ring annelation':'[cH&r6:1][cH&r6:2]>>[c:1]2cccc[c:2]2',  #3
     #'reverse_6-ring annelation':'[r6:0][c:1]2[ch][ch][ch][ch][c:2]2[r6:3]>>[*:0][c:1][c:2][*:3]',  #4

     '5-ring annelation':'[ch:1]2[c:2][c:3][c:4][c:5][c:6]2>>C1CC[ch:1]2[c:2]1[c:3][c:4][c:5][c:6]2',


}
operations_sidegroup = {
    'N 6-ring substitution': '[ch&r6:0]>>[nr6:0]', #31
     #'reverse_N 6-ring substitution': '[nr6:0]>>[ch&r6:0]', #32

  #   '6-ring annelation':'[cH&r6:1][cH&r6:2]>>[c:1]2cccc[c:2]2',  #3
     #'reverse_6-ring annelation':'[r6:0][c:1]2[ch][ch][ch][ch][c:2]2[r6:3]>>[*:0][c:1][c:2][*:3]',  #4

  #   '5-ring annelation':'[ch:1]2[c:2][c:3][c:4][c:5][c:6]2>>C1CC[ch:1]2[c:2]1[c:3][c:4][c:5][c:6]2',

    'S 6-ring substitution': '[ch&r6:0]>>[sr6:0]',

    # 5-ring: oxygen
    'O 5-ring CH2 substitution': '[#6H2r5:1]>>[O:1]', #19
    #'reverse_O 5-ring CH2 substitution': '[n,c:0]1[n,ch:1][o,O:2][n,c:3][n,c:4]1>>[*:0]1=[*:1][CH2:2][*:3]=[*:4]1', #20
    # 5-ring: sulfur
    'S 5-ring CH2 substitution': '[#6H2r5:1]>>[S:1]', # 23
    #'reverse_S 5-ring CH2 substitution': '[n,c:0]1[n,ch:1][s,S:2][n,c:3][n,c:4]1>>[*:0]1=[*:1][CH2:2][*:3]=[*:4]1', #24

    #'Dithiole formation': '[CH,ch:0]1=[C,c:1][C,c:2]=[CH,ch:3][*:4]1>>[S:0]1[*:1]=,:[*:2][S:3][*:4]1', #25
    #'reverse_Dithiole formation': '[O,o:0]1[r5:1]-,=,:[r5:2][O,o:3][*:4]1>>[CH:0]1=[C:1][C:2]=[CH:3][*:4]1', #26

    # 5-ring: nitrogen
    'N 5-ring CH2 substitution': '[#6H2r5:1]>>[NH:1]', #27
    #'reverse_N 5-ring CH2 substitution': '[n,c:0]1[n,ch:1][NH:2][n,c:3][n,c:4]1>>[*:0]1=[*:1][CH2:2][*:3]=[*:4]1', #28
    'N 5-ring CH substitution': '[chr5,CH1r5:1]>>[n:1]', #29
    #'reverse_N 5-ring CH substitution': '[nr5:0]>>[ch&r5:0]', #30

    #'Boronic acid addition': '[cH&r6,CH&r5:1]>>[*:1](-B(O[H])(O[H]))',
    #'aldehyde group addition': '[cH&r6,CH&r5:1]>>[*:1](-[CH]=O)',
    #'amino group addition':'[cH&r6,CH&r5:1]>>[*:1](-[NH2])',
    'carbonyl group addition': '[cH&r6,CH&r5:1]>>[*:1](=O)',
    #'carbonyl group addition': '[ch:1]1[c:2][c:3][c:4][c:5][c:6]1>>O=[ch:1]1[c:2][c:3][c:4][c:5][c:6]1',
    #'hydroxyl group addition': '[cH&r6,CH&r5:1]>>[*:1](-O[H])',
    #'methoxy group addition': '[cH&r6,CH&r5:1]>>[*:1](-OC)',

    '1-N(CH2CH3)2': '[cH&r6,CH&r5:1]>>[*:1](-N(-CC)CC)', #1
    '2-NH-C(=S)-NH2': '[cH&r6,CH&r5:1]>>[*:1](-N-C(=S)[NH2])', #2
    '3-C(=S)-NH2': '[cH&r6,CH&r5:1]>>[*:1](-C(=S)N)', #3
    '4-SH': '[cH&r6,CH&r5:1]>>[*:1](-[SH])', #4
    #'5-CCl3': '[cH&r6,CH&r5:1]>>[*:1](-C(-Cl)(-Cl)-Cl)',
    #'6-I': '[cH&r6,CH&r5:1]>>[*:1](-I)',
    '7-CH2-NO2': '[cH&r6,CH&r5:1]>>[*:1](-C[N+](=O)-[O-])',
    '8-CH2-CH=CH2': '[cH&r6,CH&r5:1]>>[*:1](-CC=C)',
    '9-S-CH3': '[cH&r6,CH&r5:1]>>[*:1](-SC)',
    '10-C#CH':'[cH&r6,CH&r5:1]>>[*:1](-C#C)',
    #'11-C(=O)-CH2-Cl': '[cH&r6,CH&r5:1]>>[*:1](-C(=O)CCl)',
    '12-CH=O': '[cH&r6,CH&r5:1]>>[*:1](-C=O)',
    '13-C#N': '[cH&r6,CH&r5:1]>>[*:1](-C(#N))',
    '14-(CH2)2-CH3': '[cH&r6,CH&r5:1]>>[*:1](-CCC)',
    '15-CH2-C#CH': '[cH&r6,CH&r5:1]>>[*:1](-CC#C)',
    '16-N(CH3)2': '[cH&r6,CH&r5:1]>>[*:1](-N(-C)C)',
    #'17-Br': '[cH&r6,CH&r5:1]>>[*:1](-Br)',
    '18-(CH2)5-CH3': '[cH&r6,CH&r5:1]>>[*:1](-CCCCCC)',
    '19-C(=O)-CH3': '[cH&r6,CH&r5:1]>>[*:1](-C(=O)C)',
    '20-(CH2)3-CH3': '[cH&r6,CH&r5:1]>>[*:1](-CCCC)',
    #'21-CH2-Br': '[cH&r6,CH&r5:1]>>[*:1](-CBr)',
    '22-CH2-C#N': '[cH&r6,CH&r5:1]>>[*:1](-CC#N)',
    '23-S(=O)-CH3': '[cH&r6,CH&r5:1]>>[*:1](-S(=O)C)',
    '24-CH-CH2': '[cH&r6,CH&r5:1]>>[*:1](-C=C)',
    '25-C-(=O)-NH-CH3': '[cH&r6,CH&r5:1]>>[*:1](-C(=O)NC)',
    #'26-Cl': '[cH&r6,CH&r5:1]>>[*:1](-Cl)',
    '27-C-(CH3)3': '[cH&r6,CH&r5:1]>>[*:1](-C(C)(C)C)',
    '28-NO2': '[cH&r6,CH&r5:1]>>[*:1](-[N+](=O)-[O-])',
    '29-O-CH-(CH3)2': '[cH&r6,CH&r5:1]>>[*:1](-OC(C)C)',
    '30-C(=O)-NH2': '[cH&r6,CH&r5:1]>>[*:1](-C(=O)N)',
    '31-CH2-COOH': '[cH&r6,CH&r5:1]>>[*:1](-CC(=O)O)',
    '32-O-(CH2)2-CH3': '[cH&r6,CH&r5:1]>>[*:1](-OCCC)',
    '33-CH3': '[cH&r6,CH&r5:1]>>[*:1](-C)',
    '34-NH-CH3': '[cH&r6,CH&r5:1]>>[*:1](-NC)',
    #'35-CH2-Cl': '[cH&r6,CH&r5:1]>>[*:1](-CCl)',
    '36-O-(CH2)3-CH3': '[cH&r6,CH&r5:1]>>[*:1](-OCCCC)',
    '37-O-CH2-C#CH': '[cH&r6,CH&r5:1]>>[*:1](-OCC#C)',
    '38-OH': '[cH&r6,CH&r5:1]>>[*:1](-O)',
    '39-COOH': '[cH&r6,CH&r5:1]>>[*:1](-C(=O)O)',
    '40-C(=O)-O-CH3': '[cH&r6,CH&r5:1]>>[*:1](-C(=O)OC)',
    '41-CH=N-OH': '[cH&r6,CH&r5:1]>>[*:1](-C=NO)',
    '42-O-CH3': '[cH&r6,CH&r5:1]>>[*:1](-OC)',
    '43-CH2-CH3': '[cH&r6,CH&r5:1]>>[*:1](-CC)',
    #'44-F': '[cH&r6,CH&r5:1]>>[*:1](-F)',
    '45-CH2-C(=O)-O-CH3': '[cH&r6,CH&r5:1]>>[*:1](-CC(=O)OC)',
    '46-O-CH2-CH3': '[cH&r6,CH&r5:1]>>[*:1](-OCC)',
    '47-CH(CH3)2': '[cH&r6,CH&r5:1]>>[*:1](-C(C)C)',
    '48-C(=O)-CH2-CH3': '[cH&r6,CH&r5:1]>>[*:1](-C(=O)CC)',
    '49-C(=NO)-CH3': '[cH&r6,CH&r5:1]>>[*:1](-C(=NO)C)',
    '50-CH2-N(CH3)2': '[cH&r6,CH&r5:1]>>[*:1](-CN(C)C)',
    #'51-CF3': '[cH&r6,CH&r5:1]>>[*:1](-C(F)(F)F)',
    '52-CH2(-OH)-CH3': '[cH&r6,CH&r5:1]>>[*:1](-C(O)C)',
    '53-CH2-CH=C(CH3)2': '[cH&r6,CH&r5:1]>>[*:1](-CC=C(C)C)',
    '54-CH2-OH': '[cH&r6,CH&r5:1]>>[*:1](-CO)',
    '55-C(CH2)-OH': '[cH&r6,CH&r5:1]>>[*:1](-C(C)(C)O)',
    '56-CH2-C(=O)-O-CH3': '[cH&r6,CH&r5:1]>>[*:1](-CC(=O)OCC)',
    #'57-B(OH2)':'',
    '57-NH-CH=O': '[cH&r6,CH&r5:1]>>[*:1](-NC=O)',
    '58-O-C(=O)-CH3': '[cH&r6,CH&r5:1]>>[*:1](-OC(=O)C)',
    '59-NH-C(=O)-CH3': '[cH&r6,CH&r5:1]>>[*:1](-NC(=O)C)',
    '60-(CH2)2-OH': '[cH&r6,CH&r5:1]>>[*:1](-CCO)',
    '61-NH2': '[cH&r6,CH&r5:1]>>[*:1](-N)',
    '62-SO2-NH2': '[cH&r6,CH&r5:1]>>[*:1](-S(=O)(=O)N)',
    '63-C(=O)-O-C(CH3)3': '[cH&r6,CH&r5:1]>>[*:1](-C(=O)OC(C)(C)(C))',
    '64-SO2-CH3': '[cH&r6,CH&r5:1]>>[*:1](-S(=O)(=O)C)',
    '65-NH-C(=O)-NH2': '[cH&r6,CH&r5:1]>>[*:1](-NC(=O)N)',
    '66-N=O': '[cH&r6,CH&r5:1]>>[*:1](-N=O)',
    '67-C(=O)-NH-NH2': '[cH&r6,CH&r5:1]>>[*:1](-C(=O)NN)',

    '-C(=O)-O-CH2-CH3': '[cH&r6,CH&r5:1]>>[*:1](-C(=O)OCC)',

}


# In[83]:


data=pd.read_json('df_population_5.json',orient='split')
data=data.loc[data.added_in_round>0]
df_initial=pd.read_json('df_G1.json',orient='split')
df_initial.drop_duplicates(subset=['smi'],inplace=True)
#print(df_initial)
#data=data.iloc[-40:]
cwd=os.getcwd()
#os.chdir('smi_logs')
os.chdir(dir_smi_log)

operations=list(operations_backbone.values())+list(operations_sidegroup.values())


# In[86]:


side_groups=[]
for op in operations[10:]:
    side_groups.append(op.split('>>')[1][6:-1])

df_sidegroups=pd.DataFrame(columns=['idx']+side_groups)
print(df_sidegroups)

for idx, row in data.iterrows():
    print(row.idx, row.operation)
    groups=[]
    if operations.index(row.operation) > 9:
        side_group=row.operation.split('>>')[1][6:-1]
        if side_group in side_groups:
            groups.append(side_group)
    current_smi=row.mol_last_gen
    while current_smi!='c1ccccc1':
        #print('1',current_smi)
        #os.chdir('smi_logs')
        if os.path.isfile('{}.log'.format(current_smi) ):
            with open ('{}.log'.format(current_smi), 'r') as f:
                #print('2',f.readlines()[0])
                #print('3',)
                lines=f.readlines()
                #print('3',lines)
                #print(lines[0].split('____')[0])
                op_last=lines[0].split('____')[0]
                mol_last=lines[0].split('____')[1]
                current_smi=mol_last
                #print(current_smi)
                if operations.index(op_last) > 9:
                #print(side_group)
                    side_group=op_last.split('>>')[1][6:-1]
                    if side_group in side_groups:
                        groups.append(side_group)
        else:
            #print(df_initial.loc[df_initial.smi==current_smi,'mol_last_gen'].values[0] )
            side_group=df_initial.loc[df_initial.smi==current_smi,'operation'].values[0].split('>>')[1][6:-1]
            print(side_group)
            if side_group in side_groups:
                groups.append(side_group)
            current_smi=df_initial.loc[df_initial.smi==current_smi, 'mol_last_gen'].values[0]
    #print('groups',groups)
    
    #for group in df_sidegroups.loc[idx]:
    #    print(group)
    count_group=[]
    for side_group in side_groups:
        count_group.append(groups.count(side_group))
    #print(count_group)
    df_sidegroups.loc[len(df_sidegroups)] = [row.idx]+count_group
#       if groups.count(group)!=0:
#            df_s
#df_sidegroups=pd.DataFrame(columns=side_groups)
#os.chdir('..')
os.chdir(cwd)
df_sidegroups.to_json('df_sidegroups.json',orient='split')





