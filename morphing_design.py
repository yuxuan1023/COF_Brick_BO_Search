import time, os, pickle
from os.path import exists
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

#from rdkit import Chem
#from rdkit.Chem.Draw import IPythonConsole
#from rdkit.Chem import Draw
#IPythonConsole.ipython_useSVG=True  #< set this to False if you want PNGs instead of SVGs
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms
from config import dir_smi_log
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms
import functools, operator, itertools
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from generate_symmetry import find_backbone_sites, find_sidegroup_sites, find_neighbour_sites, rdkit2ase
from generate_symmetry import generate_symmetric_BB, generate_annelation_BB



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

     #'5-ring annelation':'[ch:1]2[c:2][c:3][c:4][c:5][c:6]2>>C1CC[ch:1]2[c:2]1[c:3][c:4][c:5][c:6]2',
    #'5-ring annelation': '[ch:1]2[c:3][c:4][c:5][c:6]3[c:7][c:8][c:9][ch:2][c:10]23>>C1=C[c:1]2[c:3][c:4][c:5][c:6]3[c:7][c:8][c:9][c:2]1[c:10]23', #9
}
operations_sidegroup = {
    'N 6-ring substitution': '[ch&r6:0]>>[nr6:0]', #31
     #'reverse_N 6-ring substitution': '[nr6:0]>>[ch&r6:0]', #32

  #   '6-ring annelation':'[cH&r6:1][cH&r6:2]>>[c:1]2cccc[c:2]2',  #3
     #'reverse_6-ring annelation':'[r6:0][c:1]2[ch][ch][ch][ch][c:2]2[r6:3]>>[*:0][c:1][c:2][*:3]',  #4

  #   '5-ring annelation':'[ch:1]2[c:2][c:3][c:4][c:5][c:6]2>>C1CC[ch:1]2[c:2]1[c:3][c:4][c:5][c:6]2',

    #'S 6-ring substitution': '[ch&r6:0]>>[sr6:0]',

    # 5-ring: oxygen
    #'O 5-ring CH2 substitution': '[#6H2r5:1]>>[O:1]', #19
    #'reverse_O 5-ring CH2 substitution': '[n,c:0]1[n,ch:1][o,O:2][n,c:3][n,c:4]1>>[*:0]1=[*:1][CH2:2][*:3]=[*:4]1', #20
    # 5-ring: sulfur
    #'S 5-ring CH2 substitution': '[#6H2r5:1]>>[S:1]', # 23
    #'reverse_S 5-ring CH2 substitution': '[n,c:0]1[n,ch:1][s,S:2][n,c:3][n,c:4]1>>[*:0]1=[*:1][CH2:2][*:3]=[*:4]1', #24

    #'Dithiole formation': '[CH,ch:0]1=[C,c:1][C,c:2]=[CH,ch:3][*:4]1>>[S:0]1[*:1]=,:[*:2][S:3][*:4]1', #25
    #'reverse_Dithiole formation': '[O,o:0]1[r5:1]-,=,:[r5:2][O,o:3][*:4]1>>[CH:0]1=[C:1][C:2]=[CH:3][*:4]1', #26

    # 5-ring: nitrogen
   # 'N 5-ring CH2 substitution': '[#6H2r5:1]>>[NH:1]', #27
    #'reverse_N 5-ring CH2 substitution': '[n,c:0]1[n,ch:1][NH:2][n,c:3][n,c:4]1>>[*:0]1=[*:1][CH2:2][*:3]=[*:4]1', #28
   # 'N 5-ring CH substitution': '[chr5,CH1r5:1]>>[n:1]', #29
    #'reverse_N 5-ring CH substitution': '[nr5:0]>>[ch&r5:0]', #30

    #'Boronic acid addition': '[cH&r6,CH&r5:1]>>[*:1](-B(O[H])(O[H]))',
    #'aldehyde group addition': '[cH&r6,CH&r5:1]>>[*:1](-[CH]=O)',
    #'amino group addition':'[cH&r6,CH&r5:1]>>[*:1](-[NH2])',
    #'carbonyl group addition': '[cH&r6,CH&r5:1]>>[*:1](=O)',
    #'carbonyl group addition': '[ch:1]1[c:2][c:3][c:4][c:5][c:6]1>>O=[ch:1]1[c:2][c:3][c:4][c:5][c:6]1',
    #'hydroxyl group addition': '[cH&r6,CH&r5:1]>>[*:1](-O[H])',
    #'methoxy group addition': '[cH&r6,CH&r5:1]>>[*:1](-OC)',

    '1-N(CH2CH3)2': '[cH&r6,CH&r5:1]>>[*:1](-N(-CC)CC)', #1
    '2-NH-C(=S)-NH2': '[cH&r6,CH&r5:1]>>[*:1](-N-C(=S)[NH2])', #2
    '3-C(=S)-NH2': '[cH&r6,CH&r5:1]>>[*:1](-C(=S)N)', #3
    '4-SH': '[cH&r6,CH&r5:1]>>[*:1](-[SH])', #4
    #'5-CCl3': '[cH&r6,CH&r5:1]>>[*:1](-C(-Cl)(-Cl)-Cl)',
    #'6-I': '[cH&r6,CH&r5:1]>>[*:1](-I)',
    #'7-CH2-NO2': '[cH&r6,CH&r5:1]>>[*:1](-C[N+](=O)-[O-])',
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
    #'28-NO2': '[cH&r6,CH&r5:1]>>[*:1](-[N+](=O)-[O-])',
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




def calculate_topo_distances(mol, atoms_in_outer, atoms_in_core):
    distance_matix  =  Chem.GetDistanceMatrix(mol)
    topo_distances=[]
    for atom_in_outer in atoms_in_outer:
        topo_distances.append(min([distance_matix[atom_in_outer, atom] for atom in atoms_in_core]))
    return topo_distances

def calculate_topo_distance(mol, atom1, atom2):
    distance_matix  =  Chem.GetDistanceMatrix(mol)
    return distance_matix[atom1, atom2]

def partition_mol(mol):
    pat = Chem.MolFromSmarts('[*]-[#6;R]')
    atoms_cleaved_at = [sorted(x) for x in mol.GetSubstructMatches(pat)]
    if len(atoms_cleaved_at)==0:
        return {'cores': [mol],
            'cores_atoms_ids': [list(range(mol.GetNumAtoms()))],
            'linkers': [],
            'linkers_atoms_ids': [],
            'sidegroups': [],
            'sidegroups_atoms_ids': [],
            'atomids_pairs_cleaved': [],
            'bondids_pairs_cleaved': []}

    bonds_cleave = [mol.GetBondBetweenAtoms(at[0], at[1]) for at in atoms_cleaved_at]
    bonds_cleave = [b for b in bonds_cleave if not b.IsInRing()]
    atoms_cleaved_at2 = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds_cleave]
    bonds_cleave = [b.GetIdx() for b in bonds_cleave]


    if len(atoms_cleaved_at2)==0:
        return {'cores': [mol],
            'cores_atoms_ids': [list(range(mol.GetNumAtoms()))],
            'linkers': [],
            'linkers_atoms_ids': [],
            'sidegroups': [],
            'sidegroups_atoms_ids': [],
            'atomids_pairs_cleaved': [],
            'bondids_pairs_cleaved': []}


    frag_mol = Chem.FragmentOnBonds(mol, bonds_cleave)
    frags_ids_new = Chem.GetMolFrags(frag_mol)
    new_mols = Chem.GetMolFrags(frag_mol, asMols=True)

    resdict={}
    resdict['cores'] = []
    resdict['cores_atoms_ids'] = []
    resdict['linkers'] = []
    resdict['linkers_atoms_ids'] = []
    resdict['sidegroups'] = []
    resdict['sidegroups_atoms_ids'] = []
    resdict['atomids_pairs_cleaved'] = atoms_cleaved_at2
    resdict['bondids_pairs_cleaved'] = bonds_cleave

    count_connectors_list = [Chem.MolToSmiles(m).count('*') for m in new_mols]
    ring_infos = [m.GetRingInfo().AtomRings() for m in new_mols]

    for i,m in enumerate(new_mols):
        if count_connectors_list[i]==1: # can be sidegroup or core
            if len(ring_infos[i]) > 0: #core
                resdict['cores'].append(new_mols[i])
                resdict['cores_atoms_ids'].append(frags_ids_new[i])
            else:
                resdict['sidegroups'].append(new_mols[i])
                resdict['sidegroups_atoms_ids'].append(frags_ids_new[i])

        if count_connectors_list[i]>1: # can be linker or core
            if len(ring_infos[i]) > 0: #core
                resdict['cores'].append(new_mols[i])
                resdict['cores_atoms_ids'].append(frags_ids_new[i])
            else: #linker
                resdict['linkers'].append(new_mols[i])
                resdict['linkers_atoms_ids'].append([x for x in frags_ids_new[i] if x<mol.GetNumAtoms()])

    return resdict

def get_num_heteroatoms(mol_rdkit):
    ''' Count Heteroatoms (other than C,H) in structure '''
    listidx = range(mol_rdkit.GetNumAtoms())
    num_het=0
    for idx in listidx: #list(set(np.array(atom_id_lists_rings).flatten())):
        at=mol_rdkit.GetAtomWithIdx(idx).GetSymbol()
        if at != 'H' and at != 'C':
            num_het+=1
    return num_het


def get_non_ring_atoms(mol_rdkit):
    ''' Find all carbon atoms in linkers '''
    linker_atom_idx=[]
    for at in mol_rdkit.GetAtoms():
        if not at.IsInRing(): # or not at.GetIsAromatic():
            linker_atom_idx.append(at.GetIdx())
    return linker_atom_idx

def check_rules_fulfilled(mol_rdkit, res_dict='', verbose=False):

    max_num_atoms_allowed=0


    num_rings_allowed = 4
    num_linkers_allowed = 4
    num_core_frags_allowed = 5 # Maxmimum number of separate ring systems (separated by linkers or biphenylic bonds)
    max_num_rings_in_corefrag_allowed = 2 # maximum number of rings in a single annelated (core) ring system
    num_het_atoms_allowed = 12
    num_sidegroups_allowed = 2
    num_atoms_allowed = -1
    max_dist_allowed = 11 #the topological distance between two extreme points

    extreme_points=find_backbone_sites(mol_rdkit)
    start_end_dist=calculate_topo_distance(mol_rdkit, extreme_points[0], extreme_points[1])
    if max_dist_allowed > -1 and not start_end_dist <= max_dist_allowed:
        return False

    num_atoms = Chem.AddHs(mol_rdkit).GetNumAtoms()
    if verbose: print('Number of atoms allowed/found: {}/{}'.format(num_atoms_allowed, num_atoms))
    if num_atoms_allowed > -1 and not num_atoms <= num_atoms_allowed:
        if verbose: print('too many atoms', num_atoms)
        return False

    num_rings = CalcNumRings(mol_rdkit)
    if verbose: print('Number of rings allowed/found: {}/{}'.format(num_rings_allowed, num_rings))
    if num_rings_allowed > -1 and not num_rings <= num_rings_allowed:
        if verbose: print('too many rings', num_rings)
        return False

    num_sidegroups = len(res_dict['sidegroups'])
    if verbose: print('Number of sidegroups (in core) allowed/found: {}/{}'.format(num_sidegroups_allowed, num_sidegroups))
    if num_sidegroups_allowed > -1 and not num_sidegroups <= num_sidegroups_allowed :
        if verbose: print('too many sidegroups', num_sidegroups)
        return False
    sidegroups_sites=find_sidegroup_sites(mol_rdkit)
    if len(sidegroups_sites)==2:
        sidegroup_dist=calculate_topo_distance(mol_rdkit, sidegroups_sites[0], sidegroups_sites[1])
        if sidegroup_dist<2:
            return False

    num_het_atoms = get_num_heteroatoms(mol_rdkit)
    if verbose: print('Number of heteroatoms allowed/found: {}/{}'.format(num_het_atoms_allowed, num_het_atoms))
    if num_het_atoms_allowed > -1 and not num_het_atoms <= num_het_atoms_allowed:
        if verbose: print('too many heteroatoms', num_het_atoms)
        return False

    num_linkers = len(res_dict['linkers'])
    if verbose: print('Number of linkers allowed/found: {}/{}'.format(num_linkers_allowed, num_linkers))
    if num_linkers_allowed > -1 and not num_linkers <= num_linkers_allowed:
        if verbose: print('too many linkers', num_linkers)
        return False

    num_core_frags=len(res_dict['cores'])
    if verbose: print('Number of core fragments allowed/found: {}/{}'.format(num_core_frags_allowed, num_core_frags))
    if num_core_frags_allowed > -1 and not num_core_frags <= num_core_frags_allowed:
        if verbose: print('Too many core fragments in the molecule')
        return False

    max_num_rings_in_corefrag = max([CalcNumRings(x) for x in res_dict['cores']])
    if verbose: print('Maximum number of rings in single core fragment allowed/found: {}/{}'.format(max_num_rings_in_corefrag_allowed, max_num_rings_in_corefrag))
    if max_num_rings_in_corefrag_allowed > -1 and not max_num_rings_in_corefrag <= max_num_rings_in_corefrag_allowed:
        if verbose: print('A core fragment contains too many rings')
        return False

    return True


    
def run_mutation(mol_last_generation_smiles, return_removed=False):
    operations_list=list(operations_backbone.values())+list(operations_sidegroup.values())

    new_frames=[]
    #time_start=time.time()
    mol_last_gen=Chem.MolFromSmiles(mol_last_generation_smiles)
    if mol_last_gen == None: return []
    Chem.SanitizeMol(mol_last_gen)
    #print('smi', mol_last_generation_smiles)
    count_tested=0
    for o,op in enumerate(operations_list):
        mol_last_gen=Chem.MolFromSmiles(mol_last_generation_smiles)
        #print('Now is running reaction: {} || Progress: {}/{}'.format(op,o+1,len(operations_list)))

        try:
            backbone_sites=find_backbone_sites(mol_last_gen)
            #print(backbone_sites)
            if o<2:
                mol_last_gen=Chem.MolFromSmiles(mol_last_generation_smiles)
                for at in mol_last_gen.GetAtoms():
                    if at.GetIdx() not in backbone_sites:
                        block_idx=at.GetIdx()
                        mol_last_gen.GetAtomWithIdx(block_idx).SetProp('_protected','1')
                rxn = AllChem.ReactionFromSmarts(op)
                ps = rxn.RunReactants((mol_last_gen,))
                uniq = list(set([Chem.MolToSmiles(x[0]).split('.')[0] for x in ps]))
                #uniq_all_2.extend(uniq)
            else:
                mol_last_gen=Chem.MolFromSmiles(mol_last_generation_smiles)
                for at in mol_last_gen.GetAtoms():
                    if at.GetIdx() in backbone_sites:
                        block_idx=at.GetIdx()
                        mol_last_gen.GetAtomWithIdx(block_idx).SetProp('_protected','1')
                rxn = AllChem.ReactionFromSmarts(op)
                ps = rxn.RunReactants((mol_last_gen,))
                uniq = list(set([Chem.MolToSmiles(x[0]).split('.')[0] for x in ps]))
        except:
            continue
        if len(uniq)==0:
            continue
        for u in uniq:
            #print(u)
            #mol=Chem.MolFromSmiles(u,sanitize=False)
            mol=Chem.MolFromSmiles(u)
            try:
                #Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                res_dict = partition_mol(mol)
            except:
                continue
            if mol==None: continue
            if not check_rules_fulfilled(mol, res_dict=res_dict,verbose=False ): continue

            #os.chdir(dir_smi_log)
            #if not os.path.isfile('{}.log'.format(u)):
            #    with open('{}.log'.format(u),'w') as f:
            #        f.write('{}'.format(op)+'____'+'{}'.format(mol_last_generation_smiles))
            #os.chdir('..')
            with open( dir_smi_log ,'a') as f:
                f.write( '{}____{}____{}\n'.format( mol_last_generation_smiles, op, u  ) )
            new_frames.append(pd.DataFrame(data={
                                                'smi': [u],
                                                'operation': [op],
                                                #'operation_idx': [o],
                                                'mol_last_gen': [mol_last_generation_smiles],
                                                #'generation': [new_generation],
                                                #'added_in_round': [new_generation]
                                           }))
    #print('-------------------------------------------------------------------')

    return new_frames





