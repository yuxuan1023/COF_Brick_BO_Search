from ase.io import read
from ase.optimize import BFGS
from ase.calculators.dftb import Dftb
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
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
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms
RDLogger.DisableLog('rdApp.*')
import ase
from ase import Atoms
from ase.io import read, write
from copy import deepcopy
import multiprocessing as mp
import pandas as pd
import numpy as np
from bisect import bisect
import random
#import quadpy
from scipy.interpolate import CubicSpline, Rbf
from ase.build import attach, make_supercell
from ase.visualize import view
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdDistGeom,rdMolTransforms
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit import DistanceGeometry
from math import cos, sin, ceil
from openbabel import pybel
from generate_symmetry import generate_symmetric_BB
from config import dir_calculation, dir_dss

def linear_correct_to_b3lyp(propname, val):

    ''' Correct xTB-GFN1 descriptor-values to B3LYP using linear correlation '''

    if propname=="XTB1_lamda_h": val = 1.45 * val + 28
    elif propname=="ehomo_gfn1_b3lyp": val = 0.92 * val + 4.43

    return val

def run_opf_bfgs(ase_atoms, command):
    ''' Fallback for bfgs optimization, if ANCOPT didn't work '''

    ase_atoms.write('scratch.coord', format='turbomole')
    with open('scratch.coord', 'r') as out:
        coords=out.read()
    coords=coords.replace('$end\n', '')
    with open('opt.inp', 'w') as out:
        out.write(coords+'''$opt
    engine=lbfgs
$end''')

    os.system( command.replace('scratch.xyz', 'opt.inp') )
    print(command.replace('scratch.xyz', 'opt.inp'), os.getcwd())
    ase_atoms=read('xtbopt.inp')
    ase_atoms.write('geometry.in')
    ase_atoms.read('geometry.in')
    os.system('xtb --hess --gfnff xtbopt.inp > freq.out')
    return ase_atoms

def read_homo_lumo_energy_xtb(logfile_name):
    ''' read HOMO-, LUMO-, and totalenergy from logfile, returns values in eV '''
    homo_en=0.; lumo_en=0.; tot_en=0.

    with open(logfile_name) as out:
        for line in out.readlines():
            if '(HOMO)' in line:
                homo_en=float(line.split()[-2])
            if '(LUMO)' in line:
                lumo_en=float(line.split()[-2])
            if 'TOTAL ENERGY' in line:
                tot_en=float(line.split('TOTAL ENERGY')[1].split('Eh')[0])

    return homo_en, lumo_en, tot_en*27.2113845


def check_smi_valence(m):
    #m = Chem.MolFromSmiles(smi)
    for atom in m.GetAtoms():
        if atom.GetSymbol()=='C':
            if atom.GetExplicitValence()+atom.GetTotalNumHs() !=4:
                return False
        if atom.GetSymbol()=='N':
            if atom.GetExplicitValence()+atom.GetTotalNumHs() !=3:
                return False
        if atom.GetSymbol()=='O':
            if atom.GetExplicitValence()+atom.GetTotalNumHs() !=2:
                return False
        if atom.GetSymbol()=='S':
            if atom.GetExplicitValence()+atom.GetTotalNumHs() !=2:
                return False
    return True



def rdkit2ase(rdkit_mol):
    """ Extract 2D coordinates from RDKit mol and convert to ASE mol """
    AllChem.Compute2DCoords(rdkit_mol)
    positions=rdkit_mol.GetConformer().GetPositions()
    #print(positions)
    atoms_symbols=np.array([at.GetSymbol() for at in rdkit_mol.GetAtoms()])
    return Atoms(atoms_symbols, positions=positions)

def atoms2smiles(atoms):
    atoms.write('tmp.xyz')
    mol = next(pybel.readfile("xyz", 'tmp.xyz'))
    smi = mol.write(format="smi")
    return smi.split()[0].strip()

def calculate_topo_distance(mol, atom1, atom2):
    distance_matix  =  Chem.GetDistanceMatrix(mol)
    return distance_matix[atom1, atom2]

def idx2origin_idx(idx):
    return ceil((idx+1)/8-1)

def run_single(ase_atoms, method='GFN1',idx=0, optlevel='vtight', chg=0, mpiexe='', n_procs=1, log=True):
    ''' Optimize a single molecular conformer '''
    cmd = ''
    if mpiexe!='': cmd += mpiexe+' --np 1 '

    cmd += 'xtb scratch.xyz'

    if n_procs>=1: cmd+=' -P {}'.format(n_procs)

    if 'ff' in method: cmd +=' --gfnff '
    else: cmd +=' --gfn {} '.format(method.split('GFN')[1])

    ase_atoms=deepcopy(ase_atoms)

    if optlevel!=None: cmd+='--opt {} --cycles 500'.format(optlevel)
    if chg!=0: cmd+='--uhf 1 '

    ase_atoms.write('scratch.xyz')
    cmd+='--chrg {} --acc 1 --iterations 250'.format(chg) #(30,1,0.2)--acc 0.2 for production, default for testspace

    if log: cmd+=' > E0G0.log'
    print(cmd, os.environ['OMP_NUM_THREADS'])
    os.system(cmd)

    finish_flag=0
    with open('E0G0.log') as out:
        for line in out.readlines():
            if '* finished run on' in line:
                finish_flag=1
    if finish_flag==0:
        #os.chdir('..')
        return None


    if optlevel!=None:
        # fallback option if ANCOPT failed
        if not os.path.isfile('xtbopt.xyz'):
            ase_atoms = run_opf_bfgs(ase_atoms, cmd)
        else:
            ase_atoms=read('xtbopt.xyz')
            ase_atoms.write('geometry.in')
            ase_atoms=read('geometry.in')
    homo_en, lumo_en, tot_en = read_homo_lumo_energy_xtb('E0G0.log')

    ase_atoms.info['total_energy_eV'] = tot_en
    ase_atoms.info['homo_uncorrected'] = homo_en #eV
    ase_atoms.info['lumo'] = lumo_en
    
    return ase_atoms

'''
def atoms_optimization(atoms,idx):
    print('here1')
    if atoms==None:
        print('here2')
        return None
    os.chdir(dir_calculation)
    os.mkdir('run_{}'.format(idx))
    os.chdir('run_{}'.format(idx))

    cwd = os.getcwd()
    if os.path.isdir('run_opt'):
        shutil.rmtree('run_opt')
    os.mkdir('run_opt')
    os.chdir('run_opt')
    try:
        optlevel = 'vtight'
        method='GFN1'
        atoms_stable=run_single(deepcopy(atoms), method=method, \
                                idx=idx, optlevel=optlevel, chg=0, n_procs=1)
        os.chdir(cwd)
        os.chdir('..')
        #print(os.getcwd())
        ###os.system('rm -r run_band')
        atoms_res = atoms_stable     #optimized most stable conformer
        atoms_tmp=deepcopy(atoms_res)
        atoms_tmp.write('mol_res_tmp_{}.xyz'.format(idx))
        mol_tmp = next(pybel.readfile("xyz", 'mol_res_tmp_{}.xyz'.format(idx)))
        smi_tmp = mol_tmp.write(format="smi")
        smi_tmp=smi_tmp.split()[0].strip()
        mol_tmp=Chem.MolFromSmiles(smi_tmp)
        smi_tmp=Chem.MolToSmiles(mol_tmp)
        os.system('rm mol_res_tmp_{}.xyz'.format(idx))

        atoms_res.info['symmetry_smi'] = smi_tmp
        atoms_res.info['idx']=idx
        atoms_res.info['origin_idx']=idx2origin_idx(idx)
        atoms_res.info['c_core_idxs'] = atoms.info['c_core_idxs']
        atoms_res.info['h_outermost_idxs'] = atoms.info['h_outermost_idxs']
        atoms_res.write('mol_res_opt_{}.xyz'.format(idx))
        atoms.write('mol_res_initial_{}.xyz'.format(idx))
        os.system("rm -rf run_{}".format(idx))
        #print('Time for step {}: {}'.format(idx, time.time()-time_start))
    except:
        os.chdir(cwd)
        os.chdir('..')
        os.system("rm -rf run_{}".format(idx))
        #print('Time for failed {}: {}'.format(idx, time.time()-time_start))
        return None
    return atoms_res
'''



def do_optimization(atoms, charge=0):
    # 1000K = 0.0031668 au
    calc = Dftb(atoms=atoms,
            #label=insert_molecule,
            Driver_='ConjugateGradient',
            Driver_MaxForceComponent=0.0045,
            Driver_MaxSteps=500,
            Hamiltonian_='DFTB',  # line is included by default
            Hamiltonian_SCC='Yes',
            Hamiltonian_MaxSCCIterations = 200 ,
            Hamiltonian_SCCTolerance=1e-3,
            Hamiltonian_charge=charge,
            Hamiltonian_Filling_ = 'Fermi',
            Hamiltonian_Filling_Temperature = 0.00095004,
            Hamiltonian_MaxAngularMomentum_='',
            Hamiltonian_MaxAngularMomentum_C='p',
            Hamiltonian_MaxAngularMomentum_N='p',
            Hamiltonian_MaxAngularMomentum_O='p',
            Hamiltonian_MaxAngularMomentum_S='d',
            Hamiltonian_MaxAngularMomentum_H='s',
            Hamiltonian_Dispersion_='LennardJones',
            Hamiltonian_Dispersion_Parameters='UFFParameters {}',
            Parallel_='',
            Parallel_UseOmpThreads = 'Yes',
        )
    atoms.set_calculator(calc)
    calc.calculate(atoms)
    optmized_atoms = read('geo_end.gen')
    return optmized_atoms

def atoms_optimization(atoms,idx):
    if atoms==None:
        return None
    os.chdir(dir_calculation)
    os.mkdir('run_{}'.format(idx))
    os.chdir('run_{}'.format(idx))

    #multiproc_conformers=0
    #if multiproc_conformers<1: n_cpu=1
    #else: n_cpu = multiproc_conformers
    #n_cpu=1
    cwd = os.getcwd()
    if os.path.isdir('run_opt'):
        shutil.rmtree('run_opt')
    os.mkdir('run_opt')
    os.chdir('run_opt')
    try:
        print('i think i am not here')
        atoms_stable=do_optimization(atoms, charge=0)

        #df_band=pd.read_csv('band.out',delim_whitespace=True, skiprows=1, names=['kpt', 'energy', 'spin'])
        #lumo_idx=bisect(df_band.spin,1)
        #for index, row in df_band.iterrows():
            #if row['spin']==0 and df_band.loc[index-1,'spin']!=0:
            #    lumo_idx=index-1
        #    if row['spin']==0 and df_band.loc[index-1,'spin']!=0:
        #        lumo_idx=index
        #print('lumo idx', lumo_idx)
        #homo_idx=lumo_idx-1
        #lumo=df_band.iloc[lumo_idx]['energy']
        #homo=df_band.iloc[homo_idx]['energy']
        #print(idx, 'homo',homo,homo_idx)

        os.chdir(cwd)
        os.chdir('..')
        #print(os.getcwd())
        ###os.system('rm -r run_band')
        atoms_res = atoms_stable     #optimized most stable conformer
        #os.system("rm -rf run_{}".format(idx))
        #atoms_res.info['smi']=smi
        atoms_tmp=deepcopy(atoms_res)
        #del atoms_tmp[[atom.index for atom in atoms_tmp if atom.symbol=='H']]
        atoms_tmp.write('mol_res_tmp_{}.xyz'.format(idx))

        mol_tmp = next(pybel.readfile("xyz", 'mol_res_tmp_{}.xyz'.format(idx)))
        #print(mol_tmp)
        smi_tmp = mol_tmp.write(format="smi")
        smi_tmp=smi_tmp.split()[0].strip()
        #smi_tmp=sorted(smi_tmp.split('.'),key=len)[0]
        mol_tmp=Chem.MolFromSmiles(smi_tmp)
        smi_tmp=Chem.MolToSmiles(mol_tmp)
        #print('smi',smi_tmp)
        os.system('rm mol_res_tmp_{}.xyz'.format(idx))

        atoms_res.info['symmetry_smi'] = smi_tmp
        atoms_res.info['idx']=idx
        atoms_res.info['origin_idx']=idx2origin_idx(idx)
        #atoms_res.info['idx']=idx
        #atoms_res.info['homo'] = homo
        #atoms_res.info['lumo'] = lumo
        atoms_res.info['c_core_idxs'] = atoms.info['c_core_idxs']
        atoms_res.info['h_outermost_idxs'] = atoms.info['h_outermost_idxs']
        atoms_res.write('mol_res_opt_{}.xyz'.format(idx))
        #atoms.write('mol_res_initial_{}.xyz'.format(idx))
        os.system("rm -rf run_{}".format(idx))
        #result_atoms.append(atoms_res)
        #print(atoms_res)
        #print('Time for step {}: {}'.format(idx, time.time()-time_start))
    except:
        os.chdir(cwd)
        os.chdir('..')
        os.system("rm -rf run_{}".format(idx))
        #print('Time for failed {}: {}'.format(idx, time.time()-time_start))
        return None
    return atoms_res


def calculate_reorganization_energy_xtb_cmd(method, atoms, idx, optlevel='vtight', mode='hole', n_procs=1):
    ''' Calculate a singe reorganization energy '''
    run_cwd=os.getcwd()
    if method=='XTB1' or method=='GFN1': method_internal='GFN1'
    else: method_internal='GFN2'

    if mode=='electron': chg=-1
    else: chg=1

    try:
        os.mkdir('run_E0G0')
        os.chdir('run_E0G0')
        atoms_E0G0 = run_single(deepcopy(atoms), method=method_internal, idx=idx, \
                                optlevel=optlevel, chg=0, n_procs=n_procs)
        os.chdir(run_cwd)
        os.mkdir('run_EpG0')
        os.chdir('run_EpG0')
        atoms_EpG0 = run_single(deepcopy(atoms_E0G0), method=method_internal,idx=idx, \
                                optlevel=None, chg=chg, n_procs=n_procs)
        os.chdir(run_cwd)
        os.mkdir('run_EpGp')
        os.chdir('run_EpGp')
        atoms_EpGp = run_single(deepcopy(atoms_E0G0), method=method_internal,idx=idx, \
                                optlevel=optlevel, chg=chg, n_procs=n_procs)
        os.chdir(run_cwd)
        os.mkdir('run_E0Gp')
        os.chdir('run_E0Gp')
        atoms_E0Gp = run_single(deepcopy(atoms_EpGp), method=method_internal,idx=idx, \
                                optlevel=None, chg=0, n_procs=n_procs)
   
        E0G0 = atoms_E0G0.info['total_energy_eV']
        EpG0 = atoms_EpG0.info['total_energy_eV']
        EpGp = atoms_EpGp.info['total_energy_eV']
        E0Gp = atoms_E0Gp.info['total_energy_eV']

        e_lambda = ((EpG0 + E0Gp) - (EpGp + E0G0))*1000 #meV
        print("Estimated reorganization energy (meV, {}): {} for {}".format(mode, e_lambda, idx))
    except:
        os.chdir(run_cwd)
        return None, None, [None, None, None, None]

    return  e_lambda, idx, [atoms_E0G0, atoms_EpGp, atoms_EpG0, atoms_E0Gp]

def calculate_xtb(atoms, idx):
    #time_start=time.time()
    if atoms==None: return None
    atoms_res=None
    os.mkdir(os.path.join(dir_calculation,'run_{}'.format(idx)))
    os.chdir(os.path.join(dir_calculation,'run_{}'.format(idx)))
    cwd = os.getcwd()
    #optlevel = 'vtight'
    optlevel = 'normal'
    method='GFN1'
    # Get the hole-reorganization energy
    e_lambda_h, _, atoms_list_h = calculate_reorganization_energy_xtb_cmd(method, atoms, idx, \
                                            optlevel=optlevel, mode='hole', n_procs=1)
    # Get the electron-reorganization energy
    e_lambda_e, _, atoms_list_e = calculate_reorganization_energy_xtb_cmd(method, atoms, idx, \
                                            optlevel=optlevel, mode='electron', n_procs=1)
    atoms_res = atoms_list_h[0]

    os.chdir(cwd)
    os.chdir('..')
    try:
        atoms_tmp=deepcopy(atoms_res)
        atoms_tmp.write('mol_res_tmp_{}.xyz'.format(idx))
        mol_tmp = next(pybel.readfile("xyz", 'mol_res_tmp_{}.xyz'.format(idx)))
        smi_tmp = mol_tmp.write(format="smi")
        smi_tmp=smi_tmp.split()[0].strip()
        mol_tmp=Chem.MolFromSmiles(smi_tmp)
        smi_tmp=Chem.MolToSmiles(mol_tmp)
        os.system('rm mol_res_tmp_{}.xyz'.format(idx))

        if '.' in smi_tmp:
            os.system("rm -rf run_{}".format(idx))
            return None


        atoms_res.info['symmetry_smi'] = smi_tmp
        atoms_res.info['idx']=idx
        atoms_res.info['origin_idx']=idx2origin_idx(idx)
        atoms_res.info['c_core_idxs'] = atoms.info['c_core_idxs']
        atoms_res.info['h_outermost_idxs'] = atoms.info['h_outermost_idxs']
        atoms_res.info['e_lambda_h_uncorrected'] = e_lambda_h
        atoms_res.info['e_lambda_e'] = e_lambda_e

        homo_corrected = linear_correct_to_b3lyp('ehomo_gfn1_b3lyp',  atoms_res.info['homo_uncorrected'])
        e_lambda_h_corrected = linear_correct_to_b3lyp('XTB1_lamda_h',  atoms_res.info['e_lambda_h_uncorrected'])
        atoms_res.info['homo']=homo_corrected
        atoms_res.info['e_lambda_h'] = e_lambda_h_corrected
        os.system("rm -rf run_{}".format(idx))
        atoms_res.info['calculation_status']='calculated'
        atoms_res.write( 'mol_res_opt_{}.xyz'.format(idx) )
        atoms.write( 'mol_res_initial_{}.xyz'.format(idx) )
        #print('Time for step {}: {}'.format(idx, time.time()-time_start))
    except:
        os.system("rm -rf run_{}".format(idx))
        #print('Time for smiles failed {}: {}'.format(idx, time.time()-time_start))
        return None
    return atoms_res
