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
#print(rdkit.__version__)
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
from math import cos, sin


def calculate_topo_distance(mol, atom1, atom2):
    distance_matix  =  Chem.GetDistanceMatrix(mol)
    return distance_matix[atom1, atom2]


def find_already_backbone_sites(mol):
    pat1=Chem.MolFromSmarts('[#6;R]-[#6;R]')
    atoms1_cleaved_in_ring = [x[0] for x in mol.GetSubstructMatches(pat1)] + [x[-1] for x in mol.GetSubstructMatches(pat1)]
    pat2=Chem.MolFromSmarts('[#6;R]-C#C-[#6;R]')
    atoms2_cleaved_in_ring = [x[0] for x in mol.GetSubstructMatches(pat2)] + [x[-1] for x in mol.GetSubstructMatches(pat2)]
    atoms_backboned= atoms1_cleaved_in_ring + atoms2_cleaved_in_ring
    return atoms_backboned
def find_backbone_sites(mol):
    pat=Chem.MolFromSmarts('[*;!#1]-[#6;R]')
    atoms_cleaved = [x[0] for x in mol.GetSubstructMatches(pat)] + [x[1] for x in mol.GetSubstructMatches(pat)]
    atoms_backboned=find_already_backbone_sites(mol)
    atoms_clean=[]

    ###########################napthaphene######################################
    na_smart=Chem.MolFromSmarts('*1***2*****2*1')
    napthalenes=[sorted(x) for x in mol.GetSubstructMatches(na_smart)]
    if len(napthalenes)>0:
        #max_num_rings_in_corefrag_allowed = 2 so only one benzene is taken into consideration
        #G0
        if len(napthalenes)==1 and len(atoms_cleaved)==0:
            return [2,9]

        ring_infos=mol.GetRingInfo().AtomRings()
        bonds_na=[]
        ring_na=[]
        #the ring only connected with benzene and counted as side group only
        for i in itertools.product(ring_infos,ring_infos):
            if i[0] is not i[1]:
                bond_na=list(set(i[0]).intersection(i[1]))
                if len(bond_na)>0 and all(bond_na[0] not in bond for bond in bonds_na):
                    bonds_na.append(bond_na)
                    ring_na.append(i[0])
                    ring_na.append(i[1])
        if len(napthalenes)==1 and len(ring_infos)==2:
            listidx = range(mol.GetNumAtoms())
            atoms_sites=[]
            for idx in listidx:
                for atom in bonds_na[0]:
                    if calculate_topo_distance(mol, idx, atom)==1 and idx not in bonds_na[0]:
                        atoms_sites.append(idx)
            extreme_paris=[]
            for ring in ring_na:
                extreme_points=[]
                for atom in atoms_sites:
                    if atom in ring and atom not in atoms_cleaved:
                        extreme_points.append(atom)
                if len(extreme_points)==2:
                    extreme_paris.append(extreme_points)
                #print(extreme_paris)
            return extreme_paris[0]
    ###########################################################################
    ring_infos=mol.GetRingInfo().AtomRings()
    if len(atoms_backboned)>0:
        for ring in ring_infos:
            if len( list(set(ring).intersection(atoms_backboned)) )==1:
                for atom in ring:
                    atom_symbol=mol.GetAtomWithIdx(atom).GetSymbol()
                    if atom not in atoms_cleaved and atom_symbol in ['c','C']:
                        atoms_clean.append(atom)
    else:
        for ring in ring_infos:
            for atom in ring:
                atom_symbol=mol.GetAtomWithIdx(atom).GetSymbol()
                if atom not in atoms_cleaved and atom_symbol in ['c','C']:
                    atoms_clean.append(atom)
    max_distance=0
    extreme_point1=0
    extreme_point2=0
    for atom1 in atoms_clean:
        for atom2 in atoms_clean:
            if calculate_topo_distance(mol, atom1, atom2) >= max_distance:
                max_distance=calculate_topo_distance(mol, atom1, atom2) #max_distance must>=3 and can be calculated topologically
                extreme_point1=atom1
                extreme_point2=atom2
    return [extreme_point1,extreme_point2]

def find_sidegroup_sites(mol):
    pat = Chem.MolFromSmarts('[*]-[#6;R]')
    atoms_cleaved_at = [sorted(x) for x in mol.GetSubstructMatches(pat)]
    if len(atoms_cleaved_at)==0:
        return []
    bonds_cleave = [mol.GetBondBetweenAtoms(at[0], at[1]) for at in atoms_cleaved_at]
    bonds_cleave = [b for b in bonds_cleave if not b.IsInRing()]
    atoms_cleaved_at2 = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in bonds_cleave]
    bonds_cleave = [b.GetIdx() for b in bonds_cleave]

    #print(atoms_cleaved_at2)
    if len(atoms_cleaved_at2)==0:
        return []

    frag_mol = Chem.FragmentOnBonds(mol, bonds_cleave)
    frags_ids_new = Chem.GetMolFrags(frag_mol)
    new_mols = Chem.GetMolFrags(frag_mol, asMols=True)
    count_connectors_list = [Chem.MolToSmiles(m).count('*') for m in new_mols]
    ring_infos = [m.GetRingInfo().AtomRings() for m in new_mols]

    sidegroup_sites=[]
    for i,m in enumerate(new_mols):
        if count_connectors_list[i]==1: # can be sidegroup or core
            if len(ring_infos[i]) == 0: #sidegroup >0 is core
                for ats in atoms_cleaved_at2:
                    if len(set(ats) & set(frags_ids_new[i])) > 0:
                        at_side=list(set(ats) & set(frags_ids_new[i]))[0]
                        at_core=[at for at in ats if at != at_side][0]
                        #print(at_side,at_core)
                        sidegroup_sites.append(at_core)
    return sidegroup_sites


def find_neighbour_sites(mol, start_point):
    listidx = range(mol.GetNumAtoms())
    sidegroup_points=find_sidegroup_sites(mol)
    two_neighbours=[]
    neighbour_sites=[]
    for at in listidx:
        topo_dist=calculate_topo_distance(mol, start_point, at)
        #print(mol.GetAtomWithIdx(at).GetSymbol())
        if topo_dist==1:
            two_neighbours.append(at)
        if topo_dist==1 and mol.GetAtomWithIdx(at).GetSymbol()=='C' and at not in sidegroup_points:
            neighbour_sites.append(at)
    return two_neighbours, neighbour_sites

def rdkit2ase(rdkit_mol):
    """ Extract 2D coordinates from RDKit mol and convert to ASE mol """
    AllChem.Compute2DCoords(rdkit_mol)
    positions=rdkit_mol.GetConformer().GetPositions()
    #print(positions)
    atoms_symbols=np.array([at.GetSymbol() for at in rdkit_mol.GetAtoms()])
    return Atoms(atoms_symbols, positions=positions)

def locate_point(pt, vec, dist):
    unit_vec = vec / np.linalg.norm(vec)
    return pt + dist * unit_vec

def rotate_vec(vec, deg):
    theta = np.deg2rad(deg)
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    tmp=list(np.dot(rot,vec[0:2]))
    tmp.append(0)
    return np.array(tmp)

def remove_H(atoms_wH, N_woH, end_point):
    #11, 18: 12, 13, 14, 15, 16, 17, 18
    atoms_woH=deepcopy(atoms_wH)
    N_wH=len(atoms_wH.positions)
    min_dist=100
    removed_H=0
    for atom in range(N_woH+1, N_wH):
        #print(atom)
        H_v=atoms_wH.positions[atom]
        end_v=atoms_wH.positions[end_point]
        dist=np.linalg.norm(H_v-end_v)
        if dist<=min_dist:
            min_dist=dist
            removed_H=atom
    atoms_woH.pop(removed_H)
    return atoms_woH

def find_outermost_H(atoms_wH, N_woH, end_point):
    #find but dont do the remove
    atoms_woH=deepcopy(atoms_wH)
    N_wH=len(atoms_wH.positions)
    min_dist=100
    removed_H=0
    for atom in range(N_woH+1, N_wH):
        #print(atom)
        H_v=atoms_wH.positions[atom]
        end_v=atoms_wH.positions[end_point]
        dist=np.linalg.norm(H_v-end_v)
        if dist<=min_dist:
            min_dist=dist
            removed_H=atom
    return removed_H

def add_C(atoms, start_point, end_point, bond_length=1.5, triple_bond=0):
    atoms_copy=deepcopy(atoms)
    start_v=atoms_copy.get_positions()[start_point]
    end_v=atoms_copy.get_positions()[end_point]
    atoms_copy.append('C')
    atoms_copy.positions[-1] = locate_point(start_v, start_v-end_v, bond_length)
    if triple_bond==1:
        atoms_copy.append('C')
        atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-2], start_v-end_v, bond_length)

        atoms_copy.append('C')
        atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-2], start_v-end_v, bond_length)

    #record this index
    c_core_idxs=[len(atoms_copy.positions)-1]
    ##########################
    atoms_copy.append('C')
    atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-2], rotate_vec(start_v-end_v, -60), bond_length)
    #Add H
    atoms_copy.append('H')
    atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-2], rotate_vec(start_v-end_v, -120), bond_length)
    ##########################

    atoms_copy.append('C')
    atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-3], start_v-end_v, bond_length)
    c_core_idxs.append(len(atoms_copy.positions)-1)
    
    #######################
    atoms_copy.append('C')
    atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-2], rotate_vec(start_v-end_v, 60), bond_length)
    #Add H
    atoms_copy.append('H')
    atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-2], start_v-end_v, bond_length)
    #######################

    atoms_copy.append('C')
    atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-3], rotate_vec(start_v-end_v, 120), bond_length)
    c_core_idxs.append(len(atoms_copy.positions)-1)
    
    #################
    atoms_copy.append('C')
    atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-2], rotate_vec(start_v-end_v, 180), bond_length)
    #Add H
    atoms_copy.append('H')
    atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-2], rotate_vec(start_v-end_v, 120), bond_length)
    #########################
    return c_core_idxs, atoms_copy


def add_C_annelation(atoms, start_point, end_point, neighbour_point, two_neighbours, bond_length=1.5):
    atoms_copy=deepcopy(atoms)
    start_v=atoms_copy.get_positions()[start_point]  #c0
    end_v=atoms_copy.get_positions()[end_point]
    neighbour_v=atoms_copy.get_positions()[neighbour_point] #c1
    foretype_v=neighbour_v-start_v #c0->c1
    c_core_idxs=[]
    c_core_idxs.append(start_point)

    c_neighbour_idxs=[neighbour_point]
    #first added point need to checked the direction
    angle_rotate=0
    for angle_bias in [+120, -120]:
        c2=locate_point(start_v, rotate_vec(foretype_v, angle_bias), bond_length)
        if all([np.linalg.norm(c2-atoms_copy.get_positions()[at])>1e-2 for at in two_neighbours]):
            angle_rotate=angle_bias

    c_neighbour_idxs.append(len(atoms_copy.positions))
    #c2
    atoms_copy.append('C')
    atoms_copy.positions[-1] = locate_point(start_v, rotate_vec(foretype_v, angle_rotate), bond_length)

    c_core_idxs.append(len(atoms_copy.positions))
    #c3
    atoms_copy.append('C')
    atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-2], rotate_vec(foretype_v, 0.5*angle_rotate), bond_length)

    c_neighbour_idxs.append(len(atoms_copy.positions))
    #c4
    atoms_copy.append('C')
    atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-2], foretype_v, bond_length)

    c_core_idxs.append(len(atoms_copy.positions))
    #c5
    atoms_copy.append('C')
    atoms_copy.positions[-1] = locate_point(atoms_copy.positions[-2], rotate_vec(foretype_v, -0.5*angle_rotate), bond_length)
    return c_neighbour_idxs, c_core_idxs, atoms_copy

def rotate_symmetry(atoms, N_atoms, start_point, end_point, bond_length, outermost_H, triple_bond=0):
    h_outermost_idxs=[outermost_H]
    #N_atoms is number of atoms w/o any addition
    atoms_copy=deepcopy(atoms)
    symbols=list(atoms.symbols)
    if triple_bond==1:
        origin_pt=atoms.positions[N_atoms+5+1] #one more H
        foretype_v=atoms.positions[N_atoms+3]
        N_atoms+=2

    origin_pt=atoms.positions[N_atoms+3+1] #one more H
    foretype_v=atoms.positions[N_atoms+1]
    for target_atom in range(N_atoms):
        target_symbol=symbols[target_atom]
        atoms_copy.append(target_symbol)
        target_v=atoms.positions[target_atom]
        dist=np.linalg.norm(target_v-foretype_v)
        atoms_copy.positions[-1] = locate_point(origin_pt, rotate_vec(target_v-foretype_v, 120), dist)
        if target_atom==outermost_H:
            h_outermost_idxs.append(len(atoms_copy.positions)-1)

    origin_pt=atoms.positions[N_atoms+5+2] #two more H added already
    for target_atom in range(N_atoms):
        target_symbol=symbols[target_atom]
        atoms_copy.append(target_symbol)
        #print(len(atoms_copy.positions))
        target_v=atoms.positions[target_atom]
        dist=np.linalg.norm(target_v-foretype_v)
        atoms_copy.positions[-1] = locate_point(origin_pt, rotate_vec(target_v-foretype_v, -120), dist)
        if target_atom==outermost_H:
            h_outermost_idxs.append(len(atoms_copy.positions)-1)
    return h_outermost_idxs, atoms_copy


def rotate_symmetry_annelation(atoms, N_atoms, start_point, end_point, neighbour_point, bond_length,                                outermost_H, c_core_idxs, c_neighbour_idxs):
    h_outermost_idxs=[outermost_H]
    #N_atoms is number of atoms w/o any addition
    atoms_copy=deepcopy(atoms)
    symbols=list(atoms.symbols)

    origin_pt=atoms.positions[c_core_idxs[1]]
    foretype_v=atoms.positions[c_core_idxs[0]]

    #before rotate have to decide the rotation angle
    angle_rotate=0
    v1=atoms.positions[c_neighbour_idxs[0]]-atoms.positions[c_core_idxs[0]]
    v2=atoms.positions[c_neighbour_idxs[1]]-atoms.positions[c_core_idxs[1]]
    v3=atoms.positions[c_neighbour_idxs[2]]-atoms.positions[c_core_idxs[2]]
    dist=np.linalg.norm(v1)
    for angle_bias in [+120,-120]:
        if np.linalg.norm( atoms.positions[c_neighbour_idxs[1]]-                          locate_point(atoms.positions[c_core_idxs[1]], rotate_vec(v1, angle_bias), dist) )<1e-2:
            angle_rotate=angle_bias

    for target_atom in range(N_atoms):
        if target_atom not in [start_point, neighbour_point]:
            target_symbol=symbols[target_atom]
            atoms_copy.append(target_symbol)
            target_v=atoms.positions[target_atom]
            dist=np.linalg.norm(target_v-foretype_v)
            atoms_copy.positions[-1] = locate_point(origin_pt, rotate_vec(target_v-foretype_v, angle_rotate), dist)
            if target_atom==outermost_H:
                h_outermost_idxs.append(len(atoms_copy.positions)-1)

    origin_pt=atoms.positions[c_core_idxs[2]]
    for target_atom in range(N_atoms):
        if target_atom not in [start_point, neighbour_point]:
            target_symbol=symbols[target_atom]
            atoms_copy.append(target_symbol)
        #print(len(atoms_copy.positions))
            target_v=atoms.positions[target_atom]
            dist=np.linalg.norm(target_v-foretype_v)
            atoms_copy.positions[-1] = locate_point(origin_pt, rotate_vec(target_v-foretype_v, -angle_rotate), dist)
            if target_atom==outermost_H:
                h_outermost_idxs.append(len(atoms_copy.positions)-1)
    return h_outermost_idxs, atoms_copy

def generate_symmetric_BB(N_woH, atoms, start_point=0, end_point=0, connect_triple=0):
    #connect triple: connect with core benzene with single bond or triple bond
    #atoms_woH=remove_H(atoms, N_woH, end_point)
    atoms_woH=remove_H(atoms, N_woH, start_point)
    outermost_H = find_outermost_H(atoms_woH, N_woH, end_point)

    N_atoms=len(atoms_woH.get_positions()) #the number of atoms need to be rotationally repeated

    c_core_idxs, atoms_wC =add_C(atoms_woH,start_point,end_point,1.5, triple_bond=connect_triple)

    h_outermost_idxs,atoms_final=rotate_symmetry(atoms_wC,N_atoms,start_point,end_point,1.5,outermost_H,triple_bond=connect_triple)

    atoms_final.info['c_core_idxs']=c_core_idxs
    atoms_final.info['h_outermost_idxs']=h_outermost_idxs

    return atoms_final

def generate_annelation_BB(N_woH, atoms, start_point=0, end_point=0, neighbour_point=0, two_neighbours=[]):
    atoms_woH=remove_H(atoms, N_woH, start_point)
    atoms_woH=remove_H(atoms_woH, N_woH, neighbour_point)
    outermost_H = find_outermost_H(atoms_woH, N_woH, end_point)

    N_atoms=len(atoms_woH.get_positions()) #the number of atoms need to be rotationally repeated
    c_neighbour_idxs, c_core_idxs, atoms_wC =add_C_annelation(atoms_woH,start_point,end_point, neighbour_point, two_neighbours,1.5)

    h_outermost_idxs,atoms_final=rotate_symmetry_annelation(atoms_wC, N_atoms, start_point, end_point, neighbour_point, 1.5, outermost_H, c_core_idxs, c_neighbour_idxs)

    atoms_final.info['c_core_idxs']=c_core_idxs
    atoms_final.info['h_outermost_idxs']=h_outermost_idxs

    return atoms_final
