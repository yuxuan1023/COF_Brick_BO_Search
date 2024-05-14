import os, time, socket, shutil, random, subprocess, pickle
import math
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
import multiprocessing as mp
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import ase
from ase import Atoms
from ase.io import read, write

#from ipynb.fs.full.dftb_calculation import calculate_single
from pathos.multiprocessing import ProcessingPool as Pool
from xtb_calculation import calculate_xtb, atoms_optimization, atoms2smiles
from gpr_model import GPR_tanimoto, optimize_GPR
#from morphing_design import run_mutation
import morphing_design as morphing
#from config import dir_calculation
#morphing.new_generation=0
from rdkit import RDLogger

from math import cos, sin, ceil
from openbabel import pybel
from generate_symmetry import find_backbone_sites, find_sidegroup_sites, find_neighbour_sites, rdkit2ase
from generate_symmetry import generate_symmetric_BB, generate_annelation_BB
from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem,Draw
import numpy as np

RDLogger.DisableLog('rdApp.*')

# In[ ]:


#Pseude code flow
#1) Select 10(50-100) molecules and train GPR
#2) 10 molecules morphed two times to 400 candidates and now totally 410 canidates,
#   use the GPR model to predict lambda and homo to calculate F and Facq
#2) Seleect top 50 F or Facq(?) to retrain the model
#3) keep the 50 molecules growing and do next learning step

def RW_sm_selection(df, n_sel=100, field='similarity', random_state=42):
    ''' Roulette Wheel selection, based on ranks '''

    smiles_selected=[]
    np.random.seed(random_state)
    df=df.sort_values( by = field, ascending=False)
    df_sel = df.copy()

    if df.shape[0]<=n_sel: return df.smi.tolist()

    for i in range(n_sel):
        df_sel = df_sel[~df_sel.smi.isin(smiles_selected)]
        ranks =  [(x+1) for x in range(df_sel.shape[0])]
        sum_ranks = np.sum(ranks)
        df_sel['cum_ranks'] = list(np.cumsum(ranks))
        rand_val=np.random.randint(0, sum_ranks)
        smiles_selected.append( df_sel[df_sel.cum_ranks>=rand_val].iloc[0].smi )

    return smiles_selected


def RW_rank_selection(df, n_sel=100, field='utility', random_state=42):
    ''' Roulette Wheel selection, based on ranks '''

    smiles_selected=[]
    np.random.seed(random_state)
    df=df.sort_values( by = field, ascending=True)
    df_sel = df.copy()

    if df.shape[0]<=n_sel: return df.smi.tolist()

    for i in range(n_sel):
        df_sel = df_sel[~df_sel.smi.isin(smiles_selected)]
        ranks =  [(x+1) for x in range(df_sel.shape[0])]
        sum_ranks = np.sum(ranks)
        df_sel['cum_ranks'] = list(np.cumsum(ranks))
        rand_val=np.random.randint(0, sum_ranks)
        smiles_selected.append( df_sel[df_sel.cum_ranks>=rand_val].iloc[0].smi )

    return smiles_selected

def RW_sm_selection_df(df, n_sel=100, field='utility', random_state=42):
    ''' Roulette Wheel selection, based on ranks '''

    smiles_selected=[]
    df_added = pd. DataFrame()
    np.random.seed(random_state)
    df=df.sort_values( by = field, ascending=False)
    df_sel = df.copy()

    #if df.shape[0]<=n_sel: return df.molecule_smiles.tolist()
    if df.shape[0]<=n_sel: return df

    for i in range(n_sel):
        df_sel = df_sel[~df_sel.smi.isin(smiles_selected)]
        ranks =  [(x+1) for x in range(df_sel.shape[0])]
        sum_ranks = np.sum(ranks)
        df_sel['cum_ranks'] = list(np.cumsum(ranks))
        rand_val=np.random.randint(0, sum_ranks)
        smiles_selected.append( df_sel[df_sel.cum_ranks>=rand_val].iloc[0].smi )
        df_added = df_added.append(df_sel[df_sel.cum_ranks>=rand_val].iloc[0])
    df_added.drop(['cum_ranks'],axis=1,inplace=True)
    df_added.reset_index(drop=True,inplace=True)
    return df_added


def RW_rank_selection_df(df, n_sel=100, field='utility', random_state=42):
    ''' Roulette Wheel selection, based on ranks '''

    smiles_selected=[]
    df_added = pd. DataFrame()
    np.random.seed(random_state)
    df=df.sort_values( by = field, ascending=True)
    df_sel = df.copy()

    #if df.shape[0]<=n_sel: return df.molecule_smiles.tolist()
    if df.shape[0]<=n_sel: return df

    for i in range(n_sel):
        df_sel = df_sel[~df_sel.smi.isin(smiles_selected)]
        ranks =  [(x+1) for x in range(df_sel.shape[0])]
        sum_ranks = np.sum(ranks)
        df_sel['cum_ranks'] = list(np.cumsum(ranks))
        rand_val=np.random.randint(0, sum_ranks)
        smiles_selected.append( df_sel[df_sel.cum_ranks>=rand_val].iloc[0].smi )
        df_added = df_added.append(df_sel[df_sel.cum_ranks>=rand_val].iloc[0])
    df_added.drop(['cum_ranks'],axis=1,inplace=True)
    df_added.reset_index(drop=True,inplace=True)
    return df_added

def get_F(y, ideal_points=[0.0,-5.1], weights=[1.0,0.7],
                           std=[], kappa=1., return_array=False):

    ''' Compute utility function F and acquisition function Facq = F + k*\sigma '''
    
    y[0]/=1000. # lambda to eV
    
    utility_hole = -np.sqrt(( (y[0]-ideal_points[0]) * weights[0])**2 + ( (y[1]-ideal_points[1]) * weights[1])**2)

    if len(std)>0:

        # for compatiblity with order in AL
        std[0]/=1000.
        var_hole = np.sum( utility_hole**(-2) * np.array([weights[0], weights[1]])**4 *                           (np.array([ y[0], y[1] ]) - np.array( [ideal_points[0],ideal_points[1]] ))**2 *                            np.array( [std[0],std[1]] )**2 )

        std_hole = np.sqrt(var_hole)
        utility_hole += kappa*std_hole

        if return_array:
            return [ utility_hole, std_hole ]

    if return_array:
        return [ utility_hole, 0 ]

    return utility_hole

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


class active_learner():
    def __init__(self,
                df_initial,
                dir_scratch='/gpfs/scratch/pr47fo/ge28quz2/ge28quz2/1',
                #dir_scratch='/dss/dssfs02/lwp-dss-0001/pr47fo/pr47fo-dss-0000/ge28quz2/xtb_code/kappa_test',
                dir_dss='/dss/dssfs02/lwp-dss-0001/pr47fo/pr47fo-dss-0000/ge28quz2/xtb_code/kappa/20240319/mutation_size/last',
                dir_save='/dss/dsshome1/lxc09/ge28quz2/1_proj/20230926_parameter/20240319/mutation_size/last',
                properties=['e_lambda_h','homo'],
                default_values=[0.,0.],
                random_state=42,
                N_mutation_batch=20,
                N_mutation_limit=800,#200
                kappa=0.0,
                N_select_batch=100,#200
                #N_generation=2,
                reduced_search_space=1,
                two_generations_search = 0
                ):
        #print(type(df_initial))
        #self.dir_results=os.path.join(dir_scratch, dir_results)
        self.two_generations_search = two_generations_search
        self.reduced_search_space=reduced_search_space
        self.df_population=df_initial #with generation column 0
        #self.df_population['mutation_status']=np.nan
        #self.df_population.loc[:,'calculation_status']=np.nan
        #self.df_population['generation']=0
        #self.df_population.loc[:,'added_in_round']=0
        #self.df_population.to_json('/dss/dsshome1/lxc09/ge28quz2/1_proj/20220418_aml/0428test/df_initial.json',orient="split")
        self.properties=properties
        self.ml_column='morgan'
        self.random_state=random_state
        self.kernel_parameters={}
        for prop_name in self.properties:
            self.kernel_parameters[prop_name]={'C':1., 'length_scale':1., 'sigma_n':0.1}

        self.generation_counter=0
        self.added_in_round=0
        self.list_mol_smis_mutated=[]
        self.dir_scratch=dir_scratch
        self.dir_dss=dir_dss
        self.dir_results=os.path.join(dir_dss, 'calculations_{}'.format(self.added_in_round))
        dir_calculation=os.path.join(dir_scratch, 'calculations')
        self.dir_calculation=dir_calculation
        #df_save=os.path.join(dir_save,'df_population_{}.json'.format(self.added_in_round))
        self.dir_save=dir_save
        dir_smi_log=os.path.join(dir_dss, 'smi.log')
        self.dir_smi_log=dir_smi_log
        #os.mkdir(self.dir_smi_log)
        if os.path.isfile(self.dir_smi_log): os.system('rm {}'.format(self.dir_smi_log))
        #if not os.path.isdir(self.dir_smi_log): 
        #    print(self.dir_smi_log)
        #    os.mkdir(self.dir_smi_log)

        self.N_mutation_batch=N_mutation_batch
        self.N_mutation_limit=N_mutation_limit
        self.depth_search=1

        #self.MLModels = {}
        self.kappa=kappa
        self.N_select_batch=N_select_batch

        self.list_columns_frame = ['smi', 'operation', 'mol_last_gen', 'generation', 'added_in_round']
        self.df_already_mutated = pd.DataFrame(columns=self.list_columns_frame)
        self.max_origin_idx=np.max(self.df_population.origin_idx.tolist())

    def generate_ml_vectors(self, df, symmetry=False):
        """ Helper: Add ML-vectors to dataframe """
        self.fp_radius = 2
        fps=[]
        for i,row in df.iterrows():
            try:
                if symmetry:
                    fps.append(AllChem.GetMorganFingerprint(Chem.MolFromSmiles(row.symmetry_smi), self.fp_radius))
                else:
                    fps.append(AllChem.GetMorganFingerprint(Chem.MolFromSmiles(row.smi), self.fp_radius))
            except:
                fps.append(np.nan)
                continue

        df[self.ml_column] = [[]]*df.shape[0]
        df[self.ml_column] = df[self.ml_column].astype(object)
        df[self.ml_column] = fps
        df=df.loc[df[self.ml_column].isnull()==False]
        return df

    def get_gpr_model(self, prop_name):
        """ Helper: Fit ML model on one property """

        self.df_completed=self.df_population.loc[self.df_population['calculation_status']=='calculated']
        #self.df_completed=self.df_population.loc[~self.df_population.homo.isnull()]
        #df_population_unique = self.get_unique_df(self.df_completed)
        df_population_unique = self.generate_ml_vectors(self.df_completed,symmetry=True)

        print('')
        print('Fitting property: {}'.format(prop_name))
        print('Size of fitting set for gpr model (get_gpr_model): {}'.format(df_population_unique.shape[0]))

        x_train = np.array(df_population_unique[self.ml_column].tolist())
        y_train = df_population_unique[prop_name].to_numpy()
        #print('debug y_train 1 :',len(y_train))
        df_tmp=deepcopy( df_population_unique )
        kernel_params = self.kernel_parameters[prop_name]
        niter_local=5
        gpr = GPR_tanimoto()
        gpr = optimize_GPR( x_train, y_train, gpr,
                            starting_values=[kernel_params['C'],
                                            kernel_params['length_scale'],
                                            kernel_params['sigma_n']],
                            niter_local=niter_local,
                            random_state=self.random_state)
        kernel_params['C']=gpr.constant_kernel
        kernel_params['length_scale']=gpr.gamma_kernel
        kernel_params['sigma_n']=gpr.sigma_white
        self.kernel_parameters[prop_name] = kernel_params
        os.chdir(self.dir_save)
        df_tmp.loc[:,'y_train']=y_train
        df_tmp.loc[:,'y_train_predict']=gpr.predict(x_train)
        df_tmp.to_json('df_{}_{}.json'.format(prop_name,max(df_population_unique['added_in_round']) ),orient='split')
        os.system('echo "{} {} {} {} {}" >> kernel_params.txt'.format(kernel_params['C'],
                                                                    kernel_params['length_scale'],
                                                                    kernel_params['sigma_n'],
                                                                    max(df_population_unique['added_in_round'].tolist()),
                                                                    prop_name ))

        print('Fitted Kernel c {} rbf {} sigma {} Round: {}'.format(kernel_params['C'],
                                                                kernel_params['length_scale'],
                                                                kernel_params['sigma_n'],
                                                                max(df_population_unique['added_in_round'].tolist()) ))
        return gpr, x_train

    def get_unique_df(self, df):
        df_unique=df.drop_duplicates(subset='smi')
        return df_unique

    def calculate_properties(self):
        #if not os.path.isdir('molecules_to_calculate'): os.mkdir('molecules_to_calculate')
        if self.generation_counter==0: return
        cwd=os.getcwd()
        #if os.path.isdir(self.dir_calculation): os.system('rm -rf {}'.format(self.dir_calculation))
        os.chdir(self.dir_calculation)
        #df_population_unique=self.get_unique_df(self.df_population)
        smiles_set_to_process=sorted(list(set(self.df_population.loc[self.df_population['calculation_status'].isnull(),'smi'].to_numpy())))
        #idx_set_to_process=[]

        atoms_all_list=[]
        #origin_idx_list=[]
        idx_all_list=[]
        df_to_calculate=self.df_population.loc[self.df_population['calculation_status'].isnull()]
        for i, row in df_to_calculate.iterrows():
            atoms=read('mol_res_opt_{}.xyz'.format(row['idx']))
            atoms_all_list.append(atoms)
            idx_all_list.append(row['idx'])

        #os.mkdir(self.dir_calculation)
        #os.chdir(self.dir_calculation)
        print('idx_to_calculate', idx_all_list)
        pool = Pool()
        res_atoms = pool.map(calculate_xtb, atoms_all_list, idx_all_list)
        pool.terminate()
        pool.restart()
        #pool.close()
        #pool.join()
        #os.chdir(self.dir_scratch)

        df_lst=[]
        for atoms in res_atoms:
            if atoms is not None:
                df_lst.append(atoms.info)

        df_result = pd.DataFrame(df_lst)
        
        df_result.to_json(os.path.join(self.dir_save, 'df_check3.json'), orient='split')
        
        for i, row in df_result.iterrows():
            self.df_population.loc[self.df_population.idx==row.idx, 'e_lambda_h']=row.e_lambda_h
            self.df_population.loc[self.df_population.idx==row.idx, 'homo']=row.homo
            #self.df_population.loc[self.df_population.idx==row.idx, 'e_lambda_e']=row.e_lambda_e
            self.df_population.loc[self.df_population.idx==row.idx, 'calculation_status']=row.calculation_status
        self.df_population = self.df_population.loc[~self.df_population.calculation_status.isnull()]

        self.df_population.loc[self.df_population.e_lambda_h>1000.0, 'calculation_status']='fizzled'
        self.df_population.loc[self.df_population.e_lambda_h<0.0, 'calculation_status']='fizzled'
        #print(self.dir_save, self.added_in_round)
        self.df_population.to_json(os.path.join(self.dir_save,'df_population_{}.json'.format(self.added_in_round)), orient="split")
        #os.chdir(cwd)
        #print('dir_results',self.dir_results)
        self.dir_results=os.path.join(self.dir_dss, 'calculations_{}'.format(self.added_in_round))
        #print('dir_results',self.dir_results, self.added_in_round)
        if os.path.isdir(self.dir_results): os.system('rm -rf {}'.format(self.dir_results))
        if not os.path.isdir(self.dir_results): os.mkdir(self.dir_results)
        #print('check the dir_results',self.dir_results )
        #os.mkdir(self.dir_results)
        #dont save
        #shutil.rmtree(self.dir_calculation )
        file_names = os.listdir(self.dir_calculation)
        for file_name in file_names:
            #os.remove( os.path.join(self.dir_calculation, file_name) )
            #shutil.rmtree(os.path.join(self.dir_calculation, file_name))
            shutil.move(os.path.join(self.dir_calculation, file_name), self.dir_results)
        #os.rename(self.dir_calculation, self.dir_results)
        os.system('rm -rf {}'.format(self.dir_calculation))

    def get_new_frames(self, mol_smis):
        pool = mp.Pool()
        #print('mol_smis',mol_smis)
        new_frames = pool.map(morphing.run_mutation, mol_smis)
        pool.close()
        pool.join()
        new_frames_all = [item for sublist in new_frames for item in sublist]
        return new_frames_all

    def get_similarity(self, df_candidates):
        df_candidates=self.get_unique_df(df_candidates)
        df_candidates.reset_index(drop=True,inplace=True)
        smi_list=df_candidates.smi.to_list()
        fps=[]
        for smi in smi_list:
            mol=Chem.MolFromSmiles(smi)
            #print(mol)
            if mol!=None:
                fp = AllChem.GetMorganFingerprint(mol, 2)
                fps.append(fp)
        df_candidates_copy=df_candidates.copy()
        #df_candidates.to_json('df_test.json',orient='split')
        df_candidates.loc[:,'similarity']=np.nan
        for idx, row in df_candidates_copy.iterrows():
            mol=Chem.MolFromSmiles(row['smi'])
            if mol==None: continue
            else:
                fp = AllChem.GetMorganFingerprint(mol, 2)
                #print(idx,np.mean(DataStructs.BulkTanimotoSimilarity(fp,fps)))
                df_candidates.loc[idx,'similarity']=np.mean(DataStructs.BulkTanimotoSimilarity(fp,fps))
        return df_candidates.loc[~df_candidates.similarity.isnull()]

    def get_symm_similarity(self, df_candidates):
        df_candidates=self.get_unique_df(df_candidates)
        df_candidates.reset_index(drop=True,inplace=True)
        smi_list=df_candidates.smi.to_list()
        fps=[]
        for smi in smi_list:
            mol=Chem.MolFromSmiles(smi)
            #print(mol)
            if mol!=None:
                fp = AllChem.GetMorganFingerprint(mol, 2)
                fps.append(fp)
        df_candidates_copy=df_candidates.copy()
        #df_candidates.to_json('df_test.json',orient='split')
        df_candidates.loc[:,'similarity']=np.nan
        for idx, row in df_candidates_copy.iterrows():
            mol=Chem.MolFromSmiles(row['symmetry_smi'])
            if mol==None: continue
            else:
                fp = AllChem.GetMorganFingerprint(mol, 2)
                #print(idx,np.mean(DataStructs.BulkTanimotoSimilarity(fp,fps)))
                df_candidates.loc[idx,'similarity']=np.mean(DataStructs.BulkTanimotoSimilarity(fp,fps))
        return df_candidates.loc[~df_candidates.similarity.isnull()]


    def generate_candidates(self):
        self.generation_counter+=1
        df_population_unique=self.get_unique_df(self.df_population.loc[~self.df_population.calculation_status.isnull()])
        #df_population_unique=self.get_unique_df(self.df_population.loc[self.df_population.calculation_status=='calculated'])
        df_to_morph=df_population_unique[~df_population_unique.smi.isin(self.list_mol_smis_mutated)]
        self.list_mol_smis_mutated+=df_to_morph.smi.tolist()
        with open(os.path.join(self.dir_save ,'list_mol_smis_mutated_{}.pkl'.format(self.added_in_round)), 'wb') as f:
            pickle.dump( self.list_mol_smis_mutated, f)
        #print('check',df_to_morph.smi.tolist())
        new_frames=self.get_new_frames(df_to_morph.smi.tolist())
        # Lookahead search mutation on all unique ones
        mutate_list_lookahead = list(set([x.smi[0] for x in new_frames]))
        unique_mols_generated_lookahead=[]
        for i in range(self.two_generations_search):
            new_frames_lookahead = self.get_new_frames( mutate_list_lookahead)
            mutate_list_lookahead = list(set([x.smi[0] for x in new_frames_lookahead]))
            new_frames += new_frames_lookahead
            unique_mols_generated_lookahead += list(set([x.smi[0] for x in new_frames_lookahead]))

        n_unique_mols_generated = len(list(set([x.smi[0] for x in new_frames])))
        print('Generated by morphing: {}, Unique ones: {}'.format(len(new_frames), n_unique_mols_generated))
        print('Thereby generated by two-fold morphing: {}'.format(len(set(unique_mols_generated_lookahead))))
        df_candidates = pd.concat(new_frames)
        ###its better to count the limit one molecule by one molecule

        df_candidates = df_candidates[df_candidates.smi!=None]
        df_candidates['generation'] = self.generation_counter
        #df_candidates['added_in_round'] = self.generation_counter
        df_candidates['added_in_round'] = self.added_in_round
        self.df_already_mutated = pd.concat([self.df_already_mutated, df_candidates])

    def reduced_search_selection(self, df_candidates_unique, random_state=-1):
        """ This is the method for search space reduction  """

        df_candidates_identified = df_candidates_unique[df_candidates_unique.smi=='']
        df_candidates_identified_unique = df_candidates_identified.copy()
        self.generation_counter+=1

        if random_state<0: random_state=self.random_state

        # Lets select initial set of molecules
        smi_selected_current_round = RW_sm_selection( self.get_symm_similarity(self.get_unique_df(\
                            self.df_population.loc[self.df_population.calculation_status=='calculated']) ),
                            field='similarity',n_sel=self.N_mutation_batch, random_state = random_state )

        np.random.seed(random_state)


        for d in range(self.depth_search):
            #print('current in tree search loop',d)
            if d==0:

                df_candidates_identified_current = self.df_already_mutated[\
                    self.df_already_mutated.mol_last_gen.isin( smi_selected_current_round ) ]

                #if len(set(smi_selected_current_round) - set(self.df_already_mutated.mol_last_gen.tolist())):
                #    print('Error, d=0: Not all molecules had already been morphed')

                df_candidates_identified_current_unique = df_candidates_identified_current\
                                            [~df_candidates_identified_current.smi.\
                                            isin(self.df_population.smi)].drop_duplicates(subset='smi')

            else:
                df_candidates_identified_current = pd.concat(self.get_new_frames( smi_selected_current_round))

            df_candidates_identified_current_unique = df_candidates_identified_current[\
                            ~df_candidates_identified_current.smi.isin(self.df_population.smi)].drop_duplicates(subset='smi')

            print('Selection round depth {}. Morphed {} molecules. Candidates identified {}'.format(d,
                                          len(smi_selected_current_round), df_candidates_identified_current_unique.shape[0] ) )

            df_candidates_identified_unique=self.get_similarity(df_candidates_identified_current_unique)
            #df_candidates_identified_unique.to_json('df_tmp.json',orient='split')
            smi_selected_current_round = RW_sm_selection(df_candidates_identified_unique, n_sel=self.N_mutation_batch,
                                                           field='similarity', random_state=random_state)
            df_candidates_identified = pd.concat( [ df_candidates_identified_current,  df_candidates_identified ] )


        df_candidates_identified['generation'] = self.generation_counter
        df_candidates_identified['added_in_round'] = self.added_in_round
        df_candidates_identified_unique = df_candidates_identified[~df_candidates_identified.smi.isin(\
                                        self.df_population.smi)].drop_duplicates(subset='smi')

        #df_candidates_identified_unique['Fi_scores'] = 0.
        #last_index = np.max(self.df_population_unique.origin_idx.tolist())+1
        df_candidates_identified_unique=self.get_similarity(df_candidates_identified_unique )
        df_candidates_identified_unique=RW_sm_selection_df( df_candidates_identified_unique,self.N_mutation_limit,field='similarity', random_state=random_state )
        #df_candidates_identified_unique = df_candidates_identified_unique.sort_values( by = 'similarity', ascending=True)
        #df_candidates_identified_unique = df_candidates_identified_unique
        #df_candidates_identified_unique.to_json('df_test.json',orient='split')
        df_candidates_identified_unique.drop(['similarity'],axis=1,inplace=True)
        last_index=self.max_origin_idx+1
        print(last_index)
        df_candidates_identified_unique['origin_idx']=[x+last_index for x in range(df_candidates_identified_unique.shape[0])]
        self.max_origin_idx=np.max(df_candidates_identified_unique.origin_idx.tolist())
        #print(df_candidates_identified_unique.origin_idx.tolist() )
        #print( )
        print(self.max_origin_idx)
        #df_candidates_identified_unique.to_json('df_tmp.json',orient='split')
        print( 'Tree selection finished. Identified candidates: {}'.format(df_candidates_identified_unique.shape[0]) )
        #df_candidates_identified_unique = self.generate_ml_vectors(df_candidates_identified_unique,symmetry=False)

        return df_candidates_identified_unique



    def get_utility(self, y, kappa=-1., stds=[], return_array=False):
        """ Compute scalarized utility (F and Facq) as described in the article """

        utility_values = []
        std_hole_values = []

        y = np.array(y)
        if len(y.shape) == 1: y = np.array(y, ndmin=2).T
        stds = np.array(stds)
        if len(stds.shape) == 1: stds = np.array(stds, ndmin=2).T

        if kappa < 0.: kappa = self.kappa

        for i in range(y.shape[1]):
            if len(stds) == 0:
                utility_values.append(get_F(y[:, i], kappa=kappa,
                                            return_array=return_array))
            else:
                utility_values.append(get_F(y[:, i], std=stds[:, i],
                                            kappa=kappa, return_array=return_array))
                std_hole_values.append(utility_values[-1][1])

        return np.array(utility_values)



    def select_and_add_new_candidates(self):
        self.added_in_round+=1
        self.generate_candidates()
        df_candidates = self.df_already_mutated

        df_candidates_clean = df_candidates[ ~df_candidates.smi.isin(self.df_population.smi) ]
        df_candidates_unique = self.get_unique_df(df_candidates_clean)

        print('------------')
        print('Candidates: {}, Unique: {}'.format(df_candidates_clean.shape[0], df_candidates_unique.shape[0]))

        # In the AML model, we always fit on completed data
        # Candidates are derived from full population
        n_select=self.N_select_batch
        #df_candidates_unique = self.selection_step(df_candidates_unique, n_select).copy()
        df_added=self.selection_step(df_candidates_unique, n_select).copy()
        #print('Candidates selected (unique): {}'.format( df_candidates_unique.shape[0] ))
        df_added['calculation_status']=np.nan
        #df_candidates_unique['added_in_round']=self.added_in_round

        df_added.to_json(os.path.join(self.dir_save,'df_added_check_added.json'),orient='split')


        self.df_population=pd.concat([self.df_population,df_added], ignore_index=True) ###check is the concat really works
        #self.df_population.to_json('df_population_before_calculate_{}.json'.format(self.added_in_round),orient="split")
        print('The size of current df_population: {}'.format(len(self.df_population)))
        print('------------')

        # Run unfinished calculations on the population
        # Note, that population will only be written, when check_all_calculations_finished is called
        #self.run_calculations_population()

    def selection_step(self, df_candidates_unique, n_select):
        """ Active learner selection step: Either select from full candidate list or resort
            to the reduction of search space. """

        try: os.system('rm *_mat.npy')
        except: pass

        if self.reduced_search_space>0:

            df_candidates_unique = df_candidates_unique.loc[df_candidates_unique.smi=='']

            df_candidates_identified_unique = self.reduced_search_selection(df_candidates_unique)
            #df_candidates_identified_unique.loc[~df_candidates_identified_unique.origin_idx.isnull()]
        df_candidates_identified_unique.reset_index(drop=True, inplace=True)
        #df_candidates_identified_unique.to_json('df_check_idx_{}.json'.format(self.added_in_round),orient='split')
        #df_candidates_atoms=self.df_population[self.df_population.smi=='']#empty
        #origin_idxs=sorted([i for i in range(max(self.max_orign_idx)+1)]*4)
        print(self.max_origin_idx)
        atoms_all_list=[]
        atoms_idx=[]

        for i, row in df_candidates_identified_unique.iterrows():
            smi=row['smi']
            origin_idx=row['origin_idx']
            idx=int((origin_idx)*8)
            m_woH=Chem.MolFromSmiles(smi)
            if m_woH==None: continue
            N_woH=len(m_woH.GetAtoms())
            m=Chem.AddHs(m_woH)
            atoms=rdkit2ase(m)
            extreme_points=find_backbone_sites(m_woH)
            atoms1=generate_symmetric_BB(N_woH, atoms, extreme_points[0], extreme_points[1])
            atoms_idx.append(idx)
            atoms2=generate_symmetric_BB(N_woH, atoms, extreme_points[1], extreme_points[0])
            atoms_idx.append(idx+1)
            atoms3=generate_symmetric_BB(N_woH, atoms, extreme_points[0], extreme_points[1],connect_triple=1)
            atoms_idx.append(idx+2)
            atoms4=generate_symmetric_BB(N_woH, atoms, extreme_points[1], extreme_points[0],connect_triple=1)
            atoms_idx.append(idx+3)
            
            #atoms5,6,7,8 annelation 
            start_point=extreme_points[0]
            end_point=extreme_points[1]
            two_neighbours, neighbour_sites=find_neighbour_sites(m_woH,extreme_points[0])
            #print('check:',len(neighbour_sites) )
            if len(neighbour_sites)==0:
                atoms5=None
                atoms6=None
            if len(neighbour_sites)==1:
                atoms5=None
                atoms6=generate_annelation_BB(N_woH,atoms, start_point, end_point, neighbour_sites[0], two_neighbours)
            if len(neighbour_sites)==2:
                atoms5=generate_annelation_BB(N_woH,atoms, start_point, end_point, neighbour_sites[0], two_neighbours)
                atoms6=generate_annelation_BB(N_woH,atoms, start_point, end_point, neighbour_sites[1], two_neighbours)
            
            start_point=extreme_points[1]
            end_point=extreme_points[0]
            two_neighbours, neighbour_sites=find_neighbour_sites(m_woH,extreme_points[1])
            #print('check:',len(neighbour_sites) )
            if len(neighbour_sites)==0:
                atoms7=None
                atoms8=None
            if len(neighbour_sites)==1:
                atoms7=None
                atoms8=generate_annelation_BB(N_woH,atoms, start_point, end_point, neighbour_sites[0], two_neighbours)
            if len(neighbour_sites)==2:
                atoms7=generate_annelation_BB(N_woH,atoms, start_point, end_point, neighbour_sites[0], two_neighbours)
                atoms8=generate_annelation_BB(N_woH,atoms, start_point, end_point, neighbour_sites[1], two_neighbours)
            atoms_idx.append(idx+4)
            atoms_idx.append(idx+5)
            atoms_idx.append(idx+6)
            atoms_idx.append(idx+7)

            atoms_all_list.append(atoms1)
            atoms_all_list.append(atoms2)
            atoms_all_list.append(atoms3)
            atoms_all_list.append(atoms4)
            
            atoms_all_list.append(atoms5)
            atoms_all_list.append(atoms6)
            atoms_all_list.append(atoms7)
            atoms_all_list.append(atoms8)

        if os.path.isdir(self.dir_calculation): os.system('rm -rf {}'.format(self.dir_calculation))
        if not os.path.isdir(self.dir_calculation): os.mkdir(self.dir_calculation )
        os.chdir(self.dir_calculation)

        pool = Pool()
        res_atoms = pool.map(atoms_optimization, atoms_all_list, atoms_idx)
        pool.terminate()
        pool.restart()
        #pool.candidates_optimization(atoms_all_list)
        #os.chdir(self.dir_scratch)
        print('after atoms optimization',os.getcwd())
        df_lst=[]
        for atoms in res_atoms:
            if atoms is not None:
                #if '.' not in atoms.info['symmetry_smi']:
                #    print('check,',atoms.info['symmetry_smi'] )
                if '.' not in atoms.info['symmetry_smi'] and check_smi_valence(Chem.MolFromSmiles(atoms.info['symmetry_smi'])):
                    df_lst.append(atoms.info)
        df_candidates_atoms = pd.DataFrame(df_lst)
        df_candidates_identified_unique.to_json(os.path.join(self.dir_save, 'df_check1.json'), orient='split')
        df_candidates_atoms.to_json(os.path.join(self.dir_save, 'df_check2.json'), orient='split')
        df_candidates=pd.merge(df_candidates_identified_unique,df_candidates_atoms,on='origin_idx')
        df_candidates_clean=self.generate_ml_vectors(df_candidates,symmetry=True)
        print('In this round {} atoms optimization and {} survived'.format(len(df_candidates_identified_unique) ,\
                len(df_candidates_atoms)) )
        y_preds=[]
        stds=[]
        x_test=df_candidates_clean[self.ml_column].to_numpy()
        for prop_name in self.properties:
            #gpr=self.MLmodels[prop_name]
            gpr, x_train = self.get_gpr_model(prop_name)
            #y_tmp, std_tmp=gpr.predict(x_train, return_std=True)
            y_test_pred, std=gpr.predict(x_test, return_std=True)
            df_candidates_clean.loc[:,'predict_'+prop_name]=y_test_pred
            y_preds.append(y_test_pred)
            stds.append(std)

        Fi_scores=self.get_utility(y_preds, stds=stds, return_array=True)
        #df_candidates['utility']=self.get_utility(y)
        df_candidates_clean.loc[:,'Fi']=list(Fi_scores[:,0])
        #df_selected = df_candidates_clean.loc[df_candidates_clean.Fi>=-0.2]
        #if len(df_selected) < n_select:
        df_selected = df_candidates_clean.iloc[np.flip(\
                        np.argsort(df_candidates_clean['Fi'].tolist()),axis=0)[0:n_select]].copy()
        #print(n_select)
        #df_selected.to_json('df_selected_{}.json'.format(self.added_in_round),orient='split')
        print('In this round {} molecules selected'.format(df_selected.shape[0]))
        return df_selected

if __name__ == "__main__":
    cwd=os.getcwd()
    df_initial=pd.read_json('df_initial_with_smi.json',orient='split')
    #df_initial=df_initial.iloc[:10]
    #df_initial=df_initial.loc[~df_initial.smi.isnull()]
    df_initial=df_initial.loc[~df_initial.smi.isnull()]
    df_initial=df_initial.loc[df_initial.e_lambda_h>0]
    df_initial=df_initial.loc[df_initial.e_lambda_h<1000]
    #df_candidates_identified_unique = self._predict_Fi_scores( df_candidates_identified_current_unique )
    #adding utility
    df_initial.loc[:,'utility']=0
    for idx, row in df_initial.iterrows():
        df_initial.loc[idx,'utility']=get_F([row['e_lambda_h'], row['homo']], ideal_points=[0.0,-5.1], weights=[1.0,0.7], std=[], kappa=0., return_array=False)
    #df_initial.loc[:,'calculation_status']='calculated'
    #df_initial.loc[0,'mutation_status']='mutated'
    df_initial.loc[:,'added_in_round'] = 0
    df_initial.loc[:,'generation'] = 0
    df_initial.loc[df_initial.smi!='c1ccccc1','generation'] = 1
    #df_initial.loc[65:,'generation'] = 2

    df_initial.to_json('df_population_0.json', orient='split')

    AL=active_learner(df_initial)
    n_learning_steps=20


    for i in range(n_learning_steps):
        print('Learning Step {} / {} start...'.format(i+1,n_learning_steps))
        start_time = time.time()
        #AL.calculate_properties()
        #print('DFTB calculation cost time: {}'.format(time.time()-start_time))
        time.sleep(30)
        AL.select_and_add_new_candidates()
        AL.calculate_properties()
        print('Time for Learning Step {}:  {}'.format(i+1, time.time()-start_time))

        pickle.dump(AL, open(os.path.join(cwd,'AL_{}.pkl').format(i),'wb'))
