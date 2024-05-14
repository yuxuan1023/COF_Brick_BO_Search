#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle

from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

import rdkit
#print(rdkit.__version__)
from rdkit import rdBase,Chem
from rdkit.Chem import AllChem,Draw
from rdkit.Chem import DataStructs
#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from config import dir_dss


def generate_ml_vectors(df):
    fp_radius = 2 
    fps=[]
    for i,row in df.iterrows():
        fps.append(AllChem.GetMorganFingerprint(Chem.MolFromSmiles(row.smi), fp_radius))
                        
    df['morgan'] = [[]]*df.shape[0]
    df['morgan'] = df['morgan'].astype(object)
    df['morgan'] = fps
    return df


# In[3]:


def count_morgan_fp_matrix(mol, radius=2, size=2048):
    ''' Generate a fixed-length Morgan count-fingerprint '''

    fp = AllChem.GetHashedMorganFingerprint(mol, radius, size)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp,arr)
    return np.array(arr, dtype=np.int64) 


# In[4]:


def generate_morgan_matrix(x_array, radius=2, size=2048):
    morgan_list=[]
    for mol in x_array:
        fp = AllChem.GetHashedMorganFingerprint(Chem.MolFromSmiles(smi), radius, size)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp,arr)
        mol_array = np.array(arr, dtype=np.int64) 
        morgan_list.append(mol_array)
    #numpy.concatenate( LIST, axis=0 )
    return np.vstack( morgan_list)
        


# In[5]:


class GPR_tanimoto():
    def __init__(self, optimize=True):
        self.is_fit = False
        self.x_train, self.y_train = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize
        self.D_scratch_dir=dir_dss
        self.normalize_y=True
    def set_kernel_params(self, constant_kernel, gamma_kernel, sigma_white):
        self.constant_kernel=constant_kernel
        self.gamma_kernel=gamma_kernel
        self.sigma_white=sigma_white
    def get_kernel(self, dist):
        return self.constant_kernel**2*(1.-dist)
    
    def get_tanimoto_distmat(self,X1,X2=[]):
        sim_mat  =  []
        if X2==[]:
            for  i,x in enumerate(X1):
                sims  =  DataStructs.BulkTanimotoSimilarity(x,list(X1[i+1:]))
                sim_mat.extend(sims)
            sim_mat = squareform(sim_mat)
            sim_mat = sim_mat+np.eye(sim_mat.shape[0],sim_mat.shape[1])
        else:
            for i,x in enumerate(X1):
                sims = DataStructs.BulkTanimotoSimilarity(x,list(X2))
                sim_mat.append(sims)
            sim_mat=np.array(sim_mat)
        return 1.-sim_mat 
    
    def _kernel_gradient(self,K,D):
        return np.dstack(( (np.full((K.shape[0], K.shape[1]), 2*self.constant_kernel,
                            dtype=np.array(self.constant_kernel).dtype)*K)[:, :, np.newaxis],
                           ( self.constant_kernel**2* np.zeros(K.shape) )[:, :, np.newaxis], 
                           (np.eye(len(self.x_train), len(self.x_train)) * 2*self.sigma_white)[:, :, np.newaxis]
                        ))   
    def log_marginal_likelihood(self, eval_gradient=False):
        
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", self.y_train, self.alpha)
        log_likelihood_dims -= np.log(np.diag(self.L)).sum()
        log_likelihood_dims -= len(self.x_train) / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
        
        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", self.alpha, self.alpha)  # k: output-dimension
            tmp -= cho_solve((self.L, True), np.eye(len(self.x_train)))[:, :, np.newaxis]
            log_likelihood_gradient_dims = 0.5 * np.einsum("ijl,jik->kl", tmp, self.K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)
            return float(log_likelihood), log_likelihood_gradient

        else: 
            return float(log_likelihood)
    
    def fit(self, x_train, y_train, refit=False):
        dmat_loc = os.path.join(self.D_scratch_dir, 'D_mat.npy')
        if os.path.isfile(dmat_loc):
            D=np.load(dmat_loc)
            if not D.shape[0]==len(x_train):
                #D=self.distance_matrix(x_train)
                D=self.get_tanimoto_distmat(x_train)
                np.save(dmat_loc,D)
        else:
            #D = self.distance_matrix(x_train)
            D=self.get_tanimoto_distmat(x_train)
            np.save(dmat_loc,D)
        
        self.x_train = x_train
        
        K = self.get_kernel(D)
        
        K += np.eye(len(x_train),len(x_train)) * self.sigma_white**2
        self.K_gradient = self._kernel_gradient(K,D)

        # Scale to 0 mean unit variance
        if self.normalize_y:
            self._y_train_mean = np.mean(y_train, axis=0)
            self._y_train_std = np.std(y_train, axis=0)
            y_train = (y_train - self._y_train_mean) / self._y_train_std
            self.y_train = np.array(y_train, ndmin=2).T
        else:
            self.y_train = np.array(y_train, ndmin=2).T
        
        try:
            self.L = cholesky(K, lower=True)  
        except np.linalg.LinAlgError: 
            print('Linalg error, continuing with last value')
            os.system('echo "Linalg error {}" >> laerrs.txt '.format(len(self.y_train)))           
 
        self.alpha = cho_solve((self.L, True), self.y_train)  
        L_inv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))
        self.K_inv = L_inv.dot(L_inv.T)

    def predict(self, x_test, return_std=False, return_absolute_std=True):   
        d_star_mat_loc = os.path.join(self.D_scratch_dir, 'D_star_mat.npy')
        if os.path.isfile(d_star_mat_loc):
            D_star=np.load(d_star_mat_loc)
            if D_star.shape[0]!=len(x_test) or D_star.shape[1]!=len(self.x_train):
                D_star = self.get_tanimoto_distmat(x_test, self.x_train) 
        else:
            D_star = self.get_tanimoto_distmat(x_test, self.x_train) 
        np.save(d_star_mat_loc, D_star)       
                
        K_star = self.get_kernel(D_star)
        self.K_star=K_star # rm
        K_star_star = self.get_kernel(np.array([self.get_tanimoto_distmat([x],[x])[0][0] for x in x_test])) #+ self.sigma_white**2
        
        #print('debug',len(self.y_train))
        y_star = np.dot(np.dot(K_star, self.K_inv), self.y_train) 
        if self.normalize_y: y_star = self._y_train_std * y_star + self._y_train_mean # undo normalization
            
        if return_std:
            var_y=[]
            for i, k_star in enumerate(K_star):
                var_y.append(K_star_star[i] - np.dot( np.dot(k_star, self.K_inv), k_star.T ))
            if self.normalize_y: 
                var_y = np.array(var_y)
                if return_absolute_std: var_y = var_y * self._y_train_std**2 # undo normalization
            
            if np.any(np.isnan(np.squeeze(np.sqrt(var_y)))):
                print('Problem: Nan detected in std')
            
            return np.squeeze(y_star), np.squeeze(np.sqrt(var_y))
                
        
        return np.squeeze(y_star)
    


# In[6]:


def optimize_GPR(x_train, y_train, gpr_model, starting_values=[1.0, 1., 0.1], 
                              pbounds = {'c':(0.1,3.0),'rbf': (1.,1.),'alpha':(0.001,1.0)}, 
                              niter_local=1, random_state=1):
    random_state = check_random_state(random_state)
    res_vals=[]
    res_dicts=[]
    i=0 
    for nit in range(niter_local):
        def log_marginal_likelihood_target_localopt(x, verbose=False):
            nonlocal x_train, y_train, i, gpr_model
            i+=1
            gpr_model.set_kernel_params(x[0], x[1], x[2])
            gpr_model.fit(x_train,y_train)
            log,gradlog=gpr_model.log_marginal_likelihood(eval_gradient=True)
            if verbose: print('localopt',i, x, log, gradlog)
            return -log, -gradlog
        
        if nit!=0:
            starting_values = [random_state.uniform(pbounds['c'][0],pbounds['c'][1]),
                               random_state.uniform(pbounds['rbf'][0],pbounds['rbf'][1]),
                               random_state.uniform(pbounds['alpha'][0],pbounds['alpha'][1])]
        print('')
        print('initial guess',starting_values)
        res = minimize(log_marginal_likelihood_target_localopt, 
                   starting_values, 
                   method="L-BFGS-B", jac=True, options={'eps':1e-5}, 
                   bounds=[pbounds["c"], pbounds["rbf"], pbounds["alpha"]])
        print('Local (L-BFGS-B) opt {} finished'.format(nit), res['fun'], res['x'])
        res_dicts.append(res)
        res_vals.append(res['fun'])
        res=res_dicts[res_vals.index(min(res_vals))]
    
    gpr_model.set_kernel_params(res['x'][0], res['x'][1], res['x'][2])
    print('Best solution after local opt:', res['fun'], res['x'])
    
    gpr_model.fit(x_train, y_train)    
    #print('debug y_train 2:',len(y_train)) 
    #print(pbounds)
    return gpr_model
        


# In[16]:


def get_train_test_dfs(df_data, train_size=0.9, test_size=0.1, shuffle_data=True):
    #if shuffle_data==True: shuffle(df_data, )
    print('--------------------train size: {}--------------------'.format(train_size))
    df_train, df_test = train_test_split(df_data,train_size=train_size, test_size=test_size, random_state=42, shuffle= shuffle_data)
    return df_train, df_test


# In[19]:


def get_results(df_all, train_size=0.9, test_size=0.1, shuffle_data=True):
    #df_all=pd.read_json('df_1000L.json', orient='split')
    #df_tot=df_all[['e_lambda_e','smi']]
    df_tot=df_all[['vbm','smi']]
    df_tot_ml=generate_ml_vectors(df_tot) #smi, elambda,morgan
    df_data=df_tot_ml[['morgan','vbm']]
    df_train, df_test=get_train_test_dfs(df_data, train_size, test_size, shuffle_data)
    print('*********************{}*************************'.format(len(df_train)))
    kernel_parameters={'C':1., 'length_scale':1., 'sigma_n':0.1}
    n_iter=5
    gpr = GPR_tanimoto()

    x_train=df_train['morgan'].tolist()
    y_train=df_train['vbm'].to_numpy()
    gpr=optimize_GPR(x_train, y_train, gpr, starting_values=[1.0, 1., 0.1], 
                              pbounds = {'c':(0.1,3.0),'rbf': (1.,1.),'alpha':(0.001,1.0)}, 
                              niter_local=5, random_state=1)
    x_test=df_test['morgan'].tolist()
    y_test=df_test['vbm'].to_numpy()
    y_pred=gpr.predict(x_test, return_std=False, return_absolute_std=True)
    mse=mean_squared_error(y_test,y_pred)
    return mse, mse**0.5




