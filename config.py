import os

#cwd=os.getcwd()
global dir_calculation
dir_calculation=os.path.join('/gpfs/scratch/pr47fo/ge28quz2/ge28quz2/1','calculations')
#dir_calculation=os.path.join('/dss/dssfs02/lwp-dss-0001/pr47fo/pr47fo-dss-0000/ge28quz2/xtb_code/kappa_test','calculations')

global dir_dss
dir_dss='/dss/dssfs02/lwp-dss-0001/pr47fo/pr47fo-dss-0000/ge28quz2/xtb_code/kappa/20240319/mutation_size/last'

global dir_smi_log
dir_smi_log=os.path.join(dir_dss, 'smi.log')
#dir_smi_log=os.path.join('/dss/dssfs02/lwp-dss-0001/pr47fo/pr47fo-dss-0000/ge28quz2/kappa/3.0','smi_logs')

#global dir_save
#dir_save='/dss/dsshome1/lxc09/ge28quz2/1_proj/20230216_aml/3.0'
