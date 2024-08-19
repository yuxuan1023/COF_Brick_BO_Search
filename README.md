The main code is aml_v20230915.py, run it directly by 'python aml_v20230915.py'. All the rest python scripts will be called automatically.
'df_initial_with_smi.json',the initial population where the search started;
'config.py':constructing configurations bwtween different scripts;
'morphing_design.py':generating evolutionary bricks;
'generate_symmetry.py':forming building blocks from bricks; 
'gpr_model.py':building Gaussain progress regression models;
'xtb_calculation.py':running xTB calculations.

Please change the directories setting in the aml_v20230915.py and config.py.
'dir_scratch':where the calclation is running;
'dir_dss':current working directory and where the xtb results will be stored;
'dir_save':where to save the generated population dataframe;
'sub_tiny.cmd' is an example script code to submit the job to high performance computer based on LRZ cluster(https://doku.lrz.de/high-performance-computing-10613431.html).

Prerequired packages:
import os, time, socket, shutil, random, subprocess, pickle
import math
import numpy as np
import pandas as pd
from copy import deepcopy
from functools import partial
import multiprocessing as mp
import rdkit
import ase
from openbabel import pybel
An example conda environment setting 'environment.yml' is provided.
