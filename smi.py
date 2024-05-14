import os
from tqdm import tqdm
filenames=os.listdir('smi_logs')
for filename in tqdm(filenames):
	with open('smi_logs/{}'.format(filename)) as f:
		lines=f.readlines()

	with open('smi.log','a') as f:
		f.write(filename+'____'+lines[0]+'\n')
