import os
from scipy.io import wavfile
import numpy as np

with file('trainartifact.csv', 'w') as outfile:
  for file in os.listdir('atrain/Atraining_artifact/Atraining_artifact'):
    
    if file.endswith(".wav"):
        
        rate, wf = wavfile.read('atrain/Atraining_artifact/Atraining_artifact/'+ file)
        print(rate)
        np.savetxt(outfile, [wf],fmt='%01d',delimiter=',')
        #print(wf.shape)
