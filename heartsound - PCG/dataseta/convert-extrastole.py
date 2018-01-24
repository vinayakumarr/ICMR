import os
from scipy.io import wavfile
import numpy as np

with file('extrastole.csv', 'w') as outfile:
  for file in os.listdir('atrain/Atraining_extrahs/Atraining_extrahls'):
    
    if file.endswith(".wav"):
        
        rate, wf = wavfile.read('atrain/Atraining_extrahs/Atraining_extrahls/'+ file)
        print(rate)
        np.savetxt(outfile, [wf],fmt='%01d',delimiter=',')
        
