import os
from scipy.io import wavfile
import numpy as np

with file('murmur.csv', 'w') as outfile:
  for file in os.listdir('atrain/Atraining_murmur/Atraining_murmur'):
    
    if file.endswith(".wav"):
        
        rate, wf = wavfile.read('atrain/Atraining_murmur/Atraining_murmur/'+ file)
        print(rate)
        np.savetxt(outfile, [wf],fmt='%01d',delimiter=',')
        
