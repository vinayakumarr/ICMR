import os
from scipy.io import wavfile
import numpy as np

with file('murmur.csv', 'w') as outfile:
  for file in os.listdir('btrain/Btraining_murmur/Btraining_murmur'):
    
    if file.endswith(".wav"):
        
        rate, wf = wavfile.read('btrain/Btraining_murmur/Btraining_murmur/'+ file)
        print(type(wf))
        np.savetxt(outfile, [wf],fmt='%01d',delimiter=',')
        print(wf.shape)
