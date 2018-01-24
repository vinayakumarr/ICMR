import os
from scipy.io import wavfile
import numpy as np

with file('trainnormal.csv', 'w') as outfile:
  for file in os.listdir('btrain/Btraining_normal/Training B Normal'):
    
    if file.endswith(".wav"):
        
        rate, wf = wavfile.read('btrain/Btraining_normal/Training B Normal/'+ file)
        print(type(wf))
        np.savetxt(outfile, [wf],fmt='%01d',delimiter=',')
        print(wf.shape)
        print(rate)
