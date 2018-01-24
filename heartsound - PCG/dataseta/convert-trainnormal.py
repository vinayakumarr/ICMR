import os
from scipy.io import wavfile
import numpy as np

with file('trainnormal.csv', 'w') as outfile:
  for file in os.listdir('atrain/Atraining_normal/Atraining_normal'):
    
    if file.endswith(".wav"):
        
        rate, wf = wavfile.read('atrain/Atraining_normal/Atraining_normal/'+ file)
        print(rate)
        np.savetxt(outfile, [wf],fmt='%01d',delimiter=',')
        #print(wf.shape)
      
