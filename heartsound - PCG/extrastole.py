import os
from scipy.io import wavfile
import numpy as np

with file('extrastole.csv', 'w') as outfile:
  for file in os.listdir('btrain/Btraining_extrasystole/Btraining_extrastole'):
    
    if file.endswith(".wav"):
        
        rate, wf = wavfile.read('btrain/Btraining_extrasystole/Btraining_extrastole/'+ file)
        print(type(wf))
        np.savetxt(outfile, [wf],fmt='%01d',delimiter=',')
        print(wf.shape)
