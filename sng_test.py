#Parker Conroy
#Test file for playing with Keras and word generators 


#While the sng package is not installed, add the package's path
# (the parent directory) to the library path:

import os
import sys
#sys.path.insert(0, os.path.abspath('../../'))
import tensorflow as tf
from tensorflow import keras
#import keras
import sng

#https://stackoverflow.com/questions/35863825/how-to-print-unsupported-unicode-characters-on-windows-cmd-as-e-g-instead-o
old_stdout = sys.stdout
fd = os.dup(sys.stdout.fileno())
sys.stdout = open(fd, mode='w', errors='replace')
old_stdout.close()

#from tensorflow.keras.models import Sequential

cfg = sng.Config(
epochs=250, #how many cycles?
max_word_len=20,
min_word_len=6,
n_layers = 2,
temperature= .75 #how random is the letter generation?
)
cfg.to_dict()

canna = sng.load_builtin_wordlist('canna.txt')
canna[:5]

gen = sng.Generator(wordlist=canna, config=cfg)
gen.fit()
print('done fitting')
gen.simulate(n=4)
print('done sim')
gen.save('my_model', overwrite=True)