'''
augment.py

This script augments the paced audio clips
with noise, reverb, etc for more robust
classification since the original task may
be too easy.
'''

import sox

tfm = sox.Transformer()