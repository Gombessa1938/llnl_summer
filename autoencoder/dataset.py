import numpy as np 
pl = np.load('dataset/rts_gmlc_pl.npy')
va = np.load('dataset/rts_gmlc_va.npy')
vm = np.load('dataset/rts_gmlc_vm.npy')

concat = np.concatenate((pl,va,vm),axis=1)
np.save('concat.npy',concat)