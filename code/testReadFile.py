# test for reading mat file

from os.path import dirname, join as pjoin
import scipy.io as sio

mat_contents = sio.loadmat('testFile.mat')

print(mat_contents['FullObsTasksStruct'].shape)