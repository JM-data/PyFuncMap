import numpy as np
import sys
sys.path.insert(0, './utils/')
import MESH
import FunctionalMap as fMap
import MeshProcess
from scipy.sparse import csr_matrix, spdiags

DATASET_PATH = 'FAUST_shapes_off/'

S1 = MESH.mesh_load_and_preprocess(DATASET_PATH + "tr_reg_000.off",
                                   numEigs=50)

S2 = MESH.mesh_load_and_preprocess(DATASET_PATH + "tr_reg_001.off",
                                   numEigs=50)

desc1 = fMap.wave_kernel_signature(S1.evecs, S1.evals, S1.A, numTimes=100, ifNormalize=True)
desc1 = desc1[:, np.arange(0,100,20)]
desc2 = fMap.wave_kernel_signature(S1.evecs, S1.evals, S1.A, numTimes=100, ifNormalize=True)
desc2 = desc2[:, np.arange(0,100,20)]

B1 = S1.evecs
B2 = S2.evecs

a = np.sum(desc1)
print(a)