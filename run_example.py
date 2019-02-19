import numpy as np
import sys
sys.path.insert(0, './utils/')
import MESH
import FunctionalMap as fMap
import MeshProcess
from scipy.sparse import csr_matrix, spdiags

DATASET_PATH = 'FAUST_shapes_off/'

# Load two meshes
S1 = MESH.mesh_load_and_preprocess(DATASET_PATH + "tr_reg_000.off",
                                   numEigs=50)

S2 = MESH.mesh_load_and_preprocess(DATASET_PATH + "tr_reg_001.off",
                                   numEigs=50)

# compute the wave kernel signatures
desc1 = fMap.wave_kernel_signature(S1.evecs, S1.evals, S1.A, numTimes=100, ifNormalize=True)
desc1 = desc1[:, np.arange(0,100,20)]
desc2 = fMap.wave_kernel_signature(S2.evecs, S2.evals, S2.A, numTimes=100, ifNormalize=True)
desc2 = desc2[:, np.arange(0,100,20)]

B1 = S1.evecs[:, 0:5]
B2 = S2.evecs[:, 0:5]

Ev1 = S1.evals[0:5]
Ev2 = S2.evals[0:5]


Desc1 = fMap.descriptors_projection(desc1, B1, S1.A)
Desc2 = fMap.descriptors_projection(desc2, B2, S2.A)



# CommOp1 = fMap.descriptor_commutativity_operator(desc1, B1, S1.A)
# CommOp2 = fMap.descriptor_commutativity_operator(desc2, B2, S2.A)

OrientOp1 = fMap.descriptor_orientation_operator(desc1, B1, S1)
OrientOp2 = fMap.descriptor_orientation_operator(desc2, B2, S2)


C = desc1[0:5, 0:5]
fval, grad = fMap.regularizer_operator_commutativity(C, OrientOp1, OrientOp2, IfReversing=True)



print(fval)
print("\n\n")
print(grad)