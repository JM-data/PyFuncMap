import numpy as np
import sys
sys.path.insert(0, './utils/')
import MESH
import FunctionalMap as fMap
import MeshProcess
from scipy.sparse import csr_matrix, spdiags
from scipy.optimize import minimize

DATASET_PATH = 'FAUST_shapes_off/'
k1 = 5
k2 = 5


S1 = MESH.mesh_load_and_preprocess(DATASET_PATH + "tr_reg_000.off",
                                   numEigs=50)

S2 = MESH.mesh_load_and_preprocess(DATASET_PATH + "tr_reg_001.off",
                                   numEigs=50)

desc1 = fMap.wave_kernel_signature(S1.evecs, S1.evals, S1.A, numTimes=100, ifNormalize=True)
desc1 = desc1[:, np.arange(0,100,20)]
desc2 = fMap.wave_kernel_signature(S2.evecs, S2.evals, S2.A, numTimes=100, ifNormalize=True)
desc2 = desc2[:, np.arange(0,100,20)]

B1 = S1.evecs[:, 0:k1]
B2 = S2.evecs[:, 0:k2]

Ev1 = S1.evals[0:k1]
Ev2 = S2.evals[0:k2]


Desc1 = fMap.descriptors_projection(desc1, B1, S1.A)
Desc2 = fMap.descriptors_projection(desc2, B2, S2.A)

CommOp1 = fMap.descriptor_commutativity_operator(desc1, B1, S1.A)
CommOp2 = fMap.descriptor_commutativity_operator(desc2, B2, S2.A)


# OrientOp1 = fMap.descriptor_orientation_operator(desc1, B1, S1)
# OrientOp2 = fMap.descriptor_orientation_operator(desc2, B2, S2)


funDesc= lambda C: fMap.regularizer_descriptor_preservation(C, Desc1, Desc2)[0]
gradDesc = lambda C: fMap.regularizer_descriptor_preservation(C, Desc1, Desc2)[1]

funCommLB = lambda C: fMap.regularizer_laplacian_commutativity(C, Ev1, Ev2)[0]
gradCommLB = lambda C: fMap.regularizer_laplacian_commutativity(C, Ev1, Ev2)[1]


funCommDesc = lambda C: fMap.regularizer_operator_commutativity(C, CommOp1, CommOp2)[0]
gradCommDesc = lambda C: fMap.regularizer_operator_commutativity(C, CommOp1, CommOp2)[1]

a = 1
b = 1
c = 1

param = dict()
param['fMap_size'] = [50, 50]
param['weight_descriptor_preservation'] = 1
param['weight_laplacian_commutativity'] = 1
param['weight_descriptor_commutativity'] = 1
param['weight_descriptor_orientation'] = 1

myObj = lambda C: a*funDesc(C) + b*funCommDesc(C) + c*funCommLB(C)
myGrad = lambda C: a*gradDesc(C) + b*gradCommDesc(C) + c*gradCommLB(C)


constDesc = np.sign(B1[0, 0]*B2[0, 0])*np.sqrt(np.sum(S2.A) / np.sum(S1.A))
Cini = np.zeros((k2, k1))
Cini[0, 0] = constDesc


vec_Cini = np.reshape(Cini, (k1 * k2, 1))
vec_myObj = lambda vec_C: myObj(np.reshape(vec_C, (k2, k1)))
vec_myGrad = lambda vec_C: np.reshape(myGrad(np.reshape(vec_C, (k2, k1))), (k2*k1, 1))


x = vec_Cini
tmp = x[np.arange(0, k1*k2, k1)]
first_col_C = np.zeros((k2, 1))
first_col_C[0] = constDesc

myObj_new = lambda Cvar: myObj(np.concatenate((first_col_C, Cvar), axis=1))
myGrad_new = lambda Cvar: myGrad(np.concatenate((first_col_C, Cvar), axis=1))[:,1:k1]
vec_myObj_new = lambda vec_Cvar: myObj_new(np.reshape(vec_Cvar, (k2, k1-1)))
vec_myGrad_new = lambda vec_Cvar: np.reshape(myGrad_new(np.reshape(vec_Cvar, (k2, k1-1))), (k2*(k1-1), 1))

vec_Cini_new = np.zeros((k2*(k1-1),1))

res = minimize(vec_myObj_new, vec_Cini_new, method='SLSQP',
               jac=vec_myGrad_new,
               options={'disp': True})


Copt = np.reshape(res.x,(k2, k1-1))
Copt_full = np.concatenate((first_col_C, Copt), axis = 1)
print(Copt_full)