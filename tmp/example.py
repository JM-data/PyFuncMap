import numpy as np
import scipy.optimize as opt
import scipy.io as sio
from scipy.spatial import cKDTree
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import time

from extra import *
from solve_torch import *

#Choose size of functional maps
n_vecs = 60

#Choose two shapes. For examples tr_reg_001.mat and tr_reg_002.mat
i = 0
j = 1

t0 = time.time()

# Input Source, with LB eigenvectors and descriptors
print("Loading Source shape...")
t1 = time.time()
source_mat = './Data/tr_reg_%.3d.mat' % i
source_off = './FAUST_shapes_off/tr_reg_%.3d.off' % i
S1 = read_off(source_off)
source_evals, source_evecs, source_evecs_trans = S_info(S1, n_vecs)

source_data = sio.loadmat(source_mat)
source_shot = source_data['target_shot'] #They are all called target
print("Done in %.2f seconds" % (time.time()-t1))

# Input Target, with LB eigenvectors and descriptors
print("Loading Target shape...")
t3 = time.time()
target_mat = './Data/tr_reg_%.3d.mat' % j
target_off = './FAUST_shapes_off/tr_reg_%.3d.off' %j
S2 = read_off(target_off)
target_evals, target_evecs, target_evecs_trans = S_info(S2, n_vecs)

target_data = sio.loadmat(target_mat)
target_shot = target_data['target_shot']
print("Done in %.2f seconds" % (time.time()-t3))

# Ground-Truth functional map since mapping is one-to-one
C_gt = np.dot(source_evecs_trans, target_evecs) 

# Least-Square if choose C_
#A_ = np.transpose(np.dot(source_evecs_trans, source_shot))
#B_ = np.transpose(np.dot(target_evecs_trans, target_shot))
A = np.dot(source_evecs_trans, source_shot)
B = np.dot(target_evecs_trans, target_shot)
#C_ = np.linalg.lstsq(A_, B_, rcond=None)[0]

L1 = np.diag(source_evals)
L2 = np.diag(target_evals)

print('Pre-computing the multiplication operators...')
t4 = time.time()
P1_list = []
P2_list = []
k = 20
for i in range(k):
	eig1 = np.dot(np.dot(source_evecs_trans, np.diag(source_shot[:,i])), source_evecs)
	eig2 = np.dot(np.dot(target_evecs_trans, np.diag(target_shot[:,i])), target_evecs)
	P1_list.append(eig1)
	P2_list.append(eig2)
print('Done in %.2f seconds' % (time.time()-t4))


print('Optimizing the functional map...')
C = solve(A, B, k, L1, L2, P1_list, P2_list, 1e-1, 1e-3, 1)

# See Original paper from Maks in 2012 for explanation on KDTree
print("Calculating Matches...")
t5 = time.time()
target_gt = np.dot(C_gt, target_evecs.T)
target = np.dot(C, target_evecs.T)
kdt = cKDTree(source_evecs)
dist, indices_gt = kdt.query(target_gt.T, n_jobs = -1)
dist, indices = kdt.query(target.T, n_jobs = -1)
print('Done in %.2f seconds' % (t5-time.time()))

print('First 15 Matches of Ground Truth and Optimized FM:')
print(indices_gt[:15]+1)#+1 since Python starts at 0 and Matlab 1
print(indices[:15]+1)
print("Total runtime : %.2f seconds" % (time.time() - t0))

# Visualize Functional Maps
fig = plt.figure(figsize = (40,20))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(C_gt)
ax1.set_title('Ground Truth Functional Map', fontsize=20)
ax2.imshow(C)
ax2.set_title('Estimated Functional Map', fontsize=20)
plt.show()
# You can compare results from the ground truth map and the calculated one

