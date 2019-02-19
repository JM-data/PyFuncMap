import numpy as np
import MeshProcess
from scipy.optimize import minimize
#import torch
#DEVICE = torch.device('cuda:0')


def heat_kernel_signature(evecs, evals, A, numTimes, ifNormalize=True):
    '''
    Compute a set of Heat Kernel Signatures (HKS)
    :param evecs: the eigenfunctions of the Laplacian-beltrami operator (n-by-k)
    :param evals: the corresponding eigenvalues (k-by-1)
    :param A: the area matrix of the shape S (n-by-n)
    :param numTimes: the number of descriptors (to control the scale)
    :param ifNormalize: if we want to normalize the descriptors w.r.t the surface area
    :return: a set of descriptors (n-by-numTimes)
    '''

    D = np.matmul(evecs.transpose(), np.matmul(A.toarray(), np.power(evecs, 2)))
    abs_evals = np.abs(evals)
    emin = abs_evals[1]
    emax = abs_evals[-1]
    t = np.array(np.linspace(start=emin, stop=emax, num=numTimes), ndmin=2)
    T = np.exp(-np.abs(np.matmul(evals, t)))
    hks = np.matmul(evecs, np.matmul(D, T))
    if ifNormalize:
        hks = descriptors_normalization(hks, A)
    return hks


def wave_kernel_signature(evecs, evals, A, numTimes, ifNormalize=True):
    '''
    Compute a set of Wave Kernel Signatures (WKS)
    :param evecs: the eigenfunctions of the Laplacian-beltrami operator (n-by-k)
    :param evals: the corresponding eigenvalues (k-by-1)
    :param A: the area matrix of the shape S (n-by-n)
    :param numTimes: the number of descriptors (to control the scale)
    :param ifNormalize: if we want to normalize the descriptors w.r.t the surface area
    :return: a set of descriptors (n-by-numTimes)
    '''
    num_eigs = evecs.shape[1]
    D = np.matmul(evecs.transpose(), np.matmul(A.toarray(), np.power(evecs, 2)))
    abs_evals = np.abs(evals)
    emin = np.log(abs_evals[1])
    emax = np.log(abs_evals[-1])
    s = 7*(emax - emin) / numTimes
    emin = emin + 2*s
    emax = emax - 2*s

    t = np.array(np.linspace(start=emin, stop=emax, num=numTimes), ndmin=2)
    tmp1 = np.tile(np.log(abs_evals), (1, numTimes))
    tmp2 = np.tile(t, (num_eigs, 1))
    T = np.exp(-np.power((tmp1 - tmp2), 2) / (2 * s ** 2))
    wks = np.matmul(evecs, np.matmul(D, T))
    if ifNormalize:
        wks = descriptors_normalization(wks, A)
    return wks


def descriptors_normalization(desc, A):
    '''
    Normalize the descriptors w.r.t. the mesh area
    :param desc: a set of descriptors defined on mesh S
    :param A: the area matrix of the mesh S
    :return: the normalized desc with the same size
    '''
    tmp1 = np.sqrt(np.matmul(desc.transpose(), np.matmul(A.toarray(), desc)).diagonal())
    tmp2 = np.tile(tmp1, (desc.shape[0], 1))
    desc_normalized = np.divide(desc, tmp2)
    return desc_normalized


def descriptors_projection(desc, B, A):
    '''
    Project the given descriptors into the given basis (for fMap computation)
    :param desc: a set of descriptors defined on mesh S
    :param B: the basis of mesh S for the projection (should be consistent with the fMap)
    :param A: the area matrix of the mesh S
    :return: a #LB-by-#desc matrix
    '''
    desc_projected = np.matmul(B.transpose(), np.matmul(A.toarray(), desc))
    return desc_projected


def descriptor_commutativity_operator(desc, B, A):
    '''
    Operator construction for descriptor preservation via commutativity
    See paper "Informative Descriptor Preservation via Commutativity for Shape Matching"
    by Dorian Nogneng, and Maks Ovsjanikov
    :param desc: a set of NORMALIZED descriptors
    :param B: the basis where the fMap lives in
    :param A: the area matrix of the corresponding shape
    :return: a LIST of operators to be preserved by a fMap via commutativity,
            each operator is a #LB-by-#LB matrix, there are #desc operators in total
    '''
    num_desc = desc.shape[1]
    num_basis = B.shape[1]
    CommOp = []
    for i in range(num_desc):
        tmp = np.tile(desc[:,i], [num_basis,1]).transpose()
        op = np.matmul(B.transpose(), np.matmul(A.toarray(), np.multiply(tmp, B)))
        CommOp.append(op)
    return CommOp


def descriptor_orientation_operator(desc, B, S):
    '''
    Operator construction for orientation preserving/reversing via commutativity
    See paper "Continuous and orientation-preserving correspondence via functional maps"
    by Jing Ren, Adrien Poulenard, Peter Wonka, and Maks Ovsjanikov
    :param desc: a set of NORMALIZED descriptors
    :param B: the basis where the fMap lives in
    :param S: the mesh where the descriptors computed from
    :return: a LIST of operators encoded with the orientation information
    '''
    num_desc = desc.shape[1]
    OrientOp = []
    for i in range(num_desc):
        f = desc[:, i]
        op = MeshProcess.compute_orientation_operator_from_a_descriptor(S, B, f)
        OrientOp.append(op)
    return OrientOp


def regularizer_laplacian_commutativity(C, eval_src, eval_tar):
    '''
    fMap regularizer: Laplacian-Beltrami operator commutativity term
    :param C: functional map: source -> target (matrix k2-by-k1)
    :param eval_src: eigenvalues of the source shape (length of k1)
    :param eval_tar: eigenvalues of the target shape (length of k2)
    :return: the objective and the gradient of this regularizer
    '''
    if len(eval_tar) != C.shape[0] or len(eval_src) != C.shape[1]:
        return -1
    else:
        Mask = np.power(np.tile(eval_tar / np.sum(eval_tar) * np.sum(eval_src), (1, len(eval_src))) -
                        np.tile(eval_src.transpose() / np.sum(eval_src) * np.sum(eval_tar), (len(eval_tar), 1)), 2)
        Mask = np.divide(Mask, np.sum(np.power(Mask, 2)))
        fval = 0.5 * np.sum(np.multiply(np.multiply(C, C), Mask))
        grad = np.multiply(C, Mask)
        return fval, grad


def regularizer_descriptor_preservation(C, Desc_src, Desc_tar):
    '''
    fMap regularizer: preserve the given corresponding descriptors (projected)
    :param C: functional map : source -> target
    :param Desc_src: the projected descriptors of the source shape
    :param Desc_tar: the projected descriptors of the target shape
    :return: the objective and the gradient of this regularizer
    '''
    if Desc_src.shape[1] != Desc_tar.shape[1]:
        return -1
    else:
        fval = 0.5 * np.sum(np.power(np.matmul(C, Desc_src) - Desc_tar, 2))
        grad = np.matmul(np.matmul(C, Desc_src) - Desc_tar, Desc_src.transpose())
        return fval, grad


def regularizer_operator_commutativity(C, Op_src, Op_tar, IfReversing=False):
    '''
    fMap regularizer: preserve the operators via commutativity
    the operators can be:
    - descriptor commutativity operator
    - orientation-preserving operator
    :param C: functional map: source -> target
    :param Op_src: a list of operators on the source shape
    :param Op_tar: the corresponding operators on the target shape
    :param IfReversing: if reversing the commutativity sign - used in orientation-reversing setting
    :return: the objective and the gradient of this regularizer
    '''
    if len(Op_src) != len(Op_tar):
        return -1
    else:
        fval = 0
        grad = np.zeros(C.shape)
        for i in range(len(Op_src)):
            X = Op_tar[i]
            Y = Op_src[i]
            if not IfReversing:
                fval += 0.5 * np.sum(np.power(np.matmul(X, C) - np.matmul(C, Y), 2))
                grad += (np.matmul(X.transpose(), np.matmul(X, C) - np.matmul(C, Y)) -
                         np.matmul(np.matmul(X, C) - np.matmul(C, Y), Y.transpose()))
            else: # used for orientation-reversing setting only!
                fval += 0.5 * np.sum(np.power(np.matmul(X, C) + np.matmul(C, Y), 2))
                grad += (np.matmul(X.transpose(), np.matmul(X, C) + np.matmul(C, Y)) +
                         np.matmul(np.matmul(X, C) + np.matmul(C, Y), Y.transpose()))
        return fval, grad


def compute_functional_map_from_descriptors(S1, S2, desc1, desc2, param):
    '''
    Given a pair of shapes and a set of corresponding descriptors, compute a functional map
    :param S1: the source shape
    :param S2: the target shape
    :param desc1: a set of descriptors defined on S1
    :param desc2: the corresponding descriptors defined on S2
    :param param: the paramters for functional map optimization
    :return: a functional map from S1 to S2
    '''
    k1 = param['fMap_size'][0]
    k2 = param['fMap_size'][1]
    weights = np.array([param['weight_descriptor_preservation'],
                        param['weight_laplacian_commutativity'],
                        param['weight_descriptor_commutativity'],
                        param['weight_descriptor_orientation']])

    B1 = S1.evecs[:, 0:k1]
    B2 = S2.evecs[:, 0:k2]
    Ev1 = S1.evals[0:k1]
    Ev2 = S2.evals[0:k2]

    Desc1 = descriptors_projection(desc1, B1, S1.A)
    Desc2 = descriptors_projection(desc2, B2, S2.A)

    CommOp1 = descriptor_commutativity_operator(desc1, B1, S1.A)
    CommOp2 = descriptor_commutativity_operator(desc2, B2, S2.A)

    # OrientOp1 = descriptor_orientation_operator(desc1, B1, S1)
    # OrientOp2 = descriptor_orientation_operator(desc2, B2, S2)

    # Construct the energy function

    funDesc = lambda C: regularizer_descriptor_preservation(C, Desc1, Desc2)[0]
    gradDesc = lambda C: regularizer_descriptor_preservation(C, Desc1, Desc2)[1]

    funCommLB = lambda C: regularizer_laplacian_commutativity(C, Ev1, Ev2)[0]
    gradCommLB = lambda C: regularizer_laplacian_commutativity(C, Ev1, Ev2)[1]

    funCommDesc = lambda C: regularizer_operator_commutativity(C, CommOp1, CommOp2)[0]
    gradCommDesc = lambda C: regularizer_operator_commutativity(C, CommOp1, CommOp2)[1]

    # TODO: it is too slow to compute the orientation operator, comment it for now
    # funOrient = lambda C: regularizer_operator_commutativity(C, OrientOp1, OrientOp2)[0]
    # gradOrient = lambda C: regularizer_operator_commutativity(C, OrientOp1, OrientOp2)[1]

    # myObj = lambda C: (weights[0] * funDesc(C) + weights[1] * funCommLB(C) +
    #                   weights[2] * funCommDesc(C) + weights[3] * funOrient(C))
    # myGrad = lambda C: (weights[0] * gradDesc(C) + weights[1] * gradCommLB(C) +
    #                   weights[2] * gradCommDesc(C) + weights[3] * gradOrient(C))
    myObj = lambda C: (weights[0] * funDesc(C) + weights[1] * funCommLB(C) +
                       weights[2] * funCommDesc(C))
    myGrad = lambda C: (weights[0] * gradDesc(C) + weights[1] * gradCommLB(C) +
                       weights[2] * gradCommDesc(C))

    # the first column of the functional map is determinant (defined by the first Eigenfunctions)
    constDesc = np.sign(B1[0, 0] * B2[0, 0]) * np.sqrt(np.sum(S2.A) / np.sum(S1.A))
    first_col_C = np.zeros((k2, 1))
    first_col_C[0] = constDesc

    # i.e., the 2nd to the k1-th columns are the true variables to optimize
    myObj_new = lambda Cvar: myObj(np.concatenate((first_col_C, Cvar), axis=1))
    myGrad_new = lambda Cvar: myGrad(np.concatenate((first_col_C, Cvar), axis=1))[:, 1:k1]

    # vectorize the variable and the corresponding gradient for the solver
    vec_myObj_new = lambda vec_Cvar: myObj_new(np.reshape(vec_Cvar, (k2, k1 - 1)))
    vec_myGrad_new = lambda vec_Cvar: np.reshape(myGrad_new(np.reshape(vec_Cvar, (k2, k1 - 1))), (k2 * (k1 - 1), 1))

    # optimization
    vec_Cvar = np.zeros((k2 * (k1 - 1), 1))
    res = minimize(vec_myObj_new, vec_Cvar, method='SLSQP',
                   jac=vec_myGrad_new,
                   options={'disp': True})

    # reshape the minimum to a matrix
    Copt = np.reshape(res.x, (k2, k1 - 1))
    # add the fixed first column back to get the complete functional map
    Copt_full = np.concatenate((first_col_C, Copt), axis=1)
    return Copt_full



def convert_functional_map_to_pointwise_map(C12, B1, B2):
    '''
    Pointwise map reconstruction
    :param C12: given functional map C12: S1 -> S2
    :param B1: the basis of S1
    :param B2: the basis of S2
    :return: T21: the pointwise map T21: S2 -> S1
    '''


def convert_pointwise_map_to_funcitonal_map(T12, B1, B2):
    '''
    Convert a pointwise map to a functional map
    :param T12: given pointwise map T12: S1 -> S2
    :param B1: the basis of S1
    :param B2: the basis of S2
    :return: C21: the corresponding functional map C21: S2 -> S1
    '''


def refine_fMap_icp(C12, B1, B2, num_iters=10):
    '''
    Regular Iterative Closest Point (ICP) to refine a functional map
    :param C12: initial functional map from S1 to S2
    :param B1: the basis of S1
    :param B2: the basis of S2
    :param num_iters: the number of iterations for refinement
    :return: C12_refined, T21_refined
    '''


def refine_pMap_icp(T12, B1, B2, num_iters=10):
    '''
    Regular Iterative Closest Point (ICP) to refine a pointwise map
    :param T12: initial pointwise map from S1 to S2
    :param B1: the basis of S1
    :param B2: the basis of S2
    :param num_iters: the number of iterations for refinement
    :return: T12_refined, C21_refined
    '''


# --------------------------------------------------------------------------
#  Energy terms in torch
# --------------------------------------------------------------------------
# TODO: check
def regularizer_descriptor_preservation_torch(C, Desc_src, Desc_tar):
    '''
    fMap regularizer: preserve the given corresponding descriptors (projected)
    :param C: functional map : source -> target
    :param Desc_src: the projected descriptors of the source shape
    :param Desc_tar: the projected descriptors of the target shape
    :return: the objective and the gradient of this regularizer
    '''


# TODO: check
def regularizer_laplacian_commutativity_torch(C, eval_src, eval_tar):
    '''
    fMap regularizer: Laplacian-Beltrami operator commutativity term
    :param C: functional map: source -> target (matrix k2-by-k1)
    :param eval_src: eigenvalues of the source shape (length of k1)
    :param eval_tar: eigenvalues of the target shape (length of k2)
    :return: the objective and the gradient of this regularizer
    '''



# TODO: check
def regularizer_operator_commutativity_torch(C, CommOp_src, CommOp_tar):
    '''
    fMap regularizer: preserve the descriptor via commutativity
    :param C: functional map: source -> target
    :param CommOp_src: a list of desc_commutativity_operators of the source shape
    :param CommOp_tar: the corresponding desc_comm_op of the target shape
    :return: the objective and the gradient of this regularizer
    '''

# --------------------------------------------------------------------------
#  Energy terms in torch - End
# --------------------------------------------------------------------------