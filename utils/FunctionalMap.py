import numpy as np
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

# TODO: check
def descriptor_commutativity_operator(desc, B, A):
    '''
    Operator construction for descriptor preservation via commutativity
    See paper "Informative Descriptor Preservation via Commutativity for Shape Matching"
    by Dorian Nogneng and Maks Ovsjanikov
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

# TODO: check
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
        fval = (torch.mm(C, Desc_src) - Desc_tar).pow(2).sum()
        grad = torch.mm(torch.mm(C, Desc_src) - Desc_tar, Desc_src.transpose())
        return fval, grad

# TODO: check
def regularizer_descriptor_commutativity(C, CommOp_src, CommOp_tar):
    '''
    fMap regularizer: preserve the descriptor via commutativity
    :param C: functional map: source -> target
    :param CommOp_src: a list of desc_commutativity_operators of the source shape
    :param CommOp_tar: the corresponding desc_comm_op of the target shape
    :return: the objective and the gradient of this regularizer
    '''
    if len(CommOp_src) != len(CommOp_tar):
        return -1
    else:
        fval = 0
        grad = torch.zeros(C.shape)
        for i in range(len(CommOp_src)):
            X = CommOp_src[i]
            Y = CommOp_tar[i]
            fval += (torch.mm(X, C) - torch.mm(C, Y)).pow(2).sum()
            grad += 2*(torch.mm(X.tranpose(), torch.mm(X, C) - torch.mm(C, Y)) -
                     torch.mm(torch.mm(X, C) - torch.mm(C, Y), Y.transpose()))
        return fval, grad

# TODO: check
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
        Mask = np.power(np.tile(eval_tar, (1, len(eval_src))) -
                        np.tile(eval_src.transpose(), (len(eval_tar),1)),2)
        Mask = torch.div(Mask, Mask.pow(2))
        fval = torch.bmm(torch.bmm(C, C), Mask)
        grad = 2*torch.bmm(C, Mask)
        return fval, grad

