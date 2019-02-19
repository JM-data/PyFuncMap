#import torch
#DEVICE = torch.device('cuda:0')

def regularizer_descriptor_preservation_torch(C, Desc_src, Desc_tar):
    '''
    fMap regularizer: preserve the given corresponding descriptors (projected)
    :param C: functional map : source -> target
    :param Desc_src: the projected descriptors of the source shape
    :param Desc_tar: the projected descriptors of the target shape
    :return: the objective and the gradient of this regularizer
    '''


def regularizer_laplacian_commutativity_torch(C, eval_src, eval_tar):
    '''
    fMap regularizer: Laplacian-Beltrami operator commutativity term
    :param C: functional map: source -> target (matrix k2-by-k1)
    :param eval_src: eigenvalues of the source shape (length of k1)
    :param eval_tar: eigenvalues of the target shape (length of k2)
    :return: the objective and the gradient of this regularizer
    '''



def regularizer_operator_commutativity_torch(C, CommOp_src, CommOp_tar):
    '''
    fMap regularizer: preserve the descriptor via commutativity
    :param C: functional map: source -> target
    :param CommOp_src: a list of desc_commutativity_operators of the source shape
    :param CommOp_tar: the corresponding desc_comm_op of the target shape
    :return: the objective and the gradient of this regularizer
    '''
