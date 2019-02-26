import time
import numpy as np
import torch
DEVICE = torch.device('cuda:0') 
import torch.optim as optim
from torch.nn import functional as F
import warnings

def compute_loss(C, A, B, k, L1, L2, P1, P2, alpha, beta, gamma):
    l1 = (torch.mm(C, A) - B).pow(2).sum()
    #print(l1)
    l2 = (torch.mm(L2, C) - torch.mm(C, L1)).pow(2).sum()
    #print(l2)
    C_ = C.repeat(k,1,1)
    l3 = (torch.bmm(C_, P1) - torch.bmm(P2, C_)).pow(2).sum()
    #print(l3)
    return alpha * l1 + beta * l2 + gamma * l3

def solve_torch(n, C_, **kwargs):
    TOL = 1e-6
    MAXITER = int(1e6)
    LR = 1e-4
    C = torch.tensor(C_, requires_grad=True, dtype=torch.float32, device=DEVICE)
    current_loss = np.inf
    #optimizer = optim.RMSprop([C], lr=LR)
    optimizer = optim.Adam([C], lr=LR)
    for _ in range(MAXITER):
        loss = compute_loss(C,  **kwargs)
        loss.backward()
        optimizer.step()
        if loss.item() > np.abs(current_loss - TOL):
            print(current_loss)
            return C
        current_loss = loss.item()
    warnings.warn('Failed to achieve convergence in {} iterations'.format(MAXITER))
    return C

def solve(A, B, k, L1, L2, P1_list, P2_list, alpha, beta, gamma):
    n = A.shape[0]
    C_ = np.linalg.lstsq(np.transpose(A), np.transpose(B), rcond=None)[0]
    torch_args = {
        'A':torch.tensor(A, requires_grad=False, dtype=torch.float32, device=DEVICE),
        'B':torch.tensor(B, requires_grad=False, dtype=torch.float32, device=DEVICE),
		'k':k,
        'L1':torch.tensor(L1, requires_grad=False, dtype=torch.float32, device=DEVICE),
        'L2':torch.tensor(L2, requires_grad=False, dtype=torch.float32, device=DEVICE),
		'P1':torch.tensor(P1_list, requires_grad=False, dtype=torch.float32, device=DEVICE),	        
		'P2':torch.tensor(P2_list, requires_grad=False, dtype=torch.float32, device=DEVICE),
		'alpha':alpha,
        'beta':beta,
		'gamma':gamma
    }
    ts = time.time()
    C_torch = solve_torch(n, C_, **torch_args)
    print("Done in {:.2f} seonds".format(time.time() - ts))
    return C_torch.detach().cpu().numpy()
