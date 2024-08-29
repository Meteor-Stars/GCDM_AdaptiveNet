import time

import torch
import torch.nn.functional as F
import numpy as np

def Uncertainty_aware_Fusion( alpha,classes,balance_term=False):
    """
    Uncertainty-aware Fusion (UF) to realize Collaborative Decision Making (CDM).

    """
    def Sub_UF(alpha1, alpha2):
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = classes / S[v]
        bu0 = torch.mul(b[0], 1-u[0])
        bu1 = torch.mul(b[1], 1 - u[1])

        if balance_term:
            b_f = ((b[0] * 0.5 + b[1] * 0.5 + b[0] * b[1]) / 2 + bu0 + bu1)
            u_f = u[0] + u[1] + u[0] * u[1]
        else:
            #Fusion without balance term
            b_f = b[0] * b[1] + bu0 + bu1
            u_f = u[0] * u[1]

        S_f = classes / u_f
        e_f = torch.mul(b_f, S_f.expand(b_f.shape))
        alpha_f = e_f + 1
        return alpha_f
    alpha_fuse = alpha[0]
    for v in range(len(alpha) - 1):
        if v == 0:
            alpha_fuse = Sub_UF(alpha[0], alpha[1])
        else:
            alpha_fuse = Sub_UF(alpha_fuse, alpha[v + 1])
    return alpha_fuse