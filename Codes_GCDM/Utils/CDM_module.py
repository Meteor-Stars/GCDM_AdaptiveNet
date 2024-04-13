import time

import torch
import torch.nn.functional as F
import numpy as np

def CDM_Ori_Fusion(alpha,class_number,args=None):
    """
    This is original evidence fusion method proposed by paper "TRUSTED MULTI-VIEW CLASSIFICATION"
    :param alpha:
    :param class_number
    :param args:
    :return:
    """
    def DS_Combine_two(alpha1, alpha2):
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()

        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = class_number / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, class_number, 1), b[1].view(-1, 1, class_number))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)

        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag
        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))
        # calculate new S
        S_a = class_number / u_a #
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a
    alpha_a = alpha[0]
    for v in range(len(alpha) - 1):
        if v == 0:
            alpha_a = DS_Combine_two(alpha[0], alpha[1])
        else:
            alpha_a = DS_Combine_two(alpha_a, alpha[v + 1])
    return alpha_a

def CDM_Our_Fusion(alpha,class_number,args=None):
    """
    This is an improved evidence fusion method by introducing an effective balance term
    to slow down the changing trend of fusion values u_k^* and b_k^*

    :param alpha:
    :param class_number
    :param args:
    :return:
    """
    def DS_Combine_two(alpha1, alpha2):
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()

        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = class_number / S[v]

        # b^0 @ b^(0+1)
        bb=(b[0].view(-1, class_number, 1)+b[1].view(-1, 1, class_number))/2
        bb2 = torch.bmm(b[0].view(-1, class_number, 1), b[1].view(-1, 1, class_number))
        bb=(bb+bb2)/2
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag
        # calculate b^a
        b_a = ((b[0]*0.5+b[1]*0.5+b[0]*b[1])/2 + (bu*0.5+ub*0.5+bu*ub)/2) / ((1 - C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a =(u[0]*0.5+u[1]*0.5+u[0]*u[1])/2 / ((1 - C).view(-1, 1).expand(u[0].shape))
        # calculate new S
        S_a = class_number / u_a #
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    alpha_a = alpha[0]
    for v in range(len(alpha) - 1):
        if v == 0:
            alpha_a = DS_Combine_two(alpha[0], alpha[1])
        else:
            alpha_a = DS_Combine_two(alpha_a, alpha[v + 1])
    return alpha_a