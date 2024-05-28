import numpy as np
import scipy
import torch


def calc_pairwise_symmetric_uncertainty_for_measure_function(means_of_all_ensembles: torch.Tensor,
                                                             vars_of_all_ensembles: torch.Tensor,
                                                             ensemble_size: int, measure_func):
    """
    @param means_of_all_ensembles: Tensor with size ensemble_size x batch_size x observation_dim
    @param vars_of_all_ensembles: Tensor with size ensemble_size x batch_size x observation_dim
    @param ensemble_size: Number of components of probabilistic ensemble
    @param measure_func: the measure function to calculate the pairwise model disagreement
    @return: pairwise symmetric(all pairs once and not both) uncertainty score of size batchsize
    """
    counter_u = 1
    sum_uncertainty = measure_func(means_of_all_ensembles[0],
                                   vars_of_all_ensembles[0],
                                   means_of_all_ensembles[1],
                                   vars_of_all_ensembles[1])
    for j in range(2, ensemble_size):
        for k in range(j):
            counter_u = counter_u + 1
            sum_uncertainty += measure_func(means_of_all_ensembles[j],
                                            vars_of_all_ensembles[j],
                                            means_of_all_ensembles[k],
                                            vars_of_all_ensembles[k])
    sum_uncertainty = sum_uncertainty / counter_u
    sum_uncertainty = sum_uncertainty.cpu().numpy()
    return sum_uncertainty

def calc_pairwise_not_symmetric_uncertainty_for_measure_function(means_of_all_ensembles: torch.Tensor,
                                                                 vars_of_all_ensembles: torch.Tensor,
                                                                 ensemble_size: int, measure_func):
    """
    @param means_of_all_ensembles: Tensor with size ensemble_size x batch_size x observation_dim
    @param vars_of_all_ensembles: Tensor with size ensemble_size x batch_size x observation_dim
    @param ensemble_size: Number of components of probabilistic ensemble
    @param measure_func: the measure function to calculate the pairwise model disagreement
    @return: pairwise symmetric(all pairs (i,j) and  (j,i)) uncertainty score of size batchsize
    """
    counter_u = 2
    sum_uncertainty = measure_func(means_of_all_ensembles[0],
                                   vars_of_all_ensembles[0],
                                   means_of_all_ensembles[1],
                                   vars_of_all_ensembles[1])
    sum_uncertainty += measure_func(means_of_all_ensembles[1],
                                    vars_of_all_ensembles[1],
                                    means_of_all_ensembles[0],
                                    vars_of_all_ensembles[0])
    for j in range(0, ensemble_size):
        for k in range(0, ensemble_size):
            if j == k or (j == 0 and k == 1) or (k == 1 and j == 0):
                continue
            counter_u = counter_u + 1
            sum_uncertainty += measure_func(means_of_all_ensembles[j],
                                            vars_of_all_ensembles[j],
                                            means_of_all_ensembles[k],
                                            vars_of_all_ensembles[k])
    sum_uncertainty = sum_uncertainty / counter_u
    sum_uncertainty = sum_uncertainty.cpu().numpy()
    return sum_uncertainty

def calc_uncertainty_score_genShen(means_first_gaussian: torch.Tensor, vars_first_gaussian: torch.Tensor,
                                   means_second_gaussian: torch.Tensor, vars_second_gaussian: torch.Tensor) -> torch.Tensor:
    """
    Geometric Jensen Shannon divergence
    @param means_first_gaussian: is [batch_size x observation_dim] Tensor with
        the means of first gaussian for sampling chosen Gaussian for each observation action pair
    @param vars_first_gaussian: is [batch_size x observation_dim] Tensor with
        the vars(diagonal) of first Gaussian for sampling chosen Gaussian for each observation action pair
    @param means_second_gaussian: is [batch_size x observation_dim] Tensor with
        the means of second Gaussian for sampling chosen Gaussian for each observation action pair
    @param vars_second_gaussian: is [batch_size x observation_dim] Tensor with
        the vars(diagonal) of second Gaussian for sampling chosen Gaussian for each observation action pair
    @return: 
    """
    al = 0.5
    t1 = (1 - al) * means_first_gaussian / vars_first_gaussian * means_first_gaussian
    t1S = torch.sum(t1, 1)
    t2 = al * means_second_gaussian / vars_second_gaussian * means_second_gaussian
    t2S = torch.sum(t2, 1)
    sigAL = 1 / ((1 - al) / vars_first_gaussian + al / vars_second_gaussian)
    muAl = sigAL * ((1 - al) / vars_first_gaussian * means_first_gaussian + al / vars_second_gaussian * means_second_gaussian)
    t3 = muAl / sigAL * muAl
    t3S = torch.sum(t3, 1)
    log_det_S1 = torch.sum(torch.log(vars_first_gaussian), 1)
    log_det_S2 = torch.sum(torch.log(vars_second_gaussian), 1)
    log_det_SSum = torch.sum(torch.log(sigAL), 1)
    log_term = (1 - al) * log_det_S1 + al * log_det_S2 - log_det_SSum
    shanon = 1 / 2 * (t1S + t2S - t3S + log_term)
    return shanon


def calc_uncertainty_scoreKL(means_first_gaussian: torch.Tensor, vars_first_gaussian: torch.Tensor,
                             means_second_gaussian: torch.Tensor, vars_second_gaussian: torch.Tensor) -> torch.Tensor:
    """This function calculates the Kullback Leiber Divergence between two Gaussians
    It is used the formula in docs/resources/formulas/kl-div.png
    P1 is N(means_first_gaussian[0], vars_first_gaussian[0]) and P2 is N(means_second_gaussian[0], vars_second_gaussian[0]) e.g
    for first batch element
    Args:
        means_first_gaussian(torch.Tensor):  is [batch_size x observation_dim] Tensor with
        the means of first gaussian  for each observation action pair
        vars_first_gaussian(torch.Tensor):  is [batch_size x observation_dim] Tensor with
        the vars(diagonal) of first Gaussian  for each observation action pair
        means_second_gaussian (torch.Tensor): [batch_size x observation_dim] sized Tensor with
        the means of second Gaussian  for each observation action pair
        vars_second_gaussian (torch.Tensor): is [batch_size x observation_dim] sized Tensor with
        the vars(diagonal) of second Gaussian  for each observation action pair
    Returns:
        torch.Tensor contains the uncertainty scores between all Gaussian pairs
    """

    # all tensors here are of shape[B, n], while B are the simultaneous model rollouts and
    # n is the size of next observation plus reward site of 1
    n = torch.ones(means_first_gaussian.size(dim=0)).fill_(means_first_gaussian.size(dim=1))
    n = n.to(means_first_gaussian.device)
    # n is of size B and contains only values n
    # S1 is refering to Sigma1 so chosen_stds[0] e.g, S2 is refering to Sigma2 so ensemble_rest_std[0] e.g
    log_det_S1 = torch.sum(torch.log(vars_first_gaussian), 1)
    log_det_S2 = torch.sum(torch.log(vars_second_gaussian), 1)
    log_term = log_det_S2 - log_det_S1
    tr_term = torch.sum(torch.div(vars_first_gaussian, vars_second_gaussian), 1)

    mu2_sub_mu1 = means_second_gaussian - means_first_gaussian

    last_big_term = torch.sum(mu2_sub_mu1 * mu2_sub_mu1 / vars_second_gaussian, 1)

    uncertainty_score = 1 / 2 * (log_term - n + tr_term + last_big_term)
    assert torch.sum(torch.isinf(uncertainty_score)) == 0
    assert torch.sum(torch.isnan(uncertainty_score)) == 0

    return uncertainty_score

def calc_uncertainty_score_hellinger(means_first_gaussian: torch.Tensor, vars_first_gaussian: torch.Tensor,
                                     means_second_gaussian: torch.Tensor, vars_second_gaussian: torch.Tensor) -> torch.Tensor:
    """This function calculates the Hellinger distance between two gaussians but for whole batch
    Args:
        means_first_gaussian(torch.Tensor):  is [batch_size x observation_dim] Tensor with
        the means of first gaussian  for each observation action pair
        vars_first_gaussian(torch.Tensor):  is [batch_size x observation_dim] Tensor with
        the vars(diagonal) of first Gaussian  for each observation action pair
        means_second_gaussian (torch.Tensor): [batch_size x observation_dim] sized Tensor with
        the means of second Gaussian  for each observation action pair
        vars_second_gaussian (torch.Tensor): is [batch_size x observation_dim] sized Tensor with
        the vars(diagonal) of second Gaussian  for each observation action pair
    Returns:
        torch.Tensor contains the uncertainty scores between all Gaussian pairs
    """
    a = torch.sum(torch.log(vars_first_gaussian), axis=1)
    b = torch.sum(torch.log(vars_second_gaussian), axis=1)
    c = torch.sum(torch.log((vars_first_gaussian + vars_second_gaussian) / 2), axis=1)
    x = torch.exp(0.25 * a + 0.25 * b - 0.5 * c)
    next = (-1 / 8) * torch.sum(
        (means_first_gaussian - means_second_gaussian) / ((vars_first_gaussian + vars_second_gaussian) / 2) * (means_first_gaussian - means_second_gaussian),
        axis=1)
    final = 1 - (x * torch.exp(next))
    return torch.sqrt(final)

