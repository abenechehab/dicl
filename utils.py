import numpy as np
import matplotlib.pyplot as plt
import ot
from tqdm import tqdm

def neareast_integer_hash(dictionnaire, entier):
    cles = list(dictionnaire.keys())
    cles_triees = sorted(cles, key=lambda x: abs(x - entier))
    return cles_triees[0]

def one_prob_return(n,idx):
    res = np.zeros(n)
    res[idx] = 1.0
    return res

def generate_row_matrix(n):
    random_numbers = np.random.rand(n-1)
    random_numbers.sort()
    random_numbers = np.concatenate(([0], random_numbers, [1]))
    intervals = np.diff(random_numbers)
    return intervals

def safe_normalize_probabilities(probabilities):
    row_sums = np.sum(probabilities, axis=1, keepdims=True)
    non_zero_sums_mask = row_sums <= 1e-3
    normalized_probabilities = np.where(non_zero_sums_mask, np.true_divide(probabilities, row_sums), probabilities)
    return normalized_probabilities

def simple_normalize(prob):
    row_sums = np.sum(prob, axis=1, keepdims=True)
    non_zero_sums_mask = row_sums != 0.0
    normalized_probabilities = np.where(non_zero_sums_mask, np.true_divide(prob, row_sums), prob)
    return normalized_probabilities

def sum_printer(prob):
    res = []
    for p in prob:
        s = p.sum()
        print(s)
        res.append(s)
    return res

def bin2list(bine):
    width, height, center = bine
    sorted_data = sorted(zip(width, height, center), key=lambda x: x[2])
    histogram_values = []
    for w, h, c in sorted_data:
        repetitions = int(round((w / min(width))))
        for _ in range(repetitions):
            histogram_values.append(h)
    return histogram_values

# def bin2list_new(bine):
#     _, height, _ = bine
#     return height

def create_ns(full_series,comma_locations):
    ns = []
    old_c=0
    for i,c in enumerate(comma_locations):
        ns.append(int(full_series[old_c:c]))
        old_c=c+1
    return ns

def completion_matrix(bins, ns, loss):
    # bins = [(width, height, center)] * 1000
    P = np.zeros((1000,1000))
    values_and_losses = {}
    for idx, tok in enumerate(ns):
        if not tok in values_and_losses.keys():
            bin_list = bin2list(bins[idx])
            P[tok] = bin_list
            values_and_losses[tok] = loss[idx]
        else:
            if values_and_losses[tok] >= loss[idx]:
                bin_list = bin2list(bins[idx])
                P[tok] = bin_list
                values_and_losses[tok] = loss[idx]
    for idx in range(len(P[0])):
        if not idx in values_and_losses.keys() and 149 <= idx <= 849:
            P[idx][neareast_integer_hash(values_and_losses,idx)] = 1.0
    for idx in range(149,850):
        P[idx][:149], P[idx][850:] = [0.0] * 149, [0.0] * 150
    P = simple_normalize(P)
    P[:149], P[850:] = [one_prob_return(1000,150)] * 149, [one_prob_return(1000,849)] * 150

    return P, values_and_losses

def completion_matrix_ot(bins, ns, loss):
    # bins = [(width, height, center)] * 1000
    P = np.zeros((1000,1000))
    values_and_losses = {}
    for idx, tok in enumerate(ns):
        if not tok in values_and_losses.keys():
            bin_list = bin2list(bins[idx])
            P[tok] = bin_list
            values_and_losses[tok] = loss[idx]
        else:
            if values_and_losses[tok] >= loss[idx]:
                bin_list = bin2list(bins[idx])
                P[tok] = bin_list
                values_and_losses[tok] = loss[idx]
    interpolate_wass_barycenter(P, values_and_losses, 1e-5, 0.01)
    print(f"after ot: {np.sum(np.isnan(ps))}")
    for idx in range(149,850):
        P[idx][:149], P[idx][850:] = [0.0] * 149, [0.0] * 150
    print(f"after borders: {np.sum(np.isnan(ps))}")
    P = simple_normalize(P)
    print(f"after normalize: {np.sum(np.isnan(ps))}")
    P[:149], P[850:] = [one_prob_return(1000,150)] * 149, [one_prob_return(1000,849)] * 150
    return P, values_and_losses

def completion_matrix_ot_breg(bins, ns, loss, reg):
    # bins = [(width, height, center)] * 1000
    P = np.zeros((1000,1000))
    values_and_losses = {}
    for idx, tok in enumerate(ns):
        if not tok in values_and_losses.keys():
            bin_list = bin2list(bins[idx])
            P[tok] = bin_list
            values_and_losses[tok] = loss[idx]
        else:
            if values_and_losses[tok] >= loss[idx]:
                bin_list = bin2list(bins[idx])
                P[tok] = bin_list
                values_and_losses[tok] = loss[idx]
    print(f"after filling: {np.sum(np.isnan(P))}")
    for idx in range(149,850):
        P[idx][:149], P[idx][850:] = [0.0] * 149, [0.0] * 150
    print(f"fill borders: {np.sum(np.isnan(P))}")
    P = simple_normalize(P)
    print(f"after normalize: {np.sum(np.isnan(P))}")
    P[:149], P[850:] = [one_prob_return(1000,150)] * 149, [one_prob_return(1000,849)] * 150
    print(f"fill borders 2: {np.sum(np.isnan(P))}")
    interpolate_wass_barycenter_breg(P, values_and_losses, reg=reg)
    print(f"after ot: {np.sum(np.isnan(P))}")
    P = simple_normalize(P)
    print(f"after last normalize: {np.sum(np.isnan(P))}")
    return P, values_and_losses

def completion_matrix_ot_breg_from_matrix(P, reg):
    interpolate_wass_barycenter_breg(P, values_and_losses, reg=reg)
    P = simple_normalize(P)
    return P, values_and_losses


def interpolate_wass_barycenter(P, hashmap, reg, reg_m):
    sorted_keys = sorted(hashmap.keys())
    for i in tqdm(range(len(sorted_keys)-1)):
        n_weight = sorted_keys[i+1] - sorted_keys[i]
        if n_weight ==1: continue
        p1 = P[sorted_keys[i]]
        p2 = P[sorted_keys[i+1]]
        distribs = compute_wass_bar_unbalanced(p1, p2, n_weight, reg, reg_m)
        P[sorted_keys[i]: sorted_keys[i+1]] = distribs.T

def interpolate_wass_barycenter_breg(P, hashmap, reg):
    sorted_keys = sorted(hashmap.keys())
    for i in tqdm(range(len(sorted_keys)-1)):
        n_weight = sorted_keys[i+1] - sorted_keys[i]
        if n_weight ==1: continue
        p1 = P[sorted_keys[i]]
        p2 = P[sorted_keys[i+1]]
        distribs = compute_wass_bar_unbalanced_breg(p1, p2, n_weight, reg)
        P[sorted_keys[i]: sorted_keys[i+1]] = distribs.T

def compute_wass_bar_unbalanced(p1, p2, n_weight, reg, reg_m):

    B_wass = np.zeros((len(p1), n_weight))
    weight_list = np.linspace(0, 1, n_weight)

    A = np.vstack((p1, p2)).T

    M = ot.utils.dist0(len(p1))
    M /= M.max()
    for i in range(0, n_weight):
        weight = weight_list[i]
        weights = np.array([1 - weight, weight])
        B_wass[:, i] = ot.unbalanced.barycenter_unbalanced(A, M, reg, reg_m, weights=weights)
    return B_wass

def compute_wass_bar_unbalanced_breg(p1, p2, n_weight, reg):

    B_wass = np.zeros((len(p1), n_weight))
    weight_list = np.linspace(0, 1, n_weight)

    A = np.vstack((p1, p2)).T

    M = ot.utils.dist0(len(p1))
    M /= M.max()
    for i in range(0, n_weight):
        weight = weight_list[i]
        weights = np.array([1 - weight, weight])
        B_wass[:, i] = ot.bregman.barycenter_debiased(A, M, reg, weights=weights)
    return B_wass


def bins_completion(PDF_list):
    bins_=[]
    for i in range(len(PDF_list)):
        PDF_list[i].compute_stats()
        bins_.append((PDF_list[i].bin_width_arr,PDF_list[i].bin_height_arr,PDF_list[i].bin_center_arr))
    return bins_

def plot_matrix(hashmap,idx_last=1000,alpha=0.5):
    for idx in hashmap.keys():
        plt.hlines(idx, idx_last, 10,color='gray',alpha=alpha)

