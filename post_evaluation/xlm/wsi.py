import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from joblib import Memory
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import AgglomerativeClustering
import sklearn.cluster.hierarchical
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


def clusterize_search( word, vecs, gold_sense_ids = None ,ncs=list(range(1, 5, 1)) + list(range(5, 12, 2)),
            affinities=('cosine',), linkages=('average',)):
    if linkages is None:
        linkages = sklearn.cluster.hierarchical._TREE_BUILDERS.keys()
    if affinities is None:
        affinities = ('cosine', 'euclidean', 'manhattan')
    sdfs = []
    mem = Memory('maxari_cache', verbose=0)

    zero_vecs = ((vecs ** 2).sum(axis=-1) == 0)
    if zero_vecs.sum() > 0:
        vecs = np.concatenate((vecs, zero_vecs[:, np.newaxis].astype(vecs.dtype)), axis=-1)

    best_clids = None
    best_silhouette = 0
    distances = []

    for affinity in affinities:
        distance_matrix = cdist(vecs, vecs, metric=affinity)
        distances.append(distance_matrix)
        for nc in ncs:
            for linkage in linkages:
                if linkage == 'ward' and affinity != 'euclidean':
                    continue
                clr = AgglomerativeClustering(affinity='precomputed', linkage=linkage, n_clusters=nc, memory=mem)
                clids = clr.fit_predict(distance_matrix) if nc > 1 else np.zeros(len(vecs))

                ari = ARI(gold_sense_ids, clids) if gold_sense_ids is not None else np.nan
                sil_cosine = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids,metric='cosine')
                sil_euclidean = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids, metric='euclidean')
                vc = '' if gold_sense_ids is None else '/'.join(
                                        np.sort(pd.value_counts(gold_sense_ids).values)[::-1].astype(str))
                if sil_cosine > best_silhouette:
                    best_silhouette = sil_cosine
                    best_clids = clids

                sdf = pd.DataFrame({'ari': ari,
                                    'word': word, 'nc': nc,
                                    'sil_cosine': sil_cosine,
                                    'sil_euclidean': sil_euclidean,
                                    'vc': vc,
                                    'affinity': affinity, 'linkage': linkage}, index=[0])

                sdfs.append(sdf)

    sdf = pd.concat(sdfs, ignore_index=True)
    return best_clids, sdf, None, distances
