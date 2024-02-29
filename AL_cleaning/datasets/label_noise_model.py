#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np


def get_asym_noise_model(eta: float = 0.3) -> np.ndarray:
    """
    CLASS-DEPENDENT ASYMMETRIC LABEL NOISE
    https://proceedings.neurips.cc/paper/2018/file/f2925f97bc13ad2852a7a551802feea0-Paper.pdf
    TRUCK -> AUTOMOBILE, BIRD -> AIRPLANE, DEER -> HORSE, CAT -> DOG, and DOG -> CAT

    :param eta: The likelihood of true label switching from one of the specified classes to nearest class.
                In other words, likelihood of introducing a class-dependent label noise
    """

    # Generate a noise transition matrix.
    assert (0.0 <= eta) and (eta <= 1.0)

    eps = 1e-12
    num_classes = 10
    conf_mat = np.eye(N=num_classes)
    indices = [[2, 0], [9, 1], [5, 3], [3, 5], [4, 7]]
    for ind in indices:
        conf_mat[ind[0], ind[1]] = eta / (1.0 - eta + eps)
    return conf_mat / np.sum(conf_mat, axis=1, keepdims=True)


def get_sym_noise_model(eta: float = 0.3) -> np.ndarray:
    """
    Symmetric LABEL NOISE
    :param eta: The likelihood of true label switching from true class to rest of the classes.
    """
    # Generate a noise transition matrix.
    assert (0.0 <= eta) and (eta <= 1.0)
    assert isinstance(eta, float)

    num_classes = 10
    conf_mat = np.eye(N=num_classes)
    for ind in range(num_classes):
        conf_mat[ind, ind] -= eta
        other_classes = np.setdiff1d(range(num_classes), ind)
        for o_c in other_classes:
            conf_mat[ind, o_c] += eta / other_classes.size

    assert np.all(np.abs(np.sum(conf_mat, axis=1) - 1.0) < 1e-9)

    return conf_mat
