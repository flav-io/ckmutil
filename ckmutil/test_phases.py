import unittest
import numpy.testing as npt
from ckmutil.phases import *
from ckmutil.ckm import ckm_standard
from ckmutil.diag import msvd
import numpy as np
from math import sin, cos
from cmath import exp

# some auxiliary functions
def diagonal_phase_matrix(a1, a2, a3):
    r"""A diagonal $3\times 3$ matrix with $jj$-element $e^{i a_j}$"""
    return np.diag(np.exp(1j*np.array([a1, a2, a3])))
# general CKM/PMNS-like matrix
def unitary_matrix(t12, t13, t23, delta, delta1, delta2, delta3, phi1, phi2):
    ph1 = diagonal_phase_matrix(delta1, delta2, delta3)
    ph2 = diagonal_phase_matrix(-phi1/2, -phi2/2, 0)
    ckm = ckm_standard(t12, t13, t23, delta)
    return np.dot(ph1, np.dot(ckm, ph2))

class TestPhases(unittest.TestCase):
    def test_rephasing(self):
        # a random unitary matrix
        U = np.array([[-0.4825-0.5529j,  0.6076-0.0129j,  0.1177+0.2798j],
                   [ 0.1227-0.5748j, -0.1685-0.5786j, -0.2349-0.486j ],
                   [-0.0755+0.3322j,  0.5157+0.0393j, -0.7321-0.2837j]])
        f = mixing_phases(U)
        # check that matrix reconstructed from extracted angles and phases
        # coincides with original matrix
        npt.assert_array_almost_equal(U, unitary_matrix(**f),
            decimal=4)
        # pathological case: unit matrix. Check that everything is zero
        f = mixing_phases(np.eye(3))
        for k, v in f.items():
            self.assertEqual(v, 0, msg='{} is not zero'.format(k))
        # random mass matrices
        Mu = np.array([[ 0.6126+0.9819j,  0.0165+0.3709j,  0.0114+0.7819j],
            [ 0.6374+0.8631j,  0.1249+0.8346j,  0.6940+0.4495j],
            [ 0.1652+0.9667j,  0.4952+0.1281j,  0.7719+0.2325j]])
        Md = np.array([[ 0.2874+0.2065j,  0.2210+0.9077j,  0.4053+0.454j ],
            [ 0.0339+0.8332j,  0.5286+0.6339j,  0.3142+0.113j ],
            [ 0.9379+0.2952j,  0.8488+0.634j ,  0.1413+0.9175j]])
        UuL, Su, UuR = msvd(Mu)
        UdL, Sd, UdR = msvd(Md)
        f = mixing_phases(UuL.conj().T @ UdL)
        UuL_, UdL_, UuR_, UdR_ = rephase_standard(UuL, UdL, UuR, UdR)
        # rephased CKM
        K = UuL_.conj().T @ UdL_
        # CKM in standard parametrization
        K_std = unitary_matrix(f['t12'], f['t13'], f['t23'], f['delta'], 0, 0, 0, 0, 0)
        # ... should be equal!
        npt.assert_array_almost_equal(K, K_std, decimal=10)
