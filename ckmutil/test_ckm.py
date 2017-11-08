import unittest
from math import radians,asin,degrees
import cmath
from ckmutil.ckm import *
import numpy as np

# some values close to the real ones
Vus = 0.22
Vub = 3.5e-3
Vcb = 4.0e-2
gamma = radians(70.) # 70° in radians

# converting to other parametrizations
t12 = asin(Vus)
t13 = asin(Vub)
t23 = asin(Vcb)/cos(t13)
delta = gamma
laC = Vus
A = sin(t23)/laC**2
rho_minus_i_eta = sin(t13) * cmath.exp(-1j*delta) / (A*laC**3)
rho = rho_minus_i_eta.real
eta = -rho_minus_i_eta.imag
rhobar = rho*(1 - laC**2/2.)
etabar = eta*(1 - laC**2/2.)

class TestCKM(unittest.TestCase):
    v_s = ckm_standard(t12, t13, t23, delta)
    v_w = ckm_wolfenstein(laC, A, rhobar, etabar)
    v_t = ckm_tree(Vus, Vub, Vcb, gamma)
    v_wt = ckm_wolfenstein(*tree_to_wolfenstein(Vus, Vub, Vcb, gamma))
    par_s = dict(t12=t12,t13=t13,t23=t23,delta=delta)
    par_w = dict(laC=laC,A=A,rhobar=rhobar,etabar=etabar)
    par_t = dict(Vus=Vus,Vub=Vub,Vcb=Vcb,gamma=gamma)

    def test_ckm_parametrizations(self):
        np.testing.assert_almost_equal(self.v_t/self.v_s, np.ones((3,3)), decimal=5)
        np.testing.assert_almost_equal(self.v_t/self.v_w, np.ones((3,3)), decimal=5)
        np.testing.assert_almost_equal(self.v_t/self.v_wt, np.ones((3,3)), decimal=5)

    def test_ckm_unitarity(self):
        np.testing.assert_almost_equal(np.dot(self.v_t,self.v_t.conj().T), np.eye(3), decimal=15)
        np.testing.assert_almost_equal(np.dot(self.v_w,self.v_w.conj().T), np.eye(3), decimal=15)
        np.testing.assert_almost_equal(np.dot(self.v_s,self.v_s.conj().T), np.eye(3), decimal=15)
