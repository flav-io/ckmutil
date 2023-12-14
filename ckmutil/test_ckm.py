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
s13 = Vub
c13 = np.sqrt(1-s13**2)
s12 = Vus/c13
c12 = np.sqrt(1-s12**2)
s23 = Vcb/c13
c23 = np.sqrt(1-s23**2)
t12 = asin(s12)
t13 = asin(s13)
t23 = asin(s23)
delta = gamma_to_delta(t12, t13, t23, gamma, delta_expansion_order=None)
laC = Vus
A = s23/laC**2
rho_minus_i_eta = s13 * cmath.exp(-1j*delta) / (A*laC**3)
rho = rho_minus_i_eta.real
eta = -rho_minus_i_eta.imag
rhobar_plus_i_etabar = sqrt(1-laC**2)*(rho + 1j*eta)/(sqrt(1-A**2*laC**4)+sqrt(1-laC**2)*A**2*laC**4*(rho + 1j*eta)) # e.q. Eq. (92) in arXiv:2206.07501
rhobar = rhobar_plus_i_etabar.real
etabar = rhobar_plus_i_etabar.imag
Vcd_complex = - s12*c23 - c12*s23*s13 * np.exp(1j*delta)
Vtd_complex = s12*s23 - c12*c23*s13 * np.exp(1j*delta)
beta = np.angle(-Vcd_complex/Vtd_complex)

class TestCKM(unittest.TestCase):
    v_s = ckm_standard(t12, t13, t23, delta)
    v_w = ckm_wolfenstein(laC, A, rhobar, etabar)
    v_t = ckm_tree(Vus, Vub, Vcb, gamma, delta_expansion_order=None)
    v_wt = ckm_wolfenstein(*tree_to_wolfenstein(Vus, Vub, Vcb, gamma, delta_expansion_order=None))
    v_b = ckm_beta_gamma(Vus, Vcb, beta, gamma, delta_expansion_order=None)
    par_s = dict(t12=t12,t13=t13,t23=t23,delta=delta)
    par_w = dict(laC=laC,A=A,rhobar=rhobar,etabar=etabar)
    par_t = dict(Vus=Vus,Vub=Vub,Vcb=Vcb,gamma=gamma)

    def test_ckm_parametrizations(self):
        np.testing.assert_almost_equal(self.v_t/self.v_s, np.ones((3,3)), decimal=5)
        np.testing.assert_almost_equal(self.v_t/self.v_w, np.ones((3,3)), decimal=5)
        np.testing.assert_almost_equal(self.v_t/self.v_wt, np.ones((3,3)), decimal=5)
        np.testing.assert_almost_equal(self.v_t/self.v_b, np.ones((3,3)), decimal=5)

    def test_ckm_unitarity(self):
        np.testing.assert_almost_equal(np.dot(self.v_t,self.v_t.conj().T), np.eye(3), decimal=15)
        np.testing.assert_almost_equal(np.dot(self.v_w,self.v_w.conj().T), np.eye(3), decimal=15)
        np.testing.assert_almost_equal(np.dot(self.v_s,self.v_s.conj().T), np.eye(3), decimal=15)

    def test_translations_and_inverse(self):
        np.testing.assert_almost_equal(tuple(self.par_s.values()), tree_to_standard(*standard_to_tree(**self.par_s), delta_expansion_order=None),decimal=15)
        np.testing.assert_almost_equal(tuple(self.par_w.values()), tree_to_wolfenstein(*wolfenstein_to_tree(**self.par_w), delta_expansion_order=None),decimal=15)
        np.testing.assert_almost_equal(tuple(self.par_w.values()), standard_to_wolfenstein(*wolfenstein_to_standard(**self.par_w)),decimal=15)
        np.testing.assert_almost_equal(tuple(self.par_s.values()), beta_gamma_to_standard(*standard_to_beta_gamma(**self.par_s), delta_expansion_order=None),decimal=12)

    def test_gamma_to_delta(self):
        np.testing.assert_almost_equal(gamma_to_delta(t12, t13, t23, gamma, delta_expansion_order=0), delta, decimal=3)
        np.testing.assert_almost_equal(gamma_to_delta(t12, t13, t23, gamma, delta_expansion_order=1), delta, decimal=10)
        np.testing.assert_almost_equal(gamma_to_delta(t12, t13, t23, gamma, delta_expansion_order=2), delta, decimal=15)
        np.testing.assert_equal(gamma_to_delta(t12, t13, t23, gamma, delta_expansion_order=None), delta)

    def test_beta_gamma_to_delta(self):
        np.testing.assert_almost_equal(beta_gamma_to_delta(beta, gamma, t23, delta_expansion_order=0), delta, decimal=3)
        np.testing.assert_almost_equal(beta_gamma_to_delta(beta, gamma, t23, delta_expansion_order=1), delta, decimal=7)
        np.testing.assert_almost_equal(beta_gamma_to_delta(beta, gamma, t23, delta_expansion_order=2), delta, decimal=10)
        np.testing.assert_equal(beta_gamma_to_delta(beta, gamma, t23, delta_expansion_order=None), delta)
