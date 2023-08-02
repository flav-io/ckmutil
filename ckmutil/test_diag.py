import unittest
import numpy.testing as npt
from ckmutil.diag import msvd, mtakfac
import numpy as np

# complex
Mc = np.array([[ 0.82469+0.62495j,  0.37768+0.28744j,  0.40011+0.93478j],
       [ 0.85475+0.60855j,  0.64045+0.93049j,  0.39019+0.6188j ],
       [ 0.68798+0.40478j,  0.38995+0.86032j,  0.20555+0.601j  ]])
# complex symmetric
Mcs = np.array([[ 0.82794+0.4311j ,  0.33124+0.73104j,  0.77668+0.14696j],
       [ 0.33124+0.73104j,  0.18407+0.67383j,  0.46557+0.14199j],
       [ 0.77668+0.14696j,  0.46557+0.14199j,  0.94771+0.73933j]])
# complex symmetric with two degenerate singular values
Mcsd = np.array([[ 0.20019824+0.21595061j,  0.0563654 -0.14476247j,
         0.09818666-0.1975587j ],
       [ 0.0563654 -0.14476247j,  0.2457214 -0.00074698j,
        -0.04078761+0.19632891j],
       [ 0.09818666-0.1975587j , -0.04078761+0.19632891j,
         0.19381973+0.13898357j]])

class TestDiag(unittest.TestCase):
    def test_svd(self):
        U, S, V = msvd(Mc)
        npt.assert_array_almost_equal(U @ np.diag(S) @ V.conj().T, Mc, decimal=12)
        npt.assert_array_equal(S.imag, np.array([0, 0, 0]))
        self.assertTrue(S[1] >= S[0])
        self.assertTrue(S[2] >= S[1])

    def test_takfac(self):
        npt.assert_array_equal(Mcs - Mcs.T, np.zeros((3,3)))
        U, S = mtakfac(Mcs)
        npt.assert_array_almost_equal(U.conj() @ np.diag(S) @ U.conj().T, Mcs, decimal=6)

    def test_takfac_degenerate(self):
        npt.assert_array_equal(Mcsd - Mcsd.T, np.zeros((3,3)))
        U, S = mtakfac(Mcsd)
        npt.assert_array_almost_equal(U.conj() @ np.diag(S) @ U.conj().T, Mcsd, decimal=6)
