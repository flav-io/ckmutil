"""Functions for the extraction of mixing angles and phases and for rephasing
general mixing matrices to standard parametrizations."""

import numpy as np
from math import asin, atan, sin, cos
from cmath import phase, exp

def mixing_phases(U):
    """Return the angles and CP phases of the CKM or PMNS matrix
    in standard parametrization, starting from a matrix with arbitrary phase
    convention."""
    f = {}
    # angles
    f['t13'] = asin(abs(U[0,2]))
    if U[0,0] == 0:
        f['t12'] = pi/2
    else:
        f['t12'] = atan(abs(U[0,1])/abs(U[0,0]))
    if U[2,2] == 0:
        f['t23'] = pi/2
    else:
        f['t23'] = atan(abs(U[1,2])/abs(U[2,2]))
    s12 = sin(f['t12'])
    c12 = cos(f['t12'])
    s13 = sin(f['t13'])
    c13 = cos(f['t13'])
    s23 = sin(f['t23'])
    c23 = cos(f['t23'])
    # standard phase
    if (s12*s23) == 0 or (c12*c13**2*c23*s13) == 0:
        f['delta'] = 0
    else:
        f['delta'] = -phase((U[0,0].conj()*U[0,2]*U[2,0]*U[2,2].conj()/(c12*c13**2*c23*s13) + c12*c23*s13)/(s12*s23))
    # Majorana phases
    f['delta1']  = phase(exp(1j*f['delta']) * U[0, 2])
    f['delta2']  = phase(U[1, 2])
    f['delta3']  = phase(U[2, 2])
    f['phi1'] = 2*phase(exp(1j*f['delta1']) * U[0, 0].conj())
    f['phi2'] = 2*phase(exp(1j*f['delta1']) * U[0, 1].conj())
    return f

def rephase_standard(UuL, UdL, UuR, UdR):
    """Function to rephase the quark rotation matrices in order to
    obtain the CKM matrix in standard parametrization.

    The input matrices are assumed to diagonalize the up-type and down-type
    quark matrices like

    ```
    UuL.conj().T @ Mu @ UuR = Mu_diag
    UdL.conj().T @ Md @ UdR = Md_diag
    ```

    The CKM matrix is given as `VCKM = UuL.conj().T @ UdL`.

    Returns a tuple with the rephased versions of the input matrices.
    """
    K = UuL.conj().T @ UdL
    f = mixing_phases(K)
    Fdelta = np.diag(np.exp([1j*f['delta1'], 1j*f['delta2'], 1j*f['delta3']]))
    Fphi = np.diag(np.exp([-1j*f['phi1']/2., -1j*f['phi2']/2., 0]))
    return UuL @ Fdelta, UdL @ Fphi.conj(), UuR @ Fdelta, UdR @ Fphi.conj()
