"""Functions needed for the CKM matrix as well as for frequently used
combinations of CKM elements."""

from math import cos,sin
from cmath import exp, sqrt
import numpy as np

def ckm_standard(t12, t13, t23, delta):
    r"""CKM matrix in the standard parametrization and standard phase
    convention.

    Parameters
    ----------

    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta=\gamma$ in radians
    """
    c12 = cos(t12)
    c13 = cos(t13)
    c23 = cos(t23)
    s12 = sin(t12)
    s13 = sin(t13)
    s23 = sin(t23)
    return np.array([[c12*c13,
        c13*s12,
        s13/exp(1j*delta)],
        [-(c23*s12) - c12*exp(1j*delta)*s13*s23,
        c12*c23 - exp(1j*delta)*s12*s13*s23,
        c13*s23],
        [-(c12*c23*exp(1j*delta)*s13) + s12*s23,
        -(c23*exp(1j*delta)*s12*s13) - c12*s23,
        c13*c23]])

def tree_to_wolfenstein(Vus, Vub, Vcb, gamma):
    laC = Vus/sqrt(1-Vub**2)
    A = Vcb/sqrt(1-Vub**2)/laC**2
    rho = Vub*cos(gamma)/A/laC**3
    eta = Vub*sin(gamma)/A/laC**3
    rhobar = rho*(1 - laC**2/2.)
    etabar = eta*(1 - laC**2/2.)
    return laC, A, rhobar, etabar

def ckm_wolfenstein(laC, A, rhobar, etabar):
    r"""CKM matrix in the Wolfenstein parametrization and standard phase
    convention.

    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where $\rho = \bar\rho/(1-\lambda^2/2)$ and
    $\eta = \bar\eta/(1-\lambda^2/2)$.

    Parameters
    ----------

    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Wolfenstein parameter $\bar\rho = \rho(1-\lambda^2/2)$
    - `etabar`: Wolfenstein parameter $\bar\eta = \eta(1-\lambda^2/2)$
    """
    rho = rhobar/(1 - laC**2/2.)
    eta = etabar/(1 - laC**2/2.)
    return np.array([[sqrt(1 - laC**2)*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho)),
        laC*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho)),
        A*laC**3*((-1j)*eta + rho)],
        [-(laC*sqrt(1 - A**2*laC**4)) - A**2*laC**5*sqrt(1 - laC**2)*((1j)*eta + rho),
        sqrt(1 - laC**2)*sqrt(1 -  A**2*laC**4) - A**2*laC**6*((1j)*eta + rho),
        A*laC**2*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho))],
        [A*laC**3 - A*laC**3*sqrt(1 - laC**2)*sqrt(1 - A**2*laC**4)*((1j)*eta + rho),
        -(A*laC**2*sqrt(1 - laC**2)) - A*laC**4*sqrt(1 - A**2*laC**4)*((1j)*eta + rho),
        sqrt(1 - A**2*laC**4)*sqrt(1 - A**2*laC**6*((-1j)*eta + rho)*((1j)*eta + rho))]])

def ckm_tree(Vus, Vub, Vcb, gamma):
    r"""CKM matrix in the tree parametrization and standard phase
    convention.

    In this parametrization, the parameters are directly measured from
    tree-level $B$ decays. It is thus particularly suited for new physics
    analyses because the tree-level decays should be dominated by the Standard
    Model. This function involves no analytical approximations.

    Relation to the standard parametrization:

    - $V_{us} = \cos \theta_{13} \sin \theta_{12}$
    - $|V_{ub}| = |\sin \theta_{13}|$
    - $V_{cb} = \cos \theta_{13} \sin \theta_{23}$
    - $\gamma=\delta$

    Parameters
    ----------

    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: CKM phase $\gamma=\delta$ in radians
    """
    return np.array([[sqrt(1 - Vub**2)*sqrt(1 - Vus**2/(1 - Vub**2)),
        Vus,
        Vub/exp(1j*gamma)],
        [-((sqrt(1 - Vcb**2/(1 - Vub**2))*Vus)/sqrt(1 - Vub**2)) - (Vub*exp(1j*gamma)*Vcb*sqrt(1 - Vus**2/(1 - Vub**2)))/sqrt(1 - Vub**2),
        -((Vub*exp(1j*gamma)*Vcb*Vus)/(1 - Vub**2)) + sqrt(1 - Vcb**2/(1 - Vub**2))*sqrt(1 - Vus**2/(1 - Vub**2)),
        Vcb],
        [(Vcb*Vus)/(1 - Vub**2) - Vub*exp(1j*gamma)*sqrt(1 - Vcb**2/(1 - Vub**2))*sqrt(1 - Vus**2/(1 - Vub**2)),
        -((Vub*exp(1j*gamma)*sqrt(1 - Vcb**2/(1 - Vub**2))*Vus)/sqrt(1 - Vub**2)) - (Vcb*sqrt(1 - Vus**2/(1 - Vub**2)))/sqrt(1 - Vub**2),
        sqrt(1 - Vub**2)*sqrt(1 - Vcb**2/(1 - Vub**2))]])
