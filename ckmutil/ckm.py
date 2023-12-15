"""Functions needed for the CKM quark mixing matrix."""

from numpy import exp, sqrt, angle as phase, cos, sin, tan, arctan, arcsin
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

    Returns
    -------
    - `v`: CKM matrix in the standard parametrization and standard phase
        convention
    """
    c12 = cos(t12)
    c13 = cos(t13)
    c23 = cos(t23)
    s12 = sin(t12)
    s13 = sin(t13)
    s23 = sin(t23)
    v = np.array([[c12*c13,
        c13*s12,
        s13/exp(1j*delta)],
        [-(c23*s12) - c12*exp(1j*delta)*s13*s23,
        c12*c23 - exp(1j*delta)*s12*s13*s23,
        c13*s23],
        [-(c12*c23*exp(1j*delta)*s13) + s12*s23,
        -(c23*exp(1j*delta)*s12*s13) - c12*s23,
        c13*c23]])
    if len(v.shape) > 2:
        v = np.moveaxis(v, [0,1],[-2,-1])
    return v

def gamma_to_delta(t12, t13, t23, gamma, delta_expansion_order=None):
    r"""CKM phase $\delta$ in terms of $\gamma$.

    By default, no analytical approximation is made for the CKM phase
    $\delta$ but the optional argument `delta_expansion_order` allows to use
    the very accurate analytical approximation $\delta=\gamma$ for
    `delta_expansion_order=0` or to include higher-order corrections to this
    approximation for `delta_expansion_order=1` or `delta_expansion_order=2`.

    Parameters
    ----------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-11})$.

    Returns
    -------
    - `delta`: CKM phase $\delta$ in radians
    """
    if delta_expansion_order == 0:
        delta = gamma
    else:
        s13 = sin(t13)
        tan12 = tan(t12)
        tan23 = tan(t23)
        k = s13 * tan23 / tan12
        if delta_expansion_order == 1:
            delta = gamma + k * sin(gamma)
        elif delta_expansion_order == 2:
            delta = gamma + k * sin(gamma) + 1/6 * k**3 * sin(gamma)**3
        elif delta_expansion_order is None:
            delta = arctan((1 - k**2)/(1/tan(gamma) - k * sqrt(1/sin(gamma)**2 - k**2)))
        else:
            raise ValueError('delta_expansion_order must be 0, 1, 2, or None.')
    return delta.real

def beta_gamma_to_delta(beta, gamma, t23, delta_expansion_order=None):
    r"""CKM phase $\delta$ in terms of $\beta$ and $\gamma$.

    By default, no analytical approximation is made for the CKM phase
    $\delta$ but the optional argument `delta_expansion_order` allows to use
    the very accurate analytical approximation $\delta=\gamma$ for
    `delta_expansion_order=0` or to include higher-order corrections to this
    approximation for `delta_expansion_order=1` or `delta_expansion_order=2`.

    Parameters
    ----------
    - `beta`: Unitarity Triangle angle $\beta$ in radians
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-6})$.

    Returns
    -------
    - `delta`: CKM phase $\delta$ in radians
    """
    if delta_expansion_order == 0:
        delta = gamma
    else:
        s23 = sin(t23)
        Rb = sin(beta) / sin(beta + gamma)
        rhobar = Rb * cos(gamma)
        etabar = Rb * sin(gamma)
        if delta_expansion_order == 1:
            delta = gamma + s23**2 * etabar
        elif delta_expansion_order == 2:
            delta = gamma + s23**2 * etabar + s23**4 * rhobar * etabar
        elif delta_expansion_order is None:
            delta = arctan(1/(1/tan(gamma) - s23**2 * Rb**2 / etabar ))
        else:
            raise ValueError('delta_expansion_order must be 0, 1, 2, or None.')
    return delta.real

def tree_to_standard(Vus, Vub, Vcb, gamma, delta_expansion_order=None):
    r"""Function to convert from the CKM matrix in the tree parametrization to
    the CKM matrix in the standard parametrization.

    Parameters
    ----------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-11})$.

    Returns
    -------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians
    """
    s13 = Vub
    c13 = sqrt(1 - s13**2)
    s12 = Vus/c13
    s23 = Vcb/c13
    t13 = arcsin(s13)
    t12 = arcsin(s12)
    t23 = arcsin(s23)
    delta = gamma_to_delta(t12, t13, t23, gamma, delta_expansion_order)
    return t12.real, t13.real, t23.real, delta

def standard_to_tree(t12, t13, t23, delta):
    r"""Function to convert from the CKM matrix in the standard parametrization
    to the CKM matrix in the tree parametrization.

    Parameters
    ----------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians

    Returns
    -------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians
    """
    s12 = sin(t12)
    s13 = sin(t13)
    s23 = sin(t23)
    c12 = cos(t12)
    c13 = cos(t13)
    c23 = cos(t23)
    Vus = s12 * c13
    Vub = s13
    Vcb = s23 * c13
    Vcd_complex = - s12*c23 - c12*s23*s13 * exp(1j*delta)
    gamma = phase(-exp(1j*delta)/Vcd_complex)
    return Vus.real, Vub.real, Vcb.real, gamma

def beta_gamma_to_standard(Vus, Vcb, beta, gamma, delta_expansion_order=None):
    r"""Function to convert from the CKM matrix in the beta-gamma parametrization
    to the CKM matrix in the standard parametrization.

    Parameters
    ----------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `beta`: Unitarity Triangle angle $\beta$ in radians
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-6})$.

    Returns
    -------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians
    """
    Rb = sin(beta) / sin(beta + gamma)
    rhobar = Rb * cos(gamma)
    a = Vcb**2 * Vus**2 * (1 - Vcb**2) * Rb**2
    b = 1 - Vus**2 - Vcb**2 * (2 * rhobar * (1 - Vus**2) - Vcb**2 * Rb**2)
    c = 2 - Vus**2 - 2 * Vcb**2 * rhobar
    p = (3*b + c**2)/9
    q = (27*a + 9*b*c + 2*c**3)/54
    t = 2*sqrt(p)*sin(arcsin(p**(-3/2)*q)/3)
    s13 = sqrt(t - c/3)
    c13 = sqrt(1 - s13**2)
    s12 = Vus/c13
    s23 = Vcb/c13
    t13 = arcsin(s13)
    t12 = arcsin(s12)
    t23 = arcsin(s23)
    delta = beta_gamma_to_delta(beta, gamma, t23, delta_expansion_order)
    return t12.real, t13.real, t23.real, delta

def standard_to_beta_gamma(t12, t13, t23, delta):
    r"""Function to convert from the CKM matrix in the standard parametrization
    to the CKM matrix in the beta-gamma parametrization.

    Parameters
    ----------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians

    Returns
    -------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `beta`: Unitarity Triangle angle $\beta$ in radians
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians
    """
    s12 = sin(t12)
    s13 = sin(t13)
    s23 = sin(t23)
    c12 = cos(t12)
    c13 = cos(t13)
    c23 = cos(t23)
    Vus = s12 * c13
    Vcb = s23 * c13
    Vcd_complex = - s12*c23 - c12*s23*s13 * exp(1j*delta)
    Vtd_complex = s12*s23 - c12*c23*s13 * exp(1j*delta)
    beta = phase(-Vcd_complex/Vtd_complex)
    gamma = phase(-exp(1j*delta)/Vcd_complex)
    return Vus.real, Vcb.real, beta, gamma

def wolfenstein_to_standard(laC, A, rhobar, etabar):
    r"""Function to convert from the CKM matrix in the Wolfenstein parametrization
    to the CKM matrix in the standard parametrization.

    Parameters
    ----------
    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Real part of the apex of the Unitarity Triangle
    - `etabar`: Imaginary part of the apex of the Unitarity Triangle

    Returns
    -------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians

    Notes
    -----
    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where the Wolfenstein parameters $\rho$ and $\eta$ are related to the real
    and imaginary parts of the apex of the Unitarity Triangle $\bar\rho$ and
    $\bar\eta$ by

    - $\rho + i \eta = \frac{\sqrt{1-A^2\lambda^4}(\bar\rho + i \bar\eta)}{\sqrt{1-\lambda^2}(1-A^2\lambda^4(\bar\rho + i \bar\eta))}$

    which can be approximated by (but this approximation is not used in this
    function)

    - $\rho \approx \bar\rho/(1-\lambda^2/2)$
    - $\eta \approx \bar\eta/(1-\lambda^2/2)$
    """
    rho_plus_i_eta = sqrt(1-A**2*laC**4)*(rhobar + 1j*etabar)/(sqrt(1-laC**2)*(1 - A**2*laC**4*(rhobar + 1j*etabar))) # e.g. Eq. (93) in arXiv:2206.07501
    s12 = laC
    s23 = A*laC**2
    s13 = A*laC**3*np.abs(rho_plus_i_eta)
    delta = phase(rho_plus_i_eta)
    t12 = arcsin(s12)
    t13 = arcsin(s13)
    t23 = arcsin(s23)
    return t12.real, t13.real, t23.real, delta

def standard_to_wolfenstein(t12, t13, t23, delta):
    r"""Function to convert from the CKM matrix in the standard parametrization
    to the CKM matrix in the Wolfenstein parametrization.

    Parameters
    ----------
    - `t12`: CKM angle $\theta_{12}$ in radians
    - `t13`: CKM angle $\theta_{13}$ in radians
    - `t23`: CKM angle $\theta_{23}$ in radians
    - `delta`: CKM phase $\delta$ in radians

    Returns
    -------
    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Real part of the apex of the Unitarity Triangle
    - `etabar`: Imaginary part of the apex of the Unitarity Triangle

    Notes
    -----
    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where the Wolfenstein parameters $\rho$ and $\eta$ are related to the real
    and imaginary parts of the apex of the Unitarity Triangle $\bar\rho$ and
    $\bar\eta$ by

    - $\rho + i \eta = \frac{\sqrt{1-A^2\lambda^4}(\bar\rho + i \bar\eta)}{\sqrt{1-\lambda^2}(1-A^2\lambda^4(\bar\rho + i \bar\eta))}$

    which can be approximated by (but this approximation is not used in this
    function)

    - $\rho \approx \bar\rho/(1-\lambda^2/2)$
    - $\eta \approx \bar\eta/(1-\lambda^2/2)$
    """
    laC = sin(t12)
    A = sin(t23)/laC**2
    rho_plus_i_eta = sin(t13) * exp(1j*delta) / (A*laC**3)
    rhobar_plus_i_etabar = sqrt(1-laC**2)*rho_plus_i_eta/(sqrt(1-A**2*laC**4)+sqrt(1-laC**2)*A**2*laC**4*rho_plus_i_eta) # e.g. Eq. (92) in arXiv:2206.07501
    rhobar = rhobar_plus_i_etabar.real
    etabar = rhobar_plus_i_etabar.imag
    return laC.real, A.real, rhobar, etabar

def tree_to_wolfenstein(Vus, Vub, Vcb, gamma, delta_expansion_order=None):
    r"""Function to convert from the CKM matrix in the tree parametrization to
    the CKM matrix in the Wolfenstein parametrization.

    Parameters
    ----------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-11})$.

    Returns
    -------
    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Real part of the apex of the Unitarity Triangle
    - `etabar`: Imaginary part of the apex of the Unitarity Triangle

    Notes
    -----
    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where the Wolfenstein parameters $\rho$ and $\eta$ are related to the real
    and imaginary parts of the apex of the Unitarity Triangle $\bar\rho$ and
    $\bar\eta$ by

    - $\rho + i \eta = \frac{\sqrt{1-A^2\lambda^4}(\bar\rho + i \bar\eta)}{\sqrt{1-\lambda^2}(1-A^2\lambda^4(\bar\rho + i \bar\eta))}$

    which can be approximated by (but this approximation is not used in this
    function)

    - $\rho \approx \bar\rho/(1-\lambda^2/2)$
    - $\eta \approx \bar\eta/(1-\lambda^2/2)$
    """
    t12, t13, t23, delta = tree_to_standard(Vus, Vub, Vcb, gamma, delta_expansion_order)
    return standard_to_wolfenstein(t12, t13, t23, delta)

def wolfenstein_to_tree(laC, A, rhobar, etabar):
    r"""Function to convert from the CKM matrix in the Wolfenstein parametrization
    to the CKM matrix in the tree parametrization.

    Parameters
    ----------
    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Real part of the apex of the Unitarity Triangle
    - `etabar`: Imaginary part of the apex of the Unitarity Triangle

    Returns
    -------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Notes
    -----
    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where the Wolfenstein parameters $\rho$ and $\eta$ are related to the real
    and imaginary parts of the apex of the Unitarity Triangle by

    - $\rho + i \eta = \frac{\sqrt{1-A^2\lambda^4}(\bar\rho + i \bar\eta)}{\sqrt{1-\lambda^2}(1-A^2\lambda^4(\bar\rho + i \bar\eta))}$

    which can be approximated by (but this approximation is not used in this
    function)

    - $\rho \approx \bar\rho/(1-\lambda^2/2)$
    - $\eta \approx \bar\eta/(1-\lambda^2/2)$
    """
    t12, t13, t23, delta = wolfenstein_to_standard(laC, A, rhobar, etabar)
    return standard_to_tree(t12, t13, t23, delta)

def ckm_wolfenstein(laC, A, rhobar, etabar):
    r"""CKM matrix in the Wolfenstein parametrization and standard phase
    convention.

    This function does not rely on an expansion in the Cabibbo angle but
    defines, to all orders in $\lambda$,

    - $\lambda = \sin\theta_{12}$
    - $A\lambda^2 = \sin\theta_{23}$
    - $A\lambda^3(\rho-i \eta) = \sin\theta_{13}e^{-i\delta}$

    where the Wolfenstein parameters $\rho$ and $\eta$ are related to the real
    and imaginary parts of the apex of the Unitarity Triangle $\bar\rho$ and
    $\bar\eta$ by

    - $\rho + i \eta = \frac{\sqrt{1-A^2\lambda^4}(\bar\rho + i \bar\eta)}{\sqrt{1-\lambda^2}(1-A^2\lambda^4(\bar\rho + i \bar\eta))}$

    which can be approximated by (but this approximation is not used in this
    function)

    - $\rho \approx \bar\rho/(1-\lambda^2/2)$
    - $\eta \approx \bar\eta/(1-\lambda^2/2)$

    Parameters
    ----------

    - `laC`: Wolfenstein parameter $\lambda$ (sine of Cabibbo angle)
    - `A`: Wolfenstein parameter $A$
    - `rhobar`: Real part of the apex of the Unitarity Triangle
    - `etabar`: Imaginary part of the apex of the Unitarity Triangle

    Returns
    -------
    - `v`: CKM matrix in the Wolfenstein parametrization and standard phase
        convention
    """
    t12, t13, t23, delta = wolfenstein_to_standard(laC, A, rhobar, etabar)
    return ckm_standard(t12, t13, t23, delta)

def ckm_tree(Vus, Vub, Vcb, gamma, delta_expansion_order=None):
    r"""CKM matrix in the tree parametrization and standard phase
    convention.

    In this parametrization, the parameters are directly measured from
    tree-level $B$ decays. It is thus particularly suited for new physics
    analyses because the tree-level decays should be dominated by the Standard
    Model. By default, no analytical approximation is made for the CKM phase
    $\delta$ but the optional argument `delta_expansion_order` allows to use
    the very accurate analytical approximation $\delta=\gamma$ for
    `delta_expansion_order=0` or to include higher-order corrections to this
    approximation for `delta_expansion_order=1` or `delta_expansion_order=2`.

    Relation to the standard parametrization:

    - $V_{us} = \cos \theta_{13} \sin \theta_{12}$
    - $|V_{ub}| = \sin \theta_{13}$
    - $V_{cb} = \cos \theta_{13} \sin \theta_{23}$
    - $\delta = \gamma + \mathcal{O}(10^{-3})$

    Parameters
    ----------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vub`: Absolute value of CKM matrix element $|V_{ub}|$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `gamma`: CKM phase $\gamma=\delta$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-11})$.

    Returns
    -------
    - `v`: CKM matrix in the tree parametrization and standard phase
    """
    t12, t13, t23, delta = tree_to_standard(Vus, Vub, Vcb, gamma, delta_expansion_order)
    return ckm_standard(t12, t13, t23, delta)

def ckm_beta_gamma(Vus, Vcb, beta, gamma, delta_expansion_order=None):
    r"""CKM matrix in the beta-gamma parametrization and standard phase
    convention.

    In this parametrization, the two angles $\beta$ and $\gamma$ of the
    Unitarity Triangle are used as input parameters in addition to the CKM
    matrix elements $V_{us}$ and $V_{cb}$. By default, no analytical
    approximation is made for the CKM phase $\delta$ but the optional argument
    `delta_expansion_order` allows to use the very accurate analytical
    approximation $\delta=\gamma$ for `delta_expansion_order=0` or to include
    higher-order corrections to this approximation for
    `delta_expansion_order=1` or `delta_expansion_order=2`.

    Parameters
    ----------
    - `Vus`: CKM matrix element $V_{us}$
    - `Vcb`: CKM matrix element $V_{cb}$
    - `beta`: Unitarity Triangle angle $\beta$ in radians
    - `gamma`: Unitarity Triangle angle $\gamma$ in radians

    Optional parameters
    -------------------
    - `delta_expansion_order` (optional): `None` (default), `0`, `1`, or `2`.
        If `None` (default), the exact relation between the CKM phase $\delta$
        and the Unitarity Triangle angle $\gamma$ is used. If `0`, $\delta$ is
        set to the value of `gamma`. If `1`, $\delta$ is set to the value of
        `gamma` plus the leading permille correction $\delta = \gamma +
        \mathcal{O}(10^{-3})$. If `2`, the next-to-leading order correction is
        included, $\delta = \gamma + \mathcal{O}(10^{-3}) +
        \mathcal{O}(10^{-6})$.

    Returns
    -------
    - `v`: CKM matrix in the beta-gamma parametrization and standard phase
    """
    t12, t13, t23, delta = beta_gamma_to_standard(Vus, Vcb, beta, gamma, delta_expansion_order)
    return ckm_standard(t12, t13, t23, delta)
