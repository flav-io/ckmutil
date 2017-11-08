"""Functions for the diagonalization of fermion mass matrices."""

import numpy as np

def msvd(m):
  """Modified singular value decomposition.

  Returns U, S, V where Udagger M V = diag(S) and the singular values
  are sorted in ascending order (small to large).
  """
  u, s, vdgr = np.linalg.svd(m)
  order = s.argsort()
  # reverse the n first columns of u
  s = s[order]
  u= u[:,order]
  vdgr = vdgr[order]
  return u, s, vdgr.conj().T
