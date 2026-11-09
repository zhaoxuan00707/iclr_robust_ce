"""Compact factor preserving norms of sums of candidate feature adjoints."""
import torch

def factor_adjoint(adjoint):
 gram=adjoint@adjoint.T
 eigenvalues,eigenvectors=torch.linalg.eigh(gram)
 scale=gram.diagonal().max().clamp_min(1e-30)
 if eigenvalues.min() < -1e-10*scale:raise ValueError('Unexpected non-PSD gradient Gram matrix')
 factor=eigenvectors*eigenvalues.clamp_min(0).sqrt()[None,:]
 torch.testing.assert_close(factor@factor.T,gram,atol=float(scale)*1e-10,rtol=1e-9)
 return factor
