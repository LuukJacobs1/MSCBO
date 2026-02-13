from typing import List
from causalbo.do_calculus import SCM
from botorch.acquisition import AcquisitionFunction
from gpytorch.means.mean import Mean
from gpytorch.kernels import RBFKernel
import torch


# -----------------------------
# Mean function using batched + cached do-statistics
# -----------------------------
class CausalMean(Mean):
    """
    Mean(x) = E[Y | do(V = x)]
    Uses SCM.do_mean_var_batch under the hood (batched + cached).
    """
    def __init__(self, interventional_variable: List[str], causal_model: SCM, n_samples: int = 200, parallel: bool = True):
        super().__init__()
        self.interventional_variable = interventional_variable
        self.causal_model = causal_model
        self.n_samples = n_samples
        self.parallel = parallel

    def forward(self, x: torch.Tensor):
        # x: (..., d)
        if x.shape[-1] != len(self.interventional_variable):
            raise ValueError("Shape of data does not match number of interventional variables!")

        # Flatten to (N, d)
        x_flat = x.view(-1, x.size(-1))

        # DoWhy runs on CPU; convert to numpy
        x_np = x_flat.detach().cpu().numpy()

        means, _ = self.causal_model.do_mean_var_batch(
            interventional_variable=self.interventional_variable,
            interventional_values=x_np,
            n_samples=self.n_samples,
            parallel=self.parallel
        )

        # Back to torch on the same device/dtype as input
        mean_out = torch.from_numpy(means).to(device=x.device, dtype=x.dtype)
        return mean_out.view(*x.shape[:-1])


# -----------------------------
# Kernel with heteroskedastic augmentation: k(x,x') = k_RBF(x,x') + sigma(x)sigma(x')
# -----------------------------
class CausalRBF(RBFKernel):
    """
    Adds heteroskedastic term using sqrt(Var[Y | do(V=x)]).
    Uses SCM.do_mean_var_batch for batched variance queries (cached).
    """
    def __init__(
        self,
        interventional_variable: List[str],
        causal_model: SCM,
        ard_num_dims=None,
        batch_shape=None,
        active_dims=None,
        lengthscale_prior=None,
        lengthscale_constraint=None,
        eps=1e-06,
        n_samples: int = 200,
        parallel: bool = True,
        **kwargs
    ):
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
            eps=eps,
            **kwargs
        )
        self.interventional_variable = interventional_variable
        self.causal_model = causal_model
        self.n_samples = n_samples
        self.parallel = parallel

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        # x1: (..., d), x2: (..., d)
        d = len(self.interventional_variable)
        if x1.shape[-1] != d or x2.shape[-1] != d:
            raise ValueError("Shape of data does not match number of interventional variables!")
        
        # Base RBF part
        base = super().forward(x1, x2, diag=diag, **params)

        # For the heteroskedastic term, we need sqrt(var(do(x)))
        # Flatten to (N, d)
        x1_flat = x1.view(-1, d)
        x2_flat = x2.view(-1, d)

        # To numpy (CPU) for DoWhy
        x1_np = x1_flat.detach().cpu().numpy()
        x2_np = x2_flat.detach().cpu().numpy()

        # Only need variances; do_mean_var_batch returns (means, vars)
        _, vars1 = self.causal_model.do_mean_var_batch(
            interventional_variable=self.interventional_variable,
            interventional_values=x1_np,
            n_samples=self.n_samples,
            parallel=self.parallel
        )
        _, vars2 = self.causal_model.do_mean_var_batch(
            interventional_variable=self.interventional_variable,
            interventional_values=x2_np,
            n_samples=self.n_samples,
            parallel=self.parallel
        )

        # Convert to torch on same device/dtype
        sig1 = torch.from_numpy(vars1).to(device=x1.device, dtype=x1.dtype).sqrt()
        sig2 = torch.from_numpy(vars2).to(device=x2.device, dtype=x2.dtype).sqrt()

        # Reshape back to broadcastable shapes
        sig1 = sig1.view(*x1.shape[:-1], 1)  # (..., 1)
        sig2 = sig2.view(*x2.shape[:-1], 1)  # (..., 1)

        # Outer product term sigma(x1) * sigma(x2)^T matches RBF output shape
        # When diag=True, gpytorch expects the diagonal of the full kernel matrix.
        if diag:
            # For diag, x1 and x2 are the same batch shapes; the diag term is elementwise product
            # of sigmas plus base diag from RBF.
            # GPyTorch gives base as (...), so we add (sig1*sig2).squeeze(-1)
            het = (sig1.squeeze(-1) * sig2.squeeze(-1))
            return base + het

        # For full matrix, need pairwise product: broadcast over last two dims
        het = sig1 * sig2.transpose(-2, -1)
        return base + het


# -----------------------------
# Utility: negate any acquisition function (for minimization)
# -----------------------------
class NegateAcquisitionFunction(AcquisitionFunction):
    def __init__(self, acq_function: AcquisitionFunction):
        super().__init__(acq_function.model)
        self.acq_function = acq_function
    
    def forward(self, X: torch.Tensor):
        return -self.acq_function(X)
