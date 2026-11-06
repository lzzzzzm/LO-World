import torch

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        """
        Gaussian Distribution with diagonal covariance matrix.
        Args:
            parameters: [B, 2*C, ...]. First C channels = mean, Second C = logvar.
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        eps = torch.randn(self.mean.shape, device=self.parameters.device)
        return self.mean + self.std * eps

    def kl(self) -> torch.Tensor:
        if self.deterministic:
            return torch.tensor([0.], device=self.parameters.device)
        return 0.5 * torch.sum(
            torch.exp(self.logvar) + torch.pow(self.mean, 2) - 1.0 - self.logvar,
            dim=[1, 2]
        )