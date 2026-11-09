"""Fixed gradient task subspace with factual-specific complementary content."""
import torch


def gradient_basis(gradients, energy=0.95):
    """Uncentered SVD: preserve energy of score gradients, not latent variance."""
    _, singular, vh = torch.linalg.svd(gradients.double(), full_matrices=True)
    cumulative = singular.square().cumsum(0) / singular.square().sum()
    k = int(torch.searchsorted(cumulative, torch.tensor(energy, dtype=torch.float64))) + 1
    return vh[:k].T.contiguous(), vh[k:].T.contiguous(), singular, k


class FrozenContent:
    def __init__(self, basis, factual):
        self.basis = basis.detach().double().to(factual.device)
        self.factual = factual.detach().double()
        self.content = (self.factual - (self.factual @ self.basis) @ self.basis.T).detach()

    def encode(self, latent):
        return latent.double() @ self.basis

    def realize(self, task):
        # Particle row i always retains factual i. OT permutations affect only
        # task-space anchor assignment; they never permute factual content.
        return (self.content + task.double() @ self.basis.T).float()


class ConditionalBank:
    def __init__(self, full_bank, realization):
        self.full = full_bank
        self.realization = realization
        self.count = full_bank.count

    def logits(self, task, start=0, end=None):
        return self.full.logits(self.realization.realize(task), start, end)

    def risk(self, task):
        return self.full.risk(self.realization.realize(task))

    def canonical(self, task):
        return self.full.canonical(self.realization.realize(task))

    def cached_features(self, task):
        return self.full.cached_features(self.realization.realize(task))
