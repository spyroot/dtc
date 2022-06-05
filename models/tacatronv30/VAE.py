import torch
from torch import nn
from torch.nn import functional as F

from models.tacatronv30.inference_decoder import InferenceDecoder
from models.tacatronv30.inference_encoder import InferenceEncoder
from torch.distributions.normal import Normal


class VAE(nn.Module):
    def __init__(self, z_dim=2):
        super().__init__()

        self.z_dim = z_dim
        self.enc = InferenceEncoder(self.z_dim)
        self.dec = InferenceDecoder(self.z_dim)

        # prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1),  requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        # self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        # self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    @staticmethod
    def sample_gaussian(m, v):
        """

        :param v: tensor: (batch, ...): Variance
        :param m: tensor: (batch, ...): Mean
        :return:
        """
        epsilon = torch.randn_like(v)
        z = m + torch.sqrt(v) * epsilon
        return z

    # def gaussian_likelihood(self, x_hat, logscale, x):
    #     scale = torch.exp(logscale)
    #     mean = x_hat
    #     dist = torch.distributions.Normal(mean, scale)
    #
    #     # measure prob of seeing image under p(x|z)
    #     log_pxz = dist.log_prob(x)
    #     return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(self, x):
        """

        :param spectral:
        :return:
        """
        mu, q_var = self.vae_encode(x)
        std = torch.exp(q_var / 2)
        q = Normal(mu, std)
        z = q.rsample()

        x_hat = self.vae_decode(z)
        recon_loss = -self.bce(input=x_hat, target=x).sum(-1)

        # recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        kl = self.kl_divergence(z, mu, std)

        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        print(elbo)
        print(kl.mean())
        return elbo, kl.mean()

        # self.log_dict({
        #     'elbo': elbo,
        #     'kl': kl.mean(),
        #     'recon_loss': recon_loss.mean(),
        #     'reconstruction': recon_loss.mean(),
        # })

        # m, v = self.enc(x)
        # # sample q(z|x)
        # z = ut.sample_gaussian(m, v)
        # sampled = self.dec(z)
        #
        # kl = ut.kl_normal(m, v, self.z_prior[0], self.z_prior[1])
        # rec = -ut.log_bernoulli_with_logits(x, sampled)
        # nelbo = rec + kl
        # # return mean
        # nelbo, kl, rec = nelbo.mean(), kl.mean(), rec.mean()
        #
