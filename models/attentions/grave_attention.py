import torch
from torch import nn
from torch.nn import functional as F


class GravesAttention(nn.Module):
    """
    Graves Attention.`

    Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis

    ref2: https://arxiv.org/pdf/1906.01083.pdf
    Args:
        query_dim (int): number of channels in query tensor.
        K (int): number of Gaussian heads to be used for computing attention.
    """

    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))

    def __init__(self, query_dim, K):
        """

        :param query_dim:
        :param K:
        """
        super(GravesAttention, self).__init__()
        self._mask_value = 1e-8
        self.K = K
        # self.attention_alignment = 0.05
        self.eps = 1e-5
        self.J = None
        self.N_a = nn.Sequential(
                nn.Linear(query_dim, query_dim, bias=True),
                nn.ReLU(),
                nn.Linear(query_dim, 3 * K, bias=True))
        self.attention_weights = None
        self.mu_prev = None
        self.init_layers()

    def init_layers(self):
        """

        :return:
        """
        torch.nn.init.constant_(self.N_a[2].bias[(2*self.K):(3*self.K)], 1.)  # bias mean
        torch.nn.init.constant_(self.N_a[2].bias[self.K:(2*self.K)], 10)  # bias std

    def init_states(self, inputs):
        """

        :param inputs:
        :return:
        """
        if self.J is None or inputs.shape[1] + 1 > self.J.shape[-1]:
            self.J = torch.arange(0, inputs.shape[1] + 2.0).to(inputs.device) + 0.5

        self.attention_weights = torch.zeros(inputs.shape[0], inputs.shape[1]).to(inputs.device)
        self.mu_prev = torch.zeros(inputs.shape[0], self.K).to(inputs.device)

    def preprocess_inputs(self, inputs):
        return None

    def forward(self, query, inputs, processed_inputs, mask):
        """
        Shapes:
            query: [B, C_attention_rnn]
            inputs: [B, T_in, C_encoder]
            processed_inputs: place_holder
            mask: [B, T_in]
        """
        gbk_t = self.N_a(query)
        gbk_t = gbk_t.view(gbk_t.size(0), -1, self.K)

        # attention model parameters
        # each B x K
        g_t = gbk_t[:, 0, :]
        b_t = gbk_t[:, 1, :]
        k_t = gbk_t[:, 2, :]

        # dropout to decorrelate attention heads
        g_t = F.dropout(g_t, p=0.5, training=self.training)

        # attention GMM parameters
        sig_t = F.softplus(b_t) + self.eps

        mu_t = self.mu_prev + F.softplus(k_t)
        g_t = torch.softmax(g_t, dim=-1) + self.eps

        j = self.J[:inputs.size(1)+1]

        # attention weights
        phi_t = g_t.unsqueeze(-1) * (1 / (1 + torch.sigmoid((mu_t.unsqueeze(-1) - j) / sig_t.unsqueeze(-1))))

        # discritize attention weights
        alpha_t = torch.sum(phi_t, 1)
        alpha_t = alpha_t[:, 1:] - alpha_t[:, :-1]
        alpha_t[alpha_t == 0] = 1e-8

        # apply masking
        if mask is not None:
            alpha_t.data.masked_fill_(~mask, self._mask_value)

        context = torch.bmm(alpha_t.unsqueeze(1), inputs).squeeze(1)
        self.attention_weights = alpha_t
        self.mu_prev = mu_t
        return context
