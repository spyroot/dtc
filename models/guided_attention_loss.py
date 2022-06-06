import torch


class GuidedAttentionLoss(torch.nn.Module):
    """
        A loss implementation forces attention matrices to be
        near-diagonal, imposing progressively larger penalties for paying
        attention to regions far away from the diagonal). It is useful
        for sequence-to-sequence models in which the sequence of outputs
        is expected to corrsespond closely to the sequence of inputs,
        such as TTS or G2P

        https://arxiv.org/abs/1710.08969

    """
    def __init__(self, sigma=0.4):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.weight_factor = 2 * (sigma ** 2)

    def make_guided_mask(self, ilen, olen):
        grid_x, grid_y = torch.meshgrid(torch.arange(olen).to(olen),
                                        torch.arange(ilen).to(ilen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(- (grid_y / ilen - grid_x / olen) ** 2 / self.weight_factor)

    def _make_ga_masks(self, B, i_lens, o_lens):
        """

        :param i_lens:
        :param o_lens:
        :return:
        """
        max_ilen = max(i_lens)
        max_olen = max(i_lens)

        ga_masks = torch.zeros((B, max_olen, max_ilen))
        for idx, (i_len, o_len) in enumerate(zip(i_lens, o_lens)):
            ga_masks[idx, :o_len, :i_len] = self.make_guided_mask(i_len, o_len, self.sigma)

        return ga_masks

    def forward(self, att_ws, input_lengths, output_lengths):
        """
        :param att_ws:
        :param input_lengths:
        :param output_lengths:
        :return:  The guided attention tensor of shape (batch, max_input_len, max_target_len)
        """
        ga_masks = self._make_ga_masks(input_lengths, output_lengths).to(att_ws.device)
        seq_masks = self._make_masks(input_lengths, output_lengths).to(att_ws.device)
        losses = ga_masks * att_ws
        loss = torch.mean(losses.masked_select(seq_masks))
        return loss

    @staticmethod
    def _make_masks(i_lens, o_lens):
        in_masks = sequence_mask(i_lens)
        out_masks = sequence_mask(o_lens)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)


def test_guided_attention_test_value():

    loss = GuidedAttentionLoss()
    input_lengths = torch.tensor([2, 3])
    target_lengths = torch.tensor([3, 4])
    alignments = torch.tensor(
        [
            [
                [0.8, 0.2, 0.0],
                [0.4, 0.6, 0.0],
                [0.2, 0.8, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.6, 0.2, 0.2],
                [0.1, 0.7, 0.2],
                [0.3, 0.4, 0.3],
                [0.2, 0.3, 0.5],
            ],
        ]
    )
    loss_value = loss(alignments, input_lengths, target_lengths)
    ref_loss_value = torch.tensor(0.1142)
    assert torch.isclose(loss_value, ref_loss_value, 0.0001, 0.0001).item()