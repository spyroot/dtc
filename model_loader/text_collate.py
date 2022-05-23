import torch
import torch.utils.data
from torchtext.data.utils import get_tokenizer


class TextCollate:
    """
    If we need split audio and text to separate batch and train separately.

    Usage.
    train_iter = data(split='train')
    dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=TextCollate)
    """

    def __init__(self, label_callback,
                 text_pipeline, device, tokenizer='basic_english', n_frames_per_step=1, is_trace_time=False):
        """

        :param n_frames_per_step:
        :param is_trace_time:
        """
        self.tokenizer = get_tokenizer('basic_english')
        self.device = device
        self.label_pipeline = label_callback
        self.text_pipeline = text_pipeline

    def __call__(self, batch):
        """
        Build a batch and return label , text, offset
        :param batch:
        :return:
        """
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)

        if self.device is not None:
            label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)

        return label_list, text_list, offsets
