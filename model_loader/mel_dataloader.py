from torch.utils.data import DataLoader, DistributedSampler
from model_loader.mel_dataset_loader import TextMelLoader, TextMelCollate
from model_trainer.model_trainer_specs import ExperimentSpecs


class Mel_Dataloader:
    """

    """

    def __init__(self, experiment_specs: ExperimentSpecs):
        """

        :param experiment_specs:
        """
        self.model_trainer_spec = experiment_specs
        self.model_spec = experiment_specs.get_model_spec()
        self.encoder_spec = self.model_spec.get_encoder()

    def create(self):
        """

        :return:
        """
        training_set, validation_set, test_set = self.model_trainer_spec.get_audio_ds_files()
        train_set = TextMelLoader(self.encoder_spec, list(training_set.values()))
        val_set = TextMelLoader(self.encoder_spec, list(validation_set.values()))
        collate_fn = TextMelCollate(self.encoder_spec.n_frames_per_step)

        if self.model_trainer_spec.is_distributed_run():
            train_sampler = DistributedSampler(train_set)
            shuffle = False
        else:
            train_sampler = None
            shuffle = True

        train_loader = DataLoader(train_set, num_workers=1, shuffle=shuffle,
                                  sampler=train_sampler,
                                  batch_size=self.model_trainer_spec.batch_size,
                                  pin_memory=False,
                                  drop_last=True, collate_fn=collate_fn)

        return train_loader, val_set, collate_fn
