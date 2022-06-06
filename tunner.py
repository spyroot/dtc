import os

from model_loader.stft_dataloader import SFTFDataloader
from model_trainer.internal.call_interface import Callback
from model_trainer.trainer import Trainer

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
except ImportError:
    pass


class Trainable(tune.Trainable, Callback):
    """

    """

    def setup(self, config):
        """

        :param config:
        :return:
        """
        self.config = config
        spec = config['spec']

        batch_size = 1
        if 'batch_size' in config:
            spec.set_batch_size(int(config['batch_size']))
            assert spec.batch_size() == int(config['batch_size'])
        else:
            batch_size = spec.batch_size()

        if 'grad_clip' in config:
            config['spec'].update_grad_clip(config['grad_clip'])

        SFTFDataloader.set_logger(False)
        data = SFTFDataloader(config['spec'],
                              batch_size=batch_size,
                              rank=0,
                              world_size=config['world_size'],
                              verbose=False)

        Trainer.set_logger(False)
        self.trainer = Trainer(config['spec'],
                               data_loader=data,
                               rank=0,
                               world_size=config['world_size'],
                               verbose=False,
                               device=config['device'],
                               hp_tunner=True,
                               disable_pbar=True)

        # logger.add(self.trainer.model_files.get_model_log_file_path(remove_old=True),
        #            format="{elapsed} {level} {message}",
        #            filter="model_trainer.trainer_metrics", level="INFO", rotation="1h")
        #
        # logger.add(self.trainer.model_files.get_trace_log_file("loader"),
        #            format="{elapsed} {level} {message}",
        #            filter="model_loader.stft_dataloader", level="INFO", rotation="1h")
        #
        # logger.add(self.trainer.model_files.get_trace_log_file("trainer"),
        #            format="{elapsed} {level} {message}",
        #            filter="model_trainer.trainer", level="INFO", rotation="1h")

        self.trainer.set_logger(is_enable=False)
        self.trainer.metric.set_logger(is_enable=False)

    def on_epoch_begin(self):
        # loss = validation_loss)
        pass

    def on_epoch_end(self):
        pass

    def save_checkpoint(self, tmp_dir):
        """
        :param tmp_dir:
        :return:
        """
        checkpoint_path = os.path.join(tmp_dir, "model.pth")
        self.trainer.save_model_layer("spectrogram_layer", checkpoint_path)

        return checkpoint_path

    def load_checkpoint(self, tmp_dir):
        """
        It is separate load and save routines.  Ray keeps own checkpoints.

        :param tmp_dir:
        :return:
        """
        checkpoint_path = os.path.join(tmp_dir, "model.pth")
        self.trainer.load_model_layer("spectrogram_layer", checkpoint_path)
        return checkpoint_path

    def step(self):
        """
        We execute each step and collect metrics.

        :return:
        """
        self.trainer.train_optimizer(self.config)
        self.trainer.metric.batch_val_loss.mean()
        print(f"## train epoch grand norm loss  {self.trainer.metric.epoch_train_gn_loss.sum():.3f}")
        print(f"## train epoch loss mean        {self.trainer.metric.epoch_train_loss.sum():.3f}")
        print(f"## validation epoch             {self.trainer.metric.epoch_val_loss.sum():.3f}")

        return {
            "mean_train_loss": self.trainer.metric.epoch_train_gn_loss.sum(),
            "mean_val_loss": self.trainer.metric.epoch_val_loss.sum()
        }