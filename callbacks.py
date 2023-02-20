from fastai.core import PathOrStr, Union, Any, warn
import glob
import fastai, time, argparse, os
from fastai.distributed import *
from fastai.callbacks.tracker import TrackerCallback
from fastai.callbacks.tracker import SaveModelCallback
from fastai.torch_core import rank_distrib
from torch.distributed import *
from torch import Tensor


class CustomSaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn,
                 monitor: str = 'valid_loss',
                 mode: str = 'auto',
                 every: str = 'improvement',
                 name: str = 'bestmodel',
                 logger=None):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.logger = logger
        self.every, self.name = every, name
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
        self.best = None

    def jump_to_epoch(self, epoch: int) -> None:
        try:
            self.learn.load(f'{self.name}_{epoch-1}', purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except: print(f'Model {self.name}_{epoch-1} not found.')

    def sort_checkpoint(self):
        pattern = os.path.join(self.learn.model_dir, self.name + '_' + self.monitor + '*')
        checkpoints = glob.glob(pattern)
        checkpoints.sort(key=lambda x: float(x.split('_')[-1].rstrip('.pth')))
        return checkpoints

    def on_epoch_end(self, epoch: int, **kwargs:Any)->None:
        """ Compare the value monitored to its best score and maybe save the model."""
        if rank_distrib():
            return  # don't save checkpoint if slave proc
        if self.every == "epoch":
            self.learn.save(f'{self.name}_{epoch}', with_opt=False)
        else:
            current = self.get_monitor_value()
            if isinstance(current, Tensor): current = current.cpu()
            if current is not None and self.operator(current, self.best):
                if self.logger is not None:
                    self.logger.info(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                else:
                    print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                self.best = current
                # check if checkpoint directory to save a maximum of 3 models:
                if len(os.listdir(self.learn.model_dir)) > 2:
                    checkpoints = self.sort_checkpoint()
                    # delete the worst checkpoint:
                    if self.mode == 'min':
                        if self.logger:
                            self.logger.info('delete checkpoint file: %s' % checkpoints[-1])
                        os.remove(checkpoints[-1])
                    else:
                        if self.logger:
                            self.logger.info('delete checkpoint file: %s' % checkpoints[0])
                        os.remove(checkpoints[0])

                self.learn.save(f'{self.name}_{self.monitor}_{round(float(current), 4)}', with_opt=False)

    def on_train_end(self, **kwargs):
        """Load the best model."""
        checkpoints = self.sort_checkpoint()
        if os.listdir(self.learn.model_dir):
            if self.mode == 'min':
                base = os.path.basename(checkpoints[0])
            else:
                base = os.path.basename(checkpoints[-1])
            last_check = os.path.splitext(base)[0]
            if self.every == "improvement" and os.path.isfile(self.path/self.model_dir/f'{last_check}.pth'):
                print("loading final parameters from %s\n" % os.path.join(self.model_dir, last_check))
                self.learn.load(f'{last_check}', purge=True)
