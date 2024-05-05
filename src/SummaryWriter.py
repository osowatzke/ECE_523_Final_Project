from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch.utils.tensorboard as tensorboard

class SummaryWriter(tensorboard.SummaryWriter):
    def __init__(self, log_dir=None):
        super().__init__(log_dir)
    
    def load_state(self, event_file, max_step=None):
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        for tag in event_acc.scalars.Keys():
            events = event_acc.Scalars(tag)
            for event in events:
                if event.step <= max_step:
                    self.add_scalar(tag, scalar_value=event.value, global_step=event.step, walltime=event.wall_time)