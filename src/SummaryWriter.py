from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch.utils.tensorboard as tensorboard

class SummaryWriter(tensorboard.SummaryWriter):
    def __init__(self, log_dir=None):
        super().__init__(log_dir)
    
    def load_state(self, event_file, num_events=None):
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        for tag in event_acc.scalars.Keys():
            events = event_acc.Scalars(tag)
            if num_events == None:
                num_events = len(events)
            for event in events[:num_events]:
                self.add_scalar(tag, scalar_value=event.value, global_step=event.step, walltime=event.wall_time)