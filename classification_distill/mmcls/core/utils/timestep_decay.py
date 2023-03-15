from mmcv.runner import (HOOKS, Hook)
from mmcv.runner import get_dist_info

@HOOKS.register_module()
class TimeDecayHook(Hook):
    def __init__(self, init_timestep=1000, end_timestep=50, end_epoch=100):
        super(TimeDecayHook, self).__init__()
        self.init_timestep=init_timestep
        self.end_timestep = end_timestep
        self.end_epoch = end_epoch

    def before_run(self, runner):
        super().before_run(runner)
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        model.max_time_step = int(self.init_timestep)

    def before_train_epoch(self, runner):
        if runner.epoch > self.end_epoch:
            return
        super().before_train_epoch(runner)
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        # Linear Shedule
        ratio = runner.epoch / self.end_epoch
        model.max_time_step = int(self.init_timestep - (self.init_timestep - self.end_timestep) * ratio)
        rank, _ = get_dist_info()
        if rank == 0:
            print(f"Current Max Time Step : {model.max_time_step}; Ratio {ratio}")

@HOOKS.register_module()
class EntropyDecayHook(Hook):
    def __init__(self, init_entropy_reg=0.1, end_epoch=100):
        super(EntropyDecayHook, self).__init__()
        self.init_entropy_reg=init_entropy_reg
        self.end_epoch = end_epoch

    def before_run(self, runner):
        super().before_run(runner)
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        model.entropy_reg = self.init_entropy_reg

    def before_train_epoch(self, runner):
        if runner.epoch > self.end_epoch:
            return
        super().before_train_epoch(runner)
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        # Linear Shedule
        ratio = runner.epoch / self.end_epoch
        model.entropy_reg = self.init_entropy_reg * (1-ratio)
        rank, _ = get_dist_info()
        if rank == 0:
            print(f"Current Entropy Reg : {model.entropy_reg}; Ratio {ratio}")


        
