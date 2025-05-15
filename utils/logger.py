import os

class TrainingLogger:
    def __init__(self, log_dir="./logs", project_name="siglip_classification",
                 experiment_name=None, backends=None,
                 wandb_api_key=None, wandb_entity=None, wandb_config=None):
        self.backends = backends or []
        self.loggers = {}
        if 'tensorboard' in self.backends:
            from torch.utils.tensorboard import SummaryWriter
            self.loggers['tensorboard'] = SummaryWriter(log_dir)
        if 'wandb' in self.backends:
            import wandb
            if wandb_api_key:
                wandb.login(key=wandb_api_key, relogin=True)
            wandb_init_args = {
                "project": project_name,
                "name": experiment_name,
                "config": wandb_config or {"log_dir": log_dir},
            }
            if wandb_entity:
                wandb_init_args["entity"] = wandb_entity
            if not wandb.run:
                wandb.init(**wandb_init_args)
            self.loggers['wandb'] = wandb
        self.current_step = 0

    def log_metrics(self, metrics, step=None, prefix=''):
        if step is None:
            step = self.current_step
        if prefix and not prefix.endswith('/'):
            prefix = f"{prefix}/"
        prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        for backend in self.backends:
            if backend == 'tensorboard':
                for name, value in prefixed_metrics.items():
                    self.loggers[backend].add_scalar(name, value, step)
            elif backend == 'wandb':
                self.loggers[backend].log(prefixed_metrics, step=step)
            elif backend == 'comet':
                self.loggers[backend].log_metrics(prefixed_metrics, step=step)

    def step(self):
        self.current_step += 1

    def log_model(self, model, name='model'):
        for backend in self.backends:
            if backend == 'wandb':
                self.loggers[backend].watch(model)

    def log_hyperparams(self, params):
        for backend in self.backends:
            if backend == 'tensorboard':
                self.loggers[backend].add_hparams(params, {})
            elif backend == 'wandb':
                self.loggers[backend].config.update(params)
            elif backend == 'comet':
                self.loggers[backend].log_parameters(params)

    def close(self):
        for backend in self.backends:
            if backend == 'tensorboard':
                self.loggers[backend].close()
            elif backend == 'wandb':
                self.loggers[backend].finish()
            elif backend == 'comet':
                self.loggers[backend].close()