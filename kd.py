import torch
from config import update_config_to_current
from models import build_model

class KDTeacher:
    def __init__(self, config) -> None:
        if config.DISTILL.TEACHER_CKPT is None:
            raise ValueError("No teacher checkpoint provided")

        ckpt = torch.load(config.DISTILL.TEACHER_CKPT, map_location='cpu')
        self.teacher_config = update_config_to_current(ckpt['config'])
        self.model = build_model(self.teacher_config)
        self.model.load_state_dict(ckpt['model'], strict=False)
        # Important - teacher_config is teacher's pretraining/ft config
        self.temp = config.DISTILL.TEMPERATURE
        self.alpha = config.DISTILL.ALPHA
        del ckpt

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def cuda(self):
        self.model.cuda()
    
    def eval(self):
        self.model.eval()

    def ddp(self, device_ids: list, broadcast_buffers=True):
        self.model_without_ddp = self.model
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=device_ids, broadcast_buffers=broadcast_buffers)