from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./log")
x=range(100)

for i in x:
    writer.add_scalar('acc', i, 10)
