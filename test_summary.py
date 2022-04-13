from torch.utils.tensorboard import SummaryWriter
import numpy as np
import glob
import os
import time
import timeit
from datetime import datetime
import socket

writer = SummaryWriter(log_dir='C:/Users/whiwho/Documents/GitHub/SNN_TFG/run/run_11/')

print(f"Training dir {'C:/Users/whiwho/Documents/GitHub/SNN_TFG/run/run_11/'}")

for n_iter in range(100):
    time.sleep(2)
    print(f"Caso {n_iter}")
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)