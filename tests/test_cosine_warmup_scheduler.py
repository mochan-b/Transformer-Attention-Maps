from torch import nn, optim
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from cosine_warmup_scheduler import CosineWarmupScheduler


def test_cosine_warmup_scheduler():
    """
    Smoke test for the CosineWarmupScheduler class. Just make sure it doesn't crash.
    Can run the plot to check that the right kind of schedule is being used.
    """
    # Needed for initializing the lr scheduler
    p = nn.Parameter(torch.empty(4, 4))
    optimizer = optim.Adam([p], lr=1e-3)
    lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=2000)

    # Plotting
    draw_plot = False
    if draw_plot:
        epochs = list(range(2000))
        sns.set()
        plt.figure(figsize=(8, 3))
        plt.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
        plt.ylabel("Learning rate factor")
        plt.xlabel("Iterations (in batches)")
        plt.title("Cosine Warm-up Learning Rate Scheduler")
        plt.show()
        sns.reset_orig()
