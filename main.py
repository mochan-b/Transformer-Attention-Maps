import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils import data
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from leftright_product_dataset import LeftRightProductDataset
from reverse_dataset import ReverseDataset
from reverse_predictor import SimplePredictor

CHECKPOINT_PATH = "checkpoints"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.set_float32_matmul_precision('medium')


def train_reverse(train, **kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "ReverseTask")
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=20,
                         gradient_clip_val=5)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ReverseTask.ckpt")
    if not train and os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = SimplePredictor.load_from_checkpoint(pretrained_filename)
    else:
        model = SimplePredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
        trainer.fit(model, train_loader, val_loader)
        trainer.save_checkpoint(pretrained_filename)

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test_acc": test_result[0]["test_acc"], "val_acc": val_result[0]["test_acc"]}

    model = model.to(device)
    return model, result


def plot_attention_maps(input_data, attn_maps, idx=0, avg=False):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    # If avg is True, we average over the heads and also over the layers
    if avg:
        # Sum up the layers in the list
        attn_maps_mean = sum(attn_maps) / len(attn_maps)
        # Mean over the heads and put the head dimension as 1
        attn_maps_heads = np.mean(attn_maps_mean, axis=0)
        attn_maps_heads = np.expand_dims(attn_maps_heads, axis=0)
        attn_maps = [attn_maps_heads]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title(f"Layer {row + 1}, Head {column + 1}")
    # fig.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # adjust the space between plots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Options for running the script
    always_train = True
    use_pytorch_transformer = True

    num_categories = 10

    dataset_type = 'left_right_product'

    if dataset_type == 'reverse':
        dataset = partial(ReverseDataset, num_categories, 16)
    elif dataset_type == 'left_right_product':
        dataset = partial(LeftRightProductDataset, num_categories, 16, shift=5)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_loader = data.DataLoader(dataset(50000), batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = data.DataLoader(dataset(1000), batch_size=128)
    test_loader = data.DataLoader(dataset(10000), batch_size=128)

    reverse_model, reverse_result = train_reverse(train=always_train,
                                                  input_dim=num_categories,
                                                  model_dim=32,
                                                  num_heads=2,
                                                  num_classes=num_categories,
                                                  num_layers=2,
                                                  dropout=0.0,
                                                  lr=5e-4,
                                                  warmup=50,
                                                  use_pytorch_transformer=use_pytorch_transformer)
    print(f"Val accuracy:  {(100.0 * reverse_result['val_acc']):4.2f}%")
    print(f"Test accuracy: {(100.0 * reverse_result['test_acc']):4.2f}%")

    draw_attention_maps = True
    if draw_attention_maps:
        data_input, labels = next(iter(val_loader))
        inp_data = F.one_hot(data_input, num_classes=reverse_model.hparams.num_classes).float()
        inp_data = inp_data.to(device)
        attention_maps = reverse_model.get_attention_maps(inp_data)

        print(attention_maps[0].shape)

        plot_attention_maps(data_input, attention_maps, idx=0, avg=True)
