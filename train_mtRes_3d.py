import os
import math
import torch
import gunpowder as gp
import numpy as np
from funlib.learn.torch.models import UNet, ConvPass
from torch.utils.tensorboard import SummaryWriter
from gunpowder.torch import Train


# Constants
downsample_factor = 2
voxel_size = gp.Coordinate((4, 4, 4))
voxel_size_res2 = voxel_size * downsample_factor
input_shape = gp.Coordinate((168, 168, 168))
output_shape = gp.Coordinate((128, 128, 128))
output_shape_res2 = gp.Coordinate((34, 34, 34))
input_size_res1 = input_shape * voxel_size
output_size_res1 = output_shape * voxel_size
input_size_res2 = input_size_res1 // downsample_factor
output_size_res2 = output_shape_res2 * voxel_size_res2
samples = ['datasets/training_data_656by656by656_volume0.zarr', 'datasets/training_data_656by656by656_volume1.zarr', 'datasets/training_data_656by656by656_volume2.zarr']
neighborhood = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
batch_size = 1


class affsMlResUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.unet_res1 = UNet(
            in_channels=1,
            num_fmaps=16,
            fmap_inc_factor=5,
            downsample_factors=[
                [2, 2, 2],
                [2, 2, 2]],
            kernel_size_down=[
                [[3, 3, 3], [3, 3, 3]],
                [[3, 3, 3], [3, 3, 3]],
                [[3, 3, 3], [3, 3, 3]]],
            kernel_size_up=[
                [[3, 3, 3], [3, 3, 3]],
                [[3, 3, 3], [3, 3, 3]]])

        self.unet_res2 = UNet(
            in_channels=1,
            num_fmaps=12,
            fmap_inc_factor=5,
            downsample_factors=[
                [2, 2, 2]],
            kernel_size_down=[
                [[2, 2, 2], [2, 2, 2]],
                [[2, 2, 2], [2, 2, 2]]],
            kernel_size_up=[
                [[2, 2, 2], [2, 2, 2]]])

        self.aff_head_res1 = ConvPass(16, 3, [[1, 1, 1]], activation='Sigmoid')
        self.aff_head_res2 = ConvPass(12, 3, [[1, 1, 1]], activation='Sigmoid')

    def forward(self, input_res1, input_res2):
        z_res1 = self.unet_res1(input_res1)
        z_res2 = self.unet_res2(input_res2)
        affs_res1 = self.aff_head_res1(z_res1)
        affs_res2 = self.aff_head_res2(z_res2)

        return affs_res1, affs_res2


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self):
        super().__init__()

    def forward(self, res1_pred, res1_target, res1_weights, res2_pred, res2_target, res2_weights):     
        loss1 = super().forward(res1_pred * res1_weights, res1_target * res1_weights)
        loss2 = super().forward(res2_pred * res2_weights, res2_target * res2_weights)
        return loss1 + loss2

def train(
    checkpoint_name,
    dir,
    max_iteration,
    trained_until=0,
    save_every=100,
    latest_checkpoint=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = affsMlResUNet().to(device)

    loss = WeightedMSELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)

    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Define array keys
    raw, labels = gp.ArrayKey('RAW'), gp.ArrayKey('LABELS')
    raw_res2, labels_res2 = gp.ArrayKey('RAW_RES2'), gp.ArrayKey('LABELS_RES2')
    gt_affs_res1, affs_weights_res1 = gp.ArrayKey('GT_AFFS_RES1'), gp.ArrayKey('AFFS_WEIGHTS_RES1')
    pred_affs_res1 = gp.ArrayKey('PRED_AFFS_RES1')
    gt_affs_res2, affs_weights_res2 = gp.ArrayKey('GT_AFFS_RES2'), gp.ArrayKey('AFFS_WEIGHTS_RES2')
    pred_affs_res2 = gp.ArrayKey('PRED_AFFS_RES2')

    # Create a batch request
    request = gp.BatchRequest()
    request.add(raw, input_size_res1)
    request.add(labels, output_size_res1)
    request.add(raw_res2, input_size_res2)
    request.add(labels_res2, output_size_res2)
    request.add(gt_affs_res1, output_size_res1)
    request.add(affs_weights_res1, output_size_res1)
    request.add(pred_affs_res1, output_size_res1)
    request.add(gt_affs_res2, output_size_res2)
    request.add(affs_weights_res2, output_size_res2)
    request.add(pred_affs_res2, output_size_res2)

    # Create the sources
    sources = tuple(
        gp.ZarrSource(
            sample,
            {
                raw: 'raw',
                labels: 'labels/mask',
            },
            {
                raw: gp.ArraySpec(interpolatable=True),
                labels: gp.ArraySpec(interpolatable=False)
            }) +
        gp.Normalize(raw, factor=1 / 255) +
        gp.Pad(raw, size=None) +
        gp.Pad(labels, size=None)
        for sample in samples
    )

    # Build the pipeline
    pipeline = sources
    pipeline += gp.RandomProvider()
    pipeline += gp.SimpleAugment()
    pipeline += gp.ElasticAugment(control_point_spacing=(60, 60, 60), jitter_sigma=(2, 2, 2), rotation_interval=(0, math.pi/4))
    pipeline += gp.IntensityAugment(raw, scale_min=0.9, scale_max=1.1, shift_min=-0.1, shift_max=0.1)
    pipeline += gp.GrowBoundary(labels, steps=1)
    pipeline += gp.DownSample(raw, downsample_factor, raw_res2)
    pipeline += gp.DownSample(labels, downsample_factor, labels_res2)
    pipeline += gp.AddAffinities(affinity_neighborhood=neighborhood, labels=labels, affinities=gt_affs_res1, dtype=np.float32)
    pipeline += gp.BalanceLabels(gt_affs_res1, affs_weights_res1)
    pipeline += gp.AddAffinities(affinity_neighborhood=neighborhood, labels=labels_res2, affinities=gt_affs_res2, dtype=np.float32)
    pipeline += gp.BalanceLabels(gt_affs_res2, affs_weights_res2)
    pipeline += gp.Unsqueeze([raw, raw_res2])
    pipeline += gp.Stack(batch_size)
    pipeline += Train(model, loss, optimizer, checkpoint_basename=os.path.join(dir, 'checkpoints', checkpoint_name),
                      save_every=save_every, log_dir=os.path.join(dir, 'logs'), log_every=save_every,
                      inputs={'input_res1': raw, 'input_res2': raw_res2},
                      outputs={0: pred_affs_res1, 1: pred_affs_res2},
                      array_specs={pred_affs_res1: gp.ArraySpec(voxel_size=voxel_size), pred_affs_res2: gp.ArraySpec(voxel_size=voxel_size_res2)},
                      loss_inputs={0: pred_affs_res1, 1: gt_affs_res1, 2: affs_weights_res1, 3: pred_affs_res2, 4: gt_affs_res2, 5: affs_weights_res2})

    with gp.build(pipeline):
        summary_writer = SummaryWriter(log_dir=os.path.join(dir, 'logs'))
        for _ in range(max_iteration - trained_until):
            batch = pipeline.request_batch(request)
        summary_writer.close()

# Start training
train(
    checkpoint_name='model_vs4_164by164by164_mtResAffs_GrowBoundary_samples012_round1',
    dir='./trainingFiles3d/mlResAffs-withGrowBoundary',
    max_iteration=20000,
    trained_until=0,
    save_every=100,
    latest_checkpoint=None
)
