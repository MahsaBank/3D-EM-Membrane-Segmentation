
import gunpowder as gp
import numpy as np
import matplotlib.pyplot as plt
import zarr
import torch
from funlib.learn.torch.models import UNet, ConvPass


# Constants
downsample_factor = 2
voxel_size = gp.Coordinate((4, 4, 4))
voxel_size_res2 = voxel_size * downsample_factor
input_shape = gp.Coordinate((168, 168, 168)) #(128, 128)
output_shape = gp.Coordinate((128, 128, 128)) #(88, 88)
output_shape_res2 = gp.Coordinate((34, 34, 34)) #(24, 24)
input_size_res1 = input_shape * voxel_size
output_size_res1 = output_shape * voxel_size
input_size_res2 = input_size_res1 // downsample_factor
output_size_res2 = output_shape_res2 * voxel_size_res2


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


def predict(checkpoint, raw_file):
    # Define array keys
    raw, labels = gp.ArrayKey('RAW'), gp.ArrayKey('LABELS')
    raw_res2, labels_res2 = gp.ArrayKey('RAW_RES2'), gp.ArrayKey('LABELS_RES2')
    pred_affs_res1 = gp.ArrayKey('PRED_AFFS_RES1')
    pred_affs_res2 = gp.ArrayKey('PRED_AFFS_RES2')

    # Create a batch request
    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size_res1)
    scan_request.add(labels, input_size_res1)
    scan_request.add(raw_res2, input_size_res2)
    scan_request.add(labels_res2, input_size_res2)
    scan_request.add(pred_affs_res1, output_size_res1)
    scan_request.add(pred_affs_res2, output_size_res2)

    context = (input_size_res1 - output_size_res1) / 2

    source = gp.ZarrSource(
        raw_file,
        {
            raw: 'raw',
            labels: 'labels\mask'
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False)
        }
    )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = source.spec[raw].roi.grow(-context,-context)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = affsMlResUNet().to(device)

    model = torch.nn.DataParallel(model)

    model.eval()

    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs={
            'input_res1': raw,
            'input_res2': raw_res2,
        },
        array_specs={pred_affs_res1: gp.ArraySpec(voxel_size=voxel_size), pred_affs_res2: gp.ArraySpec(voxel_size=voxel_size_res2)},
        outputs={
            0: pred_affs_res1,
            1: pred_affs_res2,
        }
    )

    scan = gp.Scan(scan_request)

    pipeline = source

    pipeline += gp.Normalize(raw, factor=1/255)
    # raw shape = h,w

    pipeline += gp.DownSample(raw, downsample_factor, raw_res2)

    pipeline += gp.DownSample(labels, downsample_factor, labels_res2)

    pipeline += gp.Unsqueeze([raw, raw_res2])
    # raw shape = c,h,w

    pipeline += gp.Stack(1)
    # raw shape = b,c,h,w

    pipeline += predict

    pipeline += scan

    pipeline += gp.Squeeze([raw, raw_res2])
    # raw shape = c,h,w
    # pred_affs shape = b,c,h,w

    pipeline += gp.Squeeze([raw, labels, raw_res2, labels_res2, pred_affs_res1, pred_affs_res2])
    # raw shape = h,w
    # pred_affs shape = c,h,w

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request.add(labels, total_input_roi.get_end())
    predict_request.add(pred_affs_res1, total_output_roi.get_end())

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[raw].data, batch[labels].data, batch[pred_affs_res1].data

checkpoints = ['trainingFiles3d/mlResAffs-noGrowBoundary/checkpoints/model_vs4_164by164by164_mtResAffs_noGrowBoundary_checkpoint_8200']
raw_file = 'datasets/test_data_656by656by656_volume5.zarr'
slice_idx = [10, 50, 100, 400]
do_prediction = True
raw_name = raw_file.split('datasets/')[1]

for checkpoint in checkpoints: 
    model_idx = checkpoint.split('checkpoint_')[1] 
    round = checkpoint.split('_round')
    if len(round)>1:
         round_idx = round[1][0]
    else:
         round_idx = 1
    pred_data_filename = f'results/predict_mtResAffs_checkpoint{model_idx}_round{round_idx}_{raw_name}'
    
    name = pred_data_filename.split('.zarr')[0]
    save_filename = f'{name}.png'

    fig, axes = plt.subplots(
           len(slice_idx),
           3,
           figsize=(9, 3*len(slice_idx)),
           sharex=True,
           sharey=True,
           squeeze=False)
    
    if do_prediction:
          raw, labels, pred_affs = predict(checkpoint, raw_file)
          with zarr.open(pred_data_filename, mode='w') as fi:
               fi['raw'] = raw
               fi['labels'] = labels
               fi['pred_affs'] = pred_affs
    else:
          with zarr.open(pred_data_filename, mode='r') as fi:
               raw = fi['raw']
               labels = fi['labels']
               pred_affs = fi['pred_affs']

    for i, idx in enumerate(slice_idx):
          axes[i][0].imshow(raw[idx], cmap='gray')
          axes[i][1].imshow(labels[idx], cmap='jet')
          axes[i][2].imshow(np.squeeze(pred_affs[0][idx]), cmap='jet')
          axes[i][2].imshow(np.squeeze(pred_affs[1][idx]), cmap='jet', alpha=0.5)

    plt.savefig(save_filename)
