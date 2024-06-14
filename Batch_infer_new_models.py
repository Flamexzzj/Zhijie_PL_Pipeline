# %%
# This is MAC branch
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from scipy.special import softmax
from torch.utils.data import DataLoader

from PL_Support_Codes.models import build_model
from PL_Support_Codes.tools import load_cfg_file
from PL_Support_Codes.datasets.utils import generate_image_slice_object
from PL_Support_Codes.utils.utils_image import ImageStitcher_v2 as ImageStitcher
from PL_Support_Codes.datasets import build_dataset, tensors_and_lists_collate_fn

from PL_Support_Codes.models.lf_model import LateFusionModel
from PL_Support_Codes.models.ef_model import EarlyFusionModel
from PL_Support_Codes.models.water_seg_model import WaterSegmentationModel

# %%
# Setup model parameters
dataset_name = "batch_infer"
infer_split = "all"
infer_seed_num = 0
infer_train_split_pct = 0.0
infer_num_workers = 0
n_classes_model = 3
#need to chnage
model_used_here = "rexnet_unet" #unet_cbam, rexnet_unet
# optimizer_used = "adam"
model_loss_fn_a_infer = "cross_entropy"
model_loss_fn_b_infer = "cross_entropy"
model_loss_fn_a_infer_ratio = 1
model_loss_fn_b_infer_ratio = 0


base_save_dir = r"E:\Zhijie_PL_Pipeline\Infered_result\test_test_test"
checkpoint_path = r"E:\Zhijie_PL_Pipeline\Model_in_progress\2024-06-12\rexnet_csda_only\checkpoints\model-epoch=23-val_MulticlassJaccardIndex=0.8555.ckpt"
# checkpoint_path = r"E:\Zhijie_PL_Pipeline\Trained_model\Unet_PS_models\checkpoints\model-epoch=06-val_MulticlassJaccardIndex=0.8755.ckpt"

# Root folder containing the directories you waant to run inference on, under this folder, there should be different dates folder, within the dates folder, there should be imgs
ROOT_FOLDER = r"E:\Zhijie_PL_Pipeline\DATA\test\\"

# JSON file path
JSON_FILE = r"E:\Zhijie_PL_Pipeline\Zhijie_PL_Pipeline\dataset_dirs.json"

# %%
def infer_here():
    # Load configuration file.
    path_components = checkpoint_path.split(os.sep)
    experiment_dir = os.sep.join(path_components[:5])
    experiment_dir = os.path.join(experiment_dir, 'hydra')

    cfg_path = os.path.join(experiment_dir, 'config.yaml')
    print("check point file path: ", checkpoint_path)
    cfg = load_cfg_file(cfg_path)

    if 'model_n_classes' in cfg:
        n_classes_used = cfg.model_n_classes
    else:
        n_classes_used = n_classes_model
    
    if 'model_used' in cfg:
        model_used_infer = cfg.model_used
    else:
        model_used_infer = model_used_here


    


    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)
    print("Saving inference to: ",base_save_dir)
    # Load dataset.
    slice_params = generate_image_slice_object(cfg.crop_height, cfg.crop_width, min(cfg.crop_height, cfg.crop_width))
    eval_dataset = build_dataset(dataset_name,
                                 infer_split,
                                 slice_params,
                                 sensor=cfg.dataset_sensor,
                                 channels=cfg.dataset_channels,
                                 n_classes=n_classes_used,
                                 norm_mode=cfg.norm_mode,
                                 eval_region=cfg.eval_region,
                                 ignore_index=cfg.ignore_index,
                                 seed_num=infer_seed_num,
                                 train_split_pct=infer_train_split_pct,
                                 output_metadata=True,
                                 # ** allows us to pass in any additional arguments to the dataset as dictionary.
                                 **cfg.dataset_kwargs)

    eval_loader = DataLoader(eval_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=infer_num_workers, collate_fn=tensors_and_lists_collate_fn)
    
    MODELS = {
        'ms_model': WaterSegmentationModel,
        'ef_model': EarlyFusionModel,
        'lf_model': LateFusionModel
    }
# here need to retrain model and save new config file, in the new one it
#should be   cfg.model_used
    model = MODELS[cfg.model_name].load_from_checkpoint(checkpoint_path,
                                       in_channels=eval_dataset.n_channels,
                                       n_classes=eval_dataset.n_classes,
                                       lr=cfg.lr,
                                       log_image_iter=cfg.log_image_iter,
                                       to_rgb_fcn=eval_dataset.to_RGB,
                                       ignore_index=eval_dataset.ignore_index,
                                       model_used=model_used_infer,
                                       model_loss_fn_a = model_loss_fn_a_infer,
                                       model_loss_fn_b = model_loss_fn_b_infer,
                                       model_loss_fn_a_ratio = model_loss_fn_a_infer_ratio,
                                       model_loss_fn_b_ratio = model_loss_fn_b_infer_ratio,
                                       **cfg.model_kwargs)
    model._set_model_to_eval()

    # Get device.
    if torch.cuda.is_available():
        device = 'cuda'
        print("!!!!!! CUDA is available!!!!!!")
    else:
        device = 'mps'
        print("!!!!!! CUDA is not available, using MPS !!!!!!")
    model = model.to(device)

    # Generate predictions on target dataset.
    pred_canvases = {}
    with torch.no_grad(): #no_grad() prevents gradiant calculation, which is not needed for inference.
        # breakpoint()
        for batch in tqdm(eval_loader, colour='green', desc='Generating predictions'):
            # Move batch to device.
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(dtype=torch.float32).to(device)

            # Generate predictions.
            # this pass the current batch into model, to generate prediction, 
            #at this stage, the output is the raw output that is the probability distribution over the classes for the corresponding pixel in the input image. This distribution can be interpreted as 
            #the model's confidence in each class for that pixel. They are the raw score of what model thinks the possibility of each class for each pixel.
            output = model(batch).detach().cpu().numpy()
            # convert the each class probability distribution to the softmax probability distribution. meaning that the probabilities will add up to 1 between different classes
            preds = softmax(output, axis=1)

            input_images = batch['image'].detach().cpu().numpy()
            # rearrange the tensor to the format of (batch, height, width, channel)
            preds = rearrange(preds, 'b c h w -> b h w c')
            input_images = rearrange(input_images, 'b c h w -> b h w c')
            batch_mean = rearrange(batch['mean'], 'b c 1 1 -> b 1 1 c').detach().cpu().numpy()
            batch_std = rearrange(batch['std'], 'b c 1 1 -> b 1 1 c').detach().cpu().numpy()

            for b in range(output.shape[0]):# output.shape[0] is the batch size. so this code is iterating through each image in the batch

                pred = preds[b]
                metadata = batch['metadata'][b]
                input_image = input_images[b]
                region_name = metadata['region_name']

                # Check if image stitcher exists for this region.
                if region_name not in pred_canvases.keys():
                    # Get base save directories.
                    pred_save_dir = os.path.join(base_save_dir, region_name + '_pred')

                    # Initialize image stitchers.
                    pred_canvases[region_name] = ImageStitcher(pred_save_dir, save_backend='tifffile', save_ext='.tif')
                
                # Add input image and prediction to stitchers.
                unnorm_img = (input_image * batch_std[b]) + batch_mean[b]
                image_name = os.path.splitext(os.path.split(metadata['image_path'])[1])[0]
                pred_canvases[region_name].add_image(pred, image_name, metadata['crop_params'], metadata['crop_params'].og_height, metadata['crop_params'].og_width)

    # Convert stitched images to proper format.
    for region_name in pred_canvases.keys():
        # Combine images.
        pred_canvas = pred_canvases[region_name].get_combined_images()

        for image_name, image in pred_canvas.items():
            # Figure out the predicted class.
            pred = np.clip(image.argmax(axis=2), 0, 1)
            # save_path = os.path.join(pred_canvases[region_name].save_dir, image_name + '.tif')
            save_path_p1 = os.path.join(base_save_dir, region_name)
            if not os.path.exists(save_path_p1):
                os.makedirs(save_path_p1)
            save_path = os.path.join(save_path_p1, image_name + '.tif')
            print(f'Saving {save_path}')
            Image.fromarray((pred*255).astype('uint8')).save(save_path)

# %%
import os
import json
from tqdm import tqdm

counter = 1
# Loop through each sub-directory in the root folder
for dir in os.listdir(ROOT_FOLDER):
    full_dir_path = os.path.join(ROOT_FOLDER, dir)
    if os.path.isdir(full_dir_path):
        FOLDER_NAME = full_dir_path

        # Update the JSON file
        with open(JSON_FILE, 'r') as json_file:
            data = json.load(json_file)
            data['batch_infer'] = FOLDER_NAME

        with open(JSON_FILE, 'w') as json_file:
            json.dump(data, json_file)

        # Execute the command
        print("This is the ", counter, "th iteration, out of ", len(os.listdir(ROOT_FOLDER)))
        print("This is the ", counter, "th iteration, out of ", len(os.listdir(ROOT_FOLDER)))
        print("We are infering ", full_dir_path)
        print("We are infering ", full_dir_path)
        infer_here()
        


