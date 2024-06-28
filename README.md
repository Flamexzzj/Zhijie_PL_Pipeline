# Spatial+Temporal Water Segmentation

A repository containing experiments investigating the improvement of adding temporal rainfall data for water segmentation.

# Install

### Create conda environment from yml file. this installs pytorch, alongwith geospatial librarieslike rasterio and GDAL and also GEE:

`conda env create -f win_torch_ee_gdal.yml`

### Activate environment:

`conda activate geotorchee`

### Run setup.py:

`pip install -e ./`

### Run the following commands:

`pip install tensorboard`

`pip install tensorboardX`

`pip install timm`

`conda install -c fastai fastai`

# Setup dataset directories:

### On Minnow machine fill in dataset_dirs.json with:

```
{
  "thp": "/media/mule/Projects/NASA/CSDAP/Data/CombinedDataset_1122/THP/",
  "csdap": "/media/mule/Projects/NASA/CSDAP/Data/CombinedDataset_1122/CSDAP/",
  "combined": "/media/mule/Projects/NASA/CSDAP/Data/CombinedDataset_1122/",
  "s1floods": "/media/mule/Projects/NASA/CSDAP/Data/Public_Dataset/S1F11/"
}
```
Note: Use the 'combined' dataset to train models of different sensors. Use the "batch_infer" dataset for batch inference
# Formatting your code

`find . -name '*.py' -print0 | xargs -0 yapf -i`

# Steps of training a model

1. Adjust the file path of 'combined' in 'dataset_dirs.json'   
2. Go to 'combined.py' under 'PL_Support_Codes/datasets' to make sure the directory points to the dataset you want to train on  
3. Adjust Hyperparameters and settings in 'confid.yml' under 'PL_Support_Codes/conf'  
4. Run `python ./PL_Support_Codes/fit.py`  

## The pipeline also takes args in command line to change settings, for example:

`python ./PL_Support_Codes/fit.py 'eval_reigon=[region_name_1, region_name_2]'`

# Run inference with trained models:

`python Batch_infer_new_models.py`

# Visualize model training with Tensorboard

## Within VSCode

Mac: `SHIFT+CMD+P` <br />
Windows: `F1` <br />

Then search: <br />
`Python: Launch TensorBoard` <br />

Find path of experiment logs. <br />
`./outputs/<date>/<time>/tensorboard_logs/` <br />

## Through browser

`tensorboard --logdir <path_to_tensorboard_logs>`
