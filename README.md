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

# Formatting

`find . -name '*.py' -print0 | xargs -0 yapf -i`

# Train a model with default parameters:

`python ./PL_Support_Codes/fit.py`

## Train a model with multiple eval regions in validation set.

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
