from PL_Support_Codes.models.lf_model import LateFusionModel
from PL_Support_Codes.models.ef_model import EarlyFusionModel
from PL_Support_Codes.models.water_seg_model import WaterSegmentationModel

MODELS = {
    'ms_model': WaterSegmentationModel,
    'ef_model': EarlyFusionModel,
    'lf_model': LateFusionModel
}


def build_model(model_name, input_channels, n_classes, lr, log_image_iter,
                to_rgb_fcn, ignore_index, **kwargs):
    try:
        model = MODELS[model_name](input_channels, n_classes, lr,
                                   log_image_iter, to_rgb_fcn, ignore_index,
                                   **kwargs)
    except KeyError:
        print(f'Could not find model named: {model_name}')
    return model
