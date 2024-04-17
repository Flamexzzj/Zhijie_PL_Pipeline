import torch.nn as nn
import torch.optim as optim

import torch
import numpy as np
import torchmetrics
from einops import rearrange
import pytorch_lightning as pl

from PL_Support_Codes.models.unet import UNet_Orig
from PL_Support_Codes.models.unet import UNet_CBAM
# from PL_Support_Codes.models.unet import UNet
from PL_Support_Codes.tools import create_conf_matrix_pred_image


class WaterSegmentationModel(pl.LightningModule):

    def __init__(self,
                 in_channels,
                 n_classes,
                 lr,
                 log_image_iter=50,
                 to_rgb_fcn=None,
                 ignore_index=None,
                 model_used=None,
                 model_loss_fn_a=None,
                 model_loss_fn_b=None,
                 model_loss_fn_a_ratio=None,
                 model_loss_fn_b_ratio=None,
                 optimizer_name=None):
        super().__init__()
        self.lr = lr
        self.model_used = model_used
        self.model_loss_fn_a = model_loss_fn_a
        self.model_loss_fn_b = model_loss_fn_b
        self.model_loss_fn_a_ratio = model_loss_fn_a_ratio
        self.model_loss_fn_b_ratio = model_loss_fn_b_ratio
        self.optimizer_name = optimizer_name
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.ignore_index = ignore_index
        #TODO：
        self.any_validation_steps_executed = False
        self.any_test_steps_executed = False
        #TODO：

        # Build model.
        self._build_model()

        # Get metrics.
        if self.ignore_index == -1:
            self.ignore_index = self.n_classes - 1
        self.tracked_metrics = self._get_tracked_metrics()

        LOSS_FUNCS ={
            'cross_entropy': nn.CrossEntropyLoss
        }
        
        # Get loss function.
        self.loss_func_a = LOSS_FUNCS[self.model_loss_fn_a](ignore_index=self.ignore_index)
        self.loss_func_b = LOSS_FUNCS[self.model_loss_fn_b](ignore_index=self.ignore_index)

        # Log images hyperparamters.
        self.to_rgb_fcn = to_rgb_fcn
        self.log_image_iter = log_image_iter
        print("!!!!!!!!!!!!")
        print("!!!!!!!!!!!!")
        print("Model used: ",model_used)
        print("n_classes: ", n_classes)
        print(in_channels)
        print(ignore_index)
        print("optimizer_name: ",optimizer_name)
        print(lr)
        print("!!!!!!!!!!!!")
        print("!!!!!!!!!!!!")



    def _get_tracked_metrics(self, average_mode='micro'):
        metrics = torchmetrics.MetricCollection([
            torchmetrics.F1Score(task="multiclass",num_classes=self.n_classes,ignore_index=self.ignore_index,average='micro'),
            torchmetrics.JaccardIndex(task="multiclass",
                                      num_classes=self.n_classes,
                                      ignore_index=self.ignore_index,
                                      average='micro'),
            torchmetrics.Accuracy(task="multiclass",
                                  num_classes=self.n_classes,
                                  ignore_index=self.ignore_index,
                                  average='micro'),
        ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def _compute_metrics(self, conf, target):
        # conf: [batch_size, n_classes, height, width]
        # target: [batch_size, height, width]

        pred = conf.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), target.flatten()

        batch_metrics = {}
        for metric_name, metric_func in self.tracked_metrics.items():
            metric_value = metric_func(flat_pred, flat_target)
            metric_value = torch.nan_to_num(metric_value)
            batch_metrics[metric_name] = metric_value
        return batch_metrics

    def _build_model(self):
        # Build models.
        if type(self.in_channels) is dict:
            n_in_channels = 0
            for feature_channels in self.in_channels.values():
                n_in_channels += feature_channels
        MODELS_USED = {
            'unet_orig': UNet_Orig,
            'unet_cbam': UNet_CBAM
        }
        print("Model used!!!!!!!!!: ",MODELS_USED[self.model_used])
        self.model = MODELS_USED[self.model_used](n_in_channels, self.n_classes)

    def forward(self, batch):
        images = batch['image']
        output = self.model(images)
        return output

    def _set_model_to_train(self):
        self.model.train()

    def _set_model_to_eval(self):
        self.model.eval()

    def training_step(self, batch, batch_idx):
        self._set_model_to_train()
        images, target = batch['image'], batch['target']
        output = self.forward(batch)

        loss = self.model_loss_fn_a_ratio * self.loss_func_a(output, target) + self.model_loss_fn_b_ratio * self.loss_func_b(output, target)
        if torch.isnan(loss):
            # Happens when all numbers are ignore numbers.
            loss = torch.nan_to_num(loss)
        pred = output.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), batch['target'].flatten()
        metric_output = self.train_metrics(flat_pred, flat_target)
        self.log_dict(metric_output,
                      prog_bar=True,
                      on_step=True,
                      on_epoch=True)

        if False:
        # if (batch_idx % self.log_image_iter) == 0:
            # Unnormalize images
            images = (images * batch['std']) + batch['mean']
            for b in range(images.shape[0]):
                # Convert input image to RGB.
                input_image = images[b].detach().cpu().numpy()
                rgb_image = self.to_rgb_fcn(input_image)

                # Generate prediction image
                prediction = output[b].detach().argmax(dim=0).cpu().numpy()
                ground_truth = target[b].detach().cpu().numpy()
                cm_image = create_conf_matrix_pred_image(
                    prediction, ground_truth) / 255.0

                # Create title for logged image.
                # str_title = f'train_e{str(self.current_epoch).zfill(3)}_b{str(b).zfill(3)}.png'
                str_title = f'train_i{str(batch_idx).zfill(4)}_b{str(b).zfill(3)}.png'

                self.log_image_to_tensorflow(str_title, rgb_image, cm_image)

        return loss

    def validation_step(self, batch, batch_idx):
        self.any_validation_steps_executed = True
        self._set_model_to_eval()
        images, target = batch['image'], batch['target']
        output = self.forward(batch)

        loss = self.model_loss_fn_a_ratio * self.loss_func_a(output, target) + self.model_loss_fn_b_ratio * self.loss_func_b(output, target)
        if torch.isnan(loss):
            # Happens when all numbers are ignore numbers.
            loss = torch.nan_to_num(loss)

        pred = output.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), batch['target'].flatten()
        metric_output = self.valid_metrics(flat_pred, flat_target)
        self.valid_metrics.update(flat_pred, flat_target)

        # Log metrics and loss.
        metric_output['valid_loss'] = loss
        self.log_dict(metric_output,
                      prog_bar=True,
                      on_step=True,
                      on_epoch=True)

        if False:
        # if (batch_idx % self.log_image_iter) == 0:
            # Unnormalize images
            images = (images * batch['std']) + batch['mean']
            for b in range(images.shape[0]):
                # Convert input image to RGB.
                input_image = images[b].detach().cpu().numpy()
                rgb_image = self.to_rgb_fcn(input_image)

                # Generate prediction image
                prediction = output[b].detach().argmax(dim=0).cpu().numpy()
                ground_truth = target[b].detach().cpu().numpy()
                cm_image = create_conf_matrix_pred_image(
                    prediction, ground_truth) / 255.0

                # Create title for logged image.
                # str_title = f'valid_e{str(self.current_epoch).zfill(3)}_b{str(b).zfill(3)}.png'
                str_title = f'valid_i{str(batch_idx).zfill(4)}_b{str(b).zfill(3)}.png'

                self.log_image_to_tensorflow(str_title, rgb_image, cm_image)

    def test_step(self, batch, batch_idx):
        self.any_test_steps_executed = True
        self._set_model_to_eval()
        output = self.forward(batch)
        target = batch['target']

        loss = self.model_loss_fn_a_ratio * self.loss_func_a(output, target) + self.model_loss_fn_b_ratio * self.loss_func_b(output, target)

        # Track metrics.
        pred = output.argmax(dim=1)
        flat_pred, flat_target = pred.flatten(), batch['target'].flatten()
        self.test_metrics.update(flat_pred, flat_target)

        # Log metrics and loss.
        self.log_dict({'test_loss': loss},
                      prog_bar=True,
                      on_step=True,
                      on_epoch=True)

    def configure_optimizers(self):
        OPTIMIZERS = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'adamw': optim.AdamW,
            'adamax': optim.Adamax,
            'adadelta': optim.Adadelta,
            'adagrad': optim.Adagrad,
            'rmsprop': optim.RMSprop,
            'rprop': optim.Rprop,
            'asgd': optim.ASGD,
            'lbfgs': optim.LBFGS,
            'sparse_adam': optim.SparseAdam,
            'radam': optim.RAdam,
            'nadam': optim.NAdam,
        }
        optimizer = OPTIMIZERS[self.optimizer_name](self.parameters(), lr=self.lr)
        return optimizer
    def on_before_batch_transfer(self, batch, dataloader_idx=0):
    # Function to convert tensors to float32, leaving other data types unchanged
        def to_float32(item):
            if isinstance(item, torch.Tensor):
                return item.float()
            elif isinstance(item, (list, tuple)):
                return type(item)(to_float32(x) for x in item)
            elif isinstance(item, dict):
                return {key: to_float32(value) for key, value in item.items()}
            else:
                return item

        return to_float32(batch)

# # TODO:
#     def validation_epoch_end(self, validation_step_outputs):
#         if len(validation_step_outputs) == 0:
#             self.test_f1_score = 0
#             self.test_iou = 0
#             self.test_acc = 0
#         else:
#             metric_output = self.valid_metrics.compute()
#             self.log_dict(metric_output)
# # TODO:
    

#     def test_epoch_end(self, test_step_outputs) -> None:
#         if len(test_step_outputs) == 0:
#             pass
#         else:
#             metric_output = self.test_metrics.compute()
#             self.log_dict(metric_output)

#             self.f1_score = metric_output['test_F1Score'].item()
#             self.acc = metric_output['test_Accuracy'].item()
#             self.iou = metric_output['test_JaccardIndex'].item()

    def on_validation_epoch_end(self):
        if not self.any_validation_steps_executed:
            # Handle case where no validation steps were executed
            self.log("val_no_steps", True)  # Example of logging a custom flag or handling as needed
        else:
            # Compute and log metrics as usual
            metric_output = self.valid_metrics.compute()
            self.log_dict(metric_output)
            self.valid_metrics.reset()
    
    # Reset the tracker for the next epoch
        self.any_validation_steps_executed = False

    def on_test_epoch_end(self):
        if not self.any_test_steps_executed:
            # Handle case where no test steps were executed
            self.log("test_no_steps", True)  # Example of logging a custom flag or handling as needed
        else:
            # Compute and log metrics as usual
            metric_output = self.test_metrics.compute()
            self.log_dict(metric_output)
            self.f1_score = metric_output['test_F1Score'].item()
            self.acc = metric_output['test_Accuracy'].item()
            self.iou = metric_output['test_JaccardIndex'].item()
            self.test_metrics.reset()
        
        # Reset the tracker for the next usage
        self.any_test_steps_executed = False



    def log_image_to_tensorflow(self, str_title, rgb_image, cm_image):
        """_summary_

        Args:
            str_title (str): Title for the image.
            rgb_image (np.array): A np.array of shape [height, width, 3].
            cm_image (np.array): A np.array of shape [height, width, 3].
        """

        # Combine images together.
        log_image = np.concatenate((rgb_image, cm_image), axis=0).transpose(
            (2, 0, 1))
        self.logger.experiment.add_image(str_title, log_image,
                                         self.global_step)