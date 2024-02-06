# PharyngeaLSeG

This project does 3D Multi Task Model.

## Getting Started

These instructions will help you to get Pytorch Model instance.

### Prerequisites

What things you need to install the software and how to install them:

- Python (version 3.x)
- Torch


### Examples

```python
# Get 3D Unet
from model.inception_resnet_v2.multi_task.multi_task_3d import InceptionResNetV2MultiTask3D
base_model = InceptionResNetV2MultiTask3D(input_shape=(1, 32, 512, 512),
                                         class_channel=2, seg_channels=2, validity_shape=(1, 8, 8, 8), 
                                         inject_class_channel=None,
                                         block_size=8, decode_init_channel=None,
                                         skip_connect=True, dropout_proba=0.05, norm="instance", act="relu6",
                                         class_act="softmax", seg_act="softmax", validity_act="sigmoid",
                                         get_seg=True, get_class=False, get_validity=False,
                                         use_class_head_simple=True, use_seg_pixelshuffle_only=False
                                         )
model_param_num = sum(p.numel() for p in base_model.parameters())
print(f"model_param_num = {model_param_num}")
