net = dict(type='Detector', )

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet101',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)

heads = dict(type='CLRHead',
             num_priors=192,
             refine_layers=3,
             fc_hidden_dim=64,
             sample_points=36)

iou_loss_weight = 2.
cls_loss_weight = 2.
xyt_loss_weight = 0.2
seg_loss_weight = 1.0

work_dirs = "work_dirs/clr/r101_culane"

neck = dict(type='FPN',
            in_channels=[512, 1024, 2048],
            out_channels=64,
            num_outs=3,
            attention=False)

test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

epochs = 20 
batch_size = 12

optimizer = dict(type='AdamW', lr=0.3e-3)  # 3e-4 for batchsize 8
total_iter = (88880 // batch_size) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 3
save_ep = 10

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])
ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(height=img_h, width=img_w),
                 p=1.0),
            dict(name='HorizontalFlip', p=0.5),
            dict(name='ChannelShuffle', p=0.1),
            dict(name='RandomBrightnessContrast',
                 parameters=dict(brightness_limit=(-0.04, 0.04),
                                 contrast_limit=(-0.15, 0.15)),
                 p=0.6),
            dict(name='HueSaturationValue',
                 parameters=dict(hue_shift_limit=(-7, 7),
                                 sat_shift_limit=(-10, 10),
                                 val_shift_limit=0),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(blur_limit=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(blur_limit=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2),
                                 keep_ratio=True),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(height=img_h, width=img_w),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'lane_line', 'seg']),
]

val_process = [
    dict(type='GenerateLaneLine',
         transforms=[
             dict(name='Resize',
                  parameters=dict(height=img_h, width=img_w),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor', keys=['img']),
]

dataset_path = './data/CULane'
dataset_type = 'CULane'
dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='train',
    processes=train_process,
),
val=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
),
test=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
))

workers = 10
log_interval = 500
# seed = 0
num_classes = 4 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False
