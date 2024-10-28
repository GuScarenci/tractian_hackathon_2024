MODEL_NAME = "facebook/detr-resnet-50"  # or "facebook/detr-resnet-50"
IMAGE_SIZE = 480


from datasets import load_dataset

cppe5 = load_dataset("cppe-5")

if "validation" not in cppe5:
    split = cppe5["train"].train_test_split(0.15, seed=1337)
    cppe5["train"] = split["train"]
    cppe5["validation"] = split["test"]

import numpy as np
import os
from PIL import Image, ImageDraw

image = cppe5["train"][2]["image"]
annotations = cppe5["train"][2]["objects"]
draw = ImageDraw.Draw(image)

categories = cppe5["train"].features["objects"].feature["category"].names

id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

for i in range(len(annotations["id"])):
    box = annotations["bbox"][i]
    class_idx = annotations["category"][i]
    x, y, w, h = tuple(box)
    # Check if coordinates are normalized or not
    if max(box) > 1.0:
        # Coordinates are un-normalized, no need to re-scale them
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
    else:
        # Coordinates are normalized, re-scale them
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + w) * w)
        y2 = int((y + h) * h)
    draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
    draw.text((x, y), id2label[class_idx], fill="white")

from transformers import AutoImageProcessor

MAX_SIZE = IMAGE_SIZE

image_processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    do_resize=True,
    size={"height": MAX_SIZE, "width": MAX_SIZE},
    do_pad=True,
    pad_size={"height": MAX_SIZE, "width": MAX_SIZE},
)

import albumentations as A

train_augment_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], min_area=25),
)

validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"]),
)


def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


from functools import partial

# Make transform functions for batch and apply for dataset splits
train_transform_batch = partial(
    augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
)
validation_transform_batch = partial(
    augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
)

images = [
{
  "id": 1,
  "width": 700,
  "height": 700,
  "file_name": "12095087_2.jpeg",
  "license": 0,
  "date_captured": ""
},
{
  "id": 2,
  "width": 600,
  "height": 600,
  "file_name": "61f2f04142c9f.jpg",
  "license": 0,
  "date_captured": ""
},
{
  "id": 3,
  "width": 700,
  "height": 700,
  "file_name": "img_6134e_1_1_2.jpg",
  "license": 0,
  "date_captured": ""
}
]

images_out = []

for i in images:
    images_out.append(Image.open("../l/" + i["file_name"]))

annotations = [
{
  "segmentation": [
    [
      278,
      330,
      307,
      330,
      307,
      362,
      278,
      362
    ]
  ],
  "area": 928,
  "bbox": [
    278,
    330,
    29,
    32
  ],
  "iscrowd": 0,
  "id": 1,
  "image_id": 1,
  "category_id": 2
},
{
  "segmentation": [
    [
      453,
      332,
      480,
      332,
      480,
      362,
      453,
      362
    ]
  ],
  "area": 810,
  "bbox": [
    453,
    332,
    27,
    30
  ],
  "iscrowd": 0,
  "id": 2,
  "image_id": 1,
  "category_id": 2
},
{
  "segmentation": [
    [
      450,
      491,
      478,
      491,
      478,
      520,
      450,
      520
    ]
  ],
  "area": 812,
  "bbox": [
    450,
    491,
    28,
    29
  ],
  "iscrowd": 0,
  "id": 3,
  "image_id": 1,
  "category_id": 2
},
{
  "segmentation": [
    [
      279,
      492,
      312,
      492,
      312,
      521,
      279,
      521
    ]
  ],
  "area": 957,
  "bbox": [
    279,
    492,
    33,
    29
  ],
  "iscrowd": 0,
  "id": 4,
  "image_id": 1,
  "category_id": 2
},
{
  "segmentation": [
    [
      519,
      375,
      545,
      375,
      545,
      406,
      519,
      406
    ]
  ],
  "area": 806,
  "bbox": [
    519,
    375,
    26,
    31
  ],
  "iscrowd": 0,
  "id": 5,
  "image_id": 1,
  "category_id": 2
},
{
  "segmentation": [
    [
      177,
      387,
      204,
      387,
      204,
      418,
      177,
      418
    ]
  ],
  "area": 837,
  "bbox": [
    177,
    387,
    27,
    31
  ],
  "iscrowd": 0,
  "id": 6,
  "image_id": 1,
  "category_id": 2
},
{
  "segmentation": [
    [
      182,
      179,
      207,
      179,
      207,
      201,
      182,
      201
    ]
  ],
  "area": 550,
  "bbox": [
    182,
    179,
    25,
    22
  ],
  "iscrowd": 0,
  "id": 7,
  "image_id": 1,
  "category_id": 2
},
{
  "segmentation": [
    [
      311,
      172,
      455,
      172,
      455,
      242,
      311,
      242
    ]
  ],
  "area": 10080,
  "bbox": [
    311,
    172,
    144,
    70
  ],
  "iscrowd": 0,
  "id": 8,
  "image_id": 1,
  "category_id": 4
},
{
  "segmentation": [
    [
      511,
      186,
      538,
      186,
      538,
      223,
      511,
      223
    ]
  ],
  "area": 999,
  "bbox": [
    511,
    186,
    27,
    37
  ],
  "iscrowd": 0,
  "id": 9,
  "image_id": 1,
  "category_id": 2
},
{
  "segmentation": [
    [
      557,
      289,
      646,
      289,
      646,
      362,
      557,
      362
    ]
  ],
  "area": 6497,
  "bbox": [
    557,
    289,
    89,
    73
  ],
  "iscrowd": 0,
  "id": 10,
  "image_id": 1,
  "category_id": 3
},
{
  "segmentation": [
    [
      60,
      146,
      654,
      146,
      654,
      547,
      60,
      547
    ]
  ],
  "area": 238194,
  "bbox": [
    60,
    146,
    594,
    401
  ],
  "iscrowd": 0,
  "id": 11,
  "image_id": 1,
  "category_id": 1
},
{
  "segmentation": [
    [
      224,
      277,
      250,
      277,
      250,
      302,
      224,
      302
    ]
  ],
  "area": 650,
  "bbox": [
    224,
    277,
    26,
    25
  ],
  "iscrowd": 0,
  "id": 12,
  "image_id": 2,
  "category_id": 2
},
{
  "segmentation": [
    [
      138,
      235,
      158,
      235,
      158,
      254,
      138,
      254
    ]
  ],
  "area": 380,
  "bbox": [
    138,
    235,
    20,
    19
  ],
  "iscrowd": 0,
  "id": 13,
  "image_id": 2,
  "category_id": 2
},
{
  "segmentation": [
    [
      105,
      358,
      130,
      358,
      130,
      378,
      105,
      378
    ]
  ],
  "area": 500,
  "bbox": [
    105,
    358,
    25,
    20
  ],
  "iscrowd": 0,
  "id": 14,
  "image_id": 2,
  "category_id": 2
},
{
  "segmentation": [
    [
      193,
      414,
      215,
      414,
      215,
      440,
      193,
      440
    ]
  ],
  "area": 572,
  "bbox": [
    193,
    414,
    22,
    26
  ],
  "iscrowd": 0,
  "id": 15,
  "image_id": 2,
  "category_id": 2
},
{
  "segmentation": [
    [
      314,
      352,
      341,
      352,
      341,
      378,
      314,
      378
    ]
  ],
  "area": 702,
  "bbox": [
    314,
    352,
    27,
    26
  ],
  "iscrowd": 0,
  "id": 16,
  "image_id": 2,
  "category_id": 2
},
{
  "segmentation": [
    [
      414,
      197,
      438,
      197,
      438,
      222,
      414,
      222
    ]
  ],
  "area": 600,
  "bbox": [
    414,
    197,
    24,
    25
  ],
  "iscrowd": 0,
  "id": 17,
  "image_id": 2,
  "category_id": 2
},
{
  "segmentation": [
    [
      242,
      135,
      387,
      135,
      387,
      211,
      242,
      211
    ]
  ],
  "area": 11020,
  "bbox": [
    242,
    135,
    145,
    76
  ],
  "iscrowd": 0,
  "id": 18,
  "image_id": 2,
  "category_id": 4
},
{
  "segmentation": [
    [
      217,
      118,
      241,
      118,
      241,
      134,
      217,
      134
    ]
  ],
  "area": 384,
  "bbox": [
    217,
    118,
    24,
    16
  ],
  "iscrowd": 0,
  "id": 19,
  "image_id": 2,
  "category_id": 2
},
{
  "segmentation": [
    [
      409,
      298,
      537,
      298,
      537,
      400,
      409,
      400
    ]
  ],
  "area": 13056,
  "bbox": [
    409,
    298,
    128,
    102
  ],
  "iscrowd": 0,
  "id": 20,
  "image_id": 2,
  "category_id": 3
},
{
  "segmentation": [
    [
      65,
      89,
      575,
      89,
      575,
      474,
      65,
      474
    ]
  ],
  "area": 196350,
  "bbox": [
    65,
    89,
    510,
    385
  ],
  "iscrowd": 0,
  "id": 21,
  "image_id": 2,
  "category_id": 1
},
{
  "segmentation": [
    [
      270,
      231,
      318,
      231,
      318,
      277,
      270,
      277
    ]
  ],
  "area": 2208,
  "bbox": [
    270,
    231,
    48,
    46
  ],
  "iscrowd": 0,
  "id": 22,
  "image_id": 3,
  "category_id": 2
},
{
  "segmentation": [
    [
      92,
      225,
      106,
      225,
      106,
      241,
      92,
      241
    ]
  ],
  "area": 224,
  "bbox": [
    92,
    225,
    14,
    16
  ],
  "iscrowd": 0,
  "id": 23,
  "image_id": 3,
  "category_id": 2
},
{
  "segmentation": [
    [
      242,
      493,
      270,
      493,
      270,
      527,
      242,
      527
    ]
  ],
  "area": 952,
  "bbox": [
    242,
    493,
    28,
    34
  ],
  "iscrowd": 0,
  "id": 24,
  "image_id": 3,
  "category_id": 2
},
{
  "segmentation": [
    [
      53,
      458,
      83,
      458,
      83,
      487,
      53,
      487
    ]
  ],
  "area": 870,
  "bbox": [
    53,
    458,
    30,
    29
  ],
  "iscrowd": 0,
  "id": 25,
  "image_id": 3,
  "category_id": 2
},
{
  "segmentation": [
    [
      217,
      128,
      242,
      128,
      242,
      146,
      217,
      146
    ]
  ],
  "area": 450,
  "bbox": [
    217,
    128,
    25,
    18
  ],
  "iscrowd": 0,
  "id": 26,
  "image_id": 3,
  "category_id": 2
},
{
  "segmentation": [
    [
      206,
      147,
      440,
      147,
      440,
      208,
      206,
      208
    ]
  ],
  "area": 14274,
  "bbox": [
    206,
    147,
    234,
    61
  ],
  "iscrowd": 0,
  "id": 27,
  "image_id": 3,
  "category_id": 4
},
{
  "segmentation": [
    [
      507,
      160,
      530,
      160,
      530,
      189,
      507,
      189
    ]
  ],
  "area": 667,
  "bbox": [
    507,
    160,
    23,
    29
  ],
  "iscrowd": 0,
  "id": 28,
  "image_id": 3,
  "category_id": 2
},
{
  "segmentation": [
    [
      417,
      370,
      443,
      370,
      443,
      401,
      417,
      401
    ]
  ],
  "area": 806,
  "bbox": [
    417,
    370,
    26,
    31
  ],
  "iscrowd": 0,
  "id": 29,
  "image_id": 3,
  "category_id": 2
},
{
  "segmentation": [
    [
      542,
      308,
      654,
      308,
      654,
      378,
      542,
      378
    ]
  ],
  "area": 7840,
  "bbox": [
    542,
    308,
    112,
    70
  ],
  "iscrowd": 0,
  "id": 30,
  "image_id": 3,
  "category_id": 3
},
{
  "segmentation": [
    [
      527,
      490,
      550,
      490,
      550,
      512,
      527,
      512
    ]
  ],
  "area": 506,
  "bbox": [
    527,
    490,
    23,
    22
  ],
  "iscrowd": 0,
  "id": 31,
  "image_id": 3,
  "category_id": 2
},
{
  "segmentation": [
    [
      48,
      118,
      657,
      118,
      657,
      572,
      48,
      572
    ]
  ],
  "area": 276486,
  "bbox": [
    48,
    118,
    609,
    454
  ],
  "iscrowd": 0,
  "id": 32,
  "image_id": 3,
  "category_id": 1
}
]

annotations_2 = [{"image_id": 1, "annotations": []}, {"image_id": 2, "annotations": []}, {"image_id": 3, "annotations": []}]
for i in annotations:
    annotations_2[i["image_id"]-1]["annotations"].append(i)

# Apply the image processor transformations: resizing, rescaling, normalization
result = image_processor(images=images_out, annotations=annotations_2, return_tensors="pt")

if not return_pixel_mask:
    result.pop("pixel_mask", None)

cppe5["train"] = cppe5["train"].with_transform(train_transform_batch)
cppe5["validation"] = cppe5["validation"].with_transform(validation_transform_batch)
cppe5["test"] = cppe5["test"].with_transform(validation_transform_batch)


import torch

def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data

from transformers.image_transforms import center_to_corners_format

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes

import numpy as np
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

    return metrics


eval_compute_metrics_fn = partial(
    compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
)

from transformers import AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained(
    MODEL_NAME,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="detr_finetuned_maquinas",
    num_train_epochs=1,
    fp16=False,
    per_device_train_batch_size=8,
    dataloader_num_workers=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    weight_decay=1e-4,
    max_grad_norm=0.01,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    push_to_hub=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=cppe5["train"],
    eval_dataset=cppe5["validation"],
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)

trainer.train()

image = Image.open("lixo.jpeg")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

d = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )
    d.rectangle(box)

image.save("out.png")
