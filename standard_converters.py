import fiftyone as fo
import datasets
import os
import PIL

def _get_extension(image):
    if isinstance(image, PIL.PngImagePlugin.PngImageFile):
        return ".png"
    elif isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        return ".jpg"
    else:
        return 'web'
    

def _get_download_dir(hf_name_or_path, *hf_args):
    download_dir = os.path.join(
        fo.config.default_dataset_dir, "huggingface", "hub", hf_name_or_path
        )
    if len(hf_args) > 0:
        download_dir = os.path.join(download_dir, *hf_args)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    return download_dir

def _save_image(image, i, download_dir, split_name):
    ext = _get_extension(image)
    if ext == 'web':
        return None
    fp = os.path.join(download_dir, f'{split_name}_{i}{ext}')
    if not os.path.exists(fp):
        image.save(fp)
    return fp

def _add_split_or_splits(dataset, hf_dataset, split, download_dir, _add_split):
    if isinstance(hf_dataset, datasets.dataset_dict.DatasetDict):
        for split_name, split in hf_dataset.items():
            print("Adding split: ", split_name)
            _add_split(dataset, split, split_name, download_dir)
    else:
        _add_split(dataset, hf_dataset, split, download_dir)


def _is_bounding_box_field(feature):
    return isinstance(feature, datasets.Sequence) and feature.length == 4

def _get_bounding_box_field_name(hf_dataset, detection_field_name):
    for column_name, feature in hf_dataset.features[detection_field_name].feature.items():
        if _is_bounding_box_field(feature):
            return column_name
        
def _has_bounding_box(feature):
    if not hasattr(feature, "feature"):
        return False
    return any([_is_bounding_box_field(subfeature) for subfeature in feature.feature.values()])

def _has_class_label(feature):
    if not hasattr(feature, "feature"):
        return False
    return any([isinstance(subfeature, datasets.ClassLabel) for subfeature in feature.feature.values()])

def _is_class_label_field(feature):
    return isinstance(feature, datasets.ClassLabel)

def _get_detection_label_field_name(hf_dataset, detection_field_name):
    for column_name, feature in hf_dataset.features[detection_field_name].feature.items():
        if _is_class_label_field(feature):
            return column_name

def _convert_bounding_box(hf_bbox, img_size):
    x, y, w, h = hf_bbox
    if all([0 <= c <= 1 for c in [x, y, w, h]]):
        return hf_bbox
    else:
        return [x/img_size[0], y/img_size[1], w/img_size[0], h/img_size[1]]
    
def _is_detection_field(feature):
    if not isinstance(feature, datasets.Sequence):
        return False
    return _has_class_label(feature) and _has_bounding_box(feature)

def _get_detection_field_name(hf_dataset):
    for column_name, feature in hf_dataset.features.items():
        if _is_detection_field(feature):
            return column_name

def _convert_to_detections(hf_feature_dict, bbox_field_name, label_field, class_map, img_size):
    keys = list(hf_feature_dict.keys())
    
    detections = []
    num_detects = len(hf_feature_dict[bbox_field_name])
    for i in range(num_detects):
        detection_dict = {}
        for key in keys:
            if key == bbox_field_name:
                detection_dict['bounding_box'] = _convert_bounding_box(hf_feature_dict[key][i], img_size)
            elif key == label_field:
                detection_dict['label'] = class_map[hf_feature_dict[key][i]]
            elif key == "id":
                detection_dict['hf_id'] = hf_feature_dict[key][i]
            else:
                detection_dict[key] = hf_feature_dict[key][i]
        detection = fo.Detection(**detection_dict)
        detections.append(detection)
    return fo.Detections(detections=detections)

def _get_classification_field_name(hf_dataset):
    for column_name, feature in hf_dataset.features.items():
        if isinstance(feature, datasets.features.features.ClassLabel):
            return column_name
        
def _get_image_field_name(hf_dataset):
    for field_name, feature in hf_dataset.features.items():
        if isinstance(feature, datasets.features.image.Image):
            return field_name
        
def _add_classification_data_split(dataset, split, split_name, download_dir):
    label_field = _get_classification_field_name(split)
    image_field = _get_image_field_name(split)

    label_classes = split.features[label_field].names

    samples = []
    for i, item in enumerate(split):
        image = item[image_field]
        fp = _save_image(image, i, download_dir, split_name)
        if fp is None:
            continue

        label = fo.Classification(label=label_classes[item[label_field]])
        sample_dict = {'filepath': fp, label_field: label}
        if split_name is not None:
            sample_dict["tags"] = [split_name]
        sample = fo.Sample(**sample_dict)
        samples.append(sample)
    dataset.add_samples(samples)


def _add_detection_data_split(dataset, split, split_name, download_dir):
    detections_field = _get_detection_field_name(split)
    label_field = _get_detection_label_field_name(split, detections_field)
    bbox_field = _get_bounding_box_field_name(split, detections_field)

    image_field = _get_image_field_name(split)
    label_classes = split.features[detections_field].feature[label_field].names

    samples = []
    for i, item in enumerate(split):
        image = item[image_field]
        fp = _save_image(image, i, download_dir, split_name)
        if fp is None:
            continue

        detections = _convert_to_detections(item[detections_field], bbox_field, label_field, label_classes, image.size)
        sample_dict = {'filepath': fp, detections_field: detections}
        if split_name is not None:
            sample_dict["tags"] = [split_name]
        sample = fo.Sample(**sample_dict)
        samples.append(sample)
    dataset.add_samples(samples)


def _load_hf_dataset_in_fiftyone(
    task,
    hf_name_or_path,
    *hf_args,
    persistent=False,
    name=None,
    split=None,
):
    hf_kwargs = {}
    if split is not None:
        hf_kwargs["split"] = split
    hf_dataset = datasets.load_dataset(hf_name_or_path, *hf_args, **hf_kwargs)
    if name is None:
        name = hf_name_or_path

    download_dir = _get_download_dir(hf_name_or_path, *hf_args)

    dataset = fo.Dataset(name=name, persistent=persistent, overwrite=True)

    if task == "classification":
        _add_split_or_splits(
            dataset, hf_dataset, split, download_dir, _add_classification_data_split
        )
    elif task == "detection":
        _add_split_or_splits(
            dataset, hf_dataset, split, download_dir, _add_detection_data_split
        )

    return dataset

def load_hf_classification_dataset_in_fiftyone(
        hf_name_or_path, 
        *hf_args,
        persistent=False, 
        name=None, 
        split=None,
        ):
    
    """Loads a Hugging Face detection dataset into FiftyOne.

    Args:
        hf_name_or_path: the name or path of the Hugging Face dataset
        persistent: whether the dataset should be persistent
        name: the name of the FiftyOne dataset to create
        split: the split to load
    """
    hf_kwargs = {}
    if split is not None:
        hf_kwargs['split'] = split
    hf_dataset = datasets.load_dataset(hf_name_or_path, *hf_args, **hf_kwargs)
    if name is None:
        name = hf_name_or_path

    download_dir = _get_download_dir(hf_name_or_path, *hf_args)

    dataset = fo.Dataset(name=name, persistent=persistent, overwrite=True)

    _add_split_or_splits(dataset, hf_dataset, split, download_dir, _add_classification_data_split)

    return dataset



def load_hf_classification_dataset_in_fiftyone(
    hf_name_or_path,
    *hf_args,
    persistent=False,
    name=None,
    split=None,
):
    """Loads a Hugging Face detection dataset into FiftyOne.

    Args:
        hf_name_or_path: the name or path of the Hugging Face dataset
        persistent: whether the dataset should be persistent
        name: the name of the FiftyOne dataset to create
        split: the split to load
    """
    return _load_hf_dataset_in_fiftyone(
        "classification",
        hf_name_or_path,
        *hf_args,
        persistent=persistent,
        name=name,
        split=split,
    )


def load_hf_detection_dataset_in_fiftyone(
    hf_name_or_path,
    *hf_args,
    persistent=False,
    name=None,
    split=None,
):
    """Loads a Hugging Face detection dataset into FiftyOne.

    Args:
        hf_name_or_path: the name or path of the Hugging Face dataset
        persistent: whether the dataset should be persistent
        name: the name of the FiftyOne dataset to create
        split: the split to load
    """
    return _load_hf_dataset_in_fiftyone(
        "detection",
        hf_name_or_path,
        *hf_args,
        persistent=persistent,
        name=name,
        split=split,
    )