# Visualize Hugging Face Datasets in FiftyOne

This repo contains a few simple scripts for loading Hugging Face datasets into 
FiftyOne for visualization, model application, and analysis.

## Installation

To install the required packages, run:

```bash
pip install -U datasets fiftyone
```

Then clone the repo:

```bash
git clone https://github.com/jacobmarks/huggingface-fiftyone-converters.git
cd export_hf_datasets_to_fiftyone
```

## Usage

The standard (basic) converters for Image Classification and Object Detection
datasets are in `standard_converters.py`. To load a basic classification dataset,
run:

```python

import fiftyone as fo

from standard_converters import load_hf_classification_dataset_in_fiftyone

# Load the dataset
dataset = load_hf_classification_dataset_in_fiftyone("mnist")

# Launch the FiftyOne App
session = fo.launch_app(dataset)
```

For object detection datasets, run:

```python

import fiftyone as fo

from standard_converters import load_hf_object_detection_dataset_in_fiftyone

# Load the dataset
dataset = load_hf_object_detection_dataset_in_fiftyone("detection-datasets/coco")

# Launch the FiftyOne App
session = fo.launch_app(dataset)
```

These loaders support any dataset that can be loaded with `datasets.load_dataset()`.
Additionally, out of the box they support loading specific splits and subsets.
If the Hugging Face dataset has subsets, specified by a second positional argument
to `datasets.load_dataset()`, you can load a specific subset by passing the subset
name as the second argument to the loader function. For example:

```python

import fiftyone as fo

from standard_converters import load_hf_classification_dataset_in_fiftyone

# Load the dataset
dataset = load_hf_classification_dataset_in_fiftyone("svhn", "full")
```

To only load a specific split, pass the split name via the `split` argument:

```python

import fiftyone as fo

from standard_converters import load_hf_classification_dataset_in_fiftyone

# Load the dataset

dataset = load_hf_classification_dataset_in_fiftyone("mnist", split="test")
```

## Supported Datasets

The following datasets are supported out of the box:

### Image Classification

- `mnist`
- `cifar10`
- `cifar100`
- `fashion_mnist`
- `beans`
- `nelorth/oxford-flowers`
- `cats_vs_dogs`
- `imagenet-1k`
- `frgfm/imagenette`
- `zh-plus/tiny-imagenet`
- `food101`
- `timm/imagenet-1k-wds`

### Object Detection

- `detection-datasets/coco`
- `cppe-5`
- `keremberke/aerial-sheep-object-detection`
- `keremberke/plane-detection`
- `Francesco/grass-weeds`
- `keremberke/license-plate-object-detection`



## Custom Loaders

For more complex datasets, you can write your own loaders. To illustrate this
process, we've included a few examples in `custom_converters.py`.

```python

from custom_converters import *

pokemon = load_pokemon_captions_dataset_in_fiftyone()

newyorker = load_newyorker_captions_dataset_in_fiftyone()

### 👇 you may want to sanitize this before viewing!!! 👇
imagerewarddb = load_imagerewarddb_dataset_in_fiftyone() 
```
