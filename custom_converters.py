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
        return "web"


def _get_download_dir(hf_name_or_path, *hf_args):
    download_dir = os.path.join(
        fo.config.default_dataset_dir, "huggingface", "hub", hf_name_or_path
    )
    if len(hf_args) > 0:
        download_dir = os.path.join(download_dir, *hf_args)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    return download_dir




def load_pokemon_captions_dataset_in_fiftyone():
    name_or_path = "lambdalabs/pokemon-blip-captions"
    hf_dataset = datasets.load_dataset(name_or_path)
    dataset = fo.Dataset(name="pokemon-blip-captions", persistent=True, overwrite=True)

    splits = ["train"]
    download_dir = _get_download_dir(name_or_path)

    for split in splits:
        samples = []
        for i, item in enumerate(hf_dataset[split]):
            img = item["image"]
            ext = _get_extension(img)
            fp = os.path.join(download_dir, f"{split}_{i}{ext}")
            if not os.path.exists(fp):
                img.save(fp)

            sample_dict = {
                "filepath": fp,
                "tags": [split],
                "text": item["text"],
            }

            sample = fo.Sample(**sample_dict)
            samples.append(sample)

        dataset.add_samples(samples)

    return dataset



def load_newyorker_captions_dataset_in_fiftyone():
    name_or_path = "jmhessel/newyorker_caption_contest"
    args = ("explanation")

    hf_dataset = datasets.load_dataset(name_or_path, *args)
    dataset = fo.Dataset(name="newyorker_caption_contest", persistent=True, overwrite=True)

    splits = ["train", "validation", "test"]

    download_dir = _get_download_dir(name_or_path, *args)

    for split in splits:
        samples = []
        for i, item in enumerate(hf_dataset[split]):
            img = item['image']
            ext = _get_extension(img)
            fp = os.path.join(download_dir, f'{split}_{i}{ext}')
            if not os.path.exists(fp):
                img.save(fp)

            sample_dict = {
            "filepath": fp,
            "tags": [split],
            "contest_number": item['contest_number'],
            "image_location": item['image_location'],
            "image_description": item['image_description'],
            'image_uncanny_description': item['image_uncanny_description'],
            'entities': item['entities'],
            'caption_choices': item['caption_choices'],
            'from_description': item['from_description'],
            'label': item['label'],
            }

            sample = fo.Sample(**sample_dict)
            samples.append(sample)

        dataset.add_samples(samples)

    return dataset


def load_imagerewarddb_dataset_in_fiftyone():
    name_or_path = "THUDM/ImageRewardDB"
    hf_dataset = datasets.load_dataset(name_or_path)

    splits = ["train", "validation", "test"]

    dataset = fo.Dataset(name="ImageRewardDB", persistent=True, overwrite=True)
    download_dir = _get_download_dir(name_or_path)

    samples = []

    for split in splits:
        for i, item in enumerate(hf_dataset[split]):
            img = item['image']
            ext = _get_extension(img)
            fp = os.path.join(download_dir, f'{split}_{i}{ext}')
            if not os.path.exists(fp):
                img.save(fp)

            sample_dict = {
                "filepath": fp,
                "tags": [split],
                "prompt_id": item['prompt_id'],
                "prompt": item['prompt'],
                "classification": item['classification'],
                "image_amount_in_total": item['image_amount_in_total'],
                "rank": item['rank'],
                "overall_rating": item['overall_rating'],
                "image_text_alignment_rating": item['image_text_alignment_rating'],
                "fidelity_rating": item['fidelity_rating'],
            }

            sample = fo.Sample(**sample_dict)
            samples.append(sample)

    dataset.add_samples(samples)
    return dataset
