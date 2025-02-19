import os
from typing import List, Tuple, cast

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    VerificationMode,
)

USE_LOCAL_DATASET = os.environ.get("USE_LOCAL_DATASET", "1") == "1"


def add_metadata_column(dataset, column_name, value):
    def add_source(example):
        example[column_name] = value
        return example

    return dataset.map(add_source)


def load_train_set() -> DatasetDict:
    ds_path = "colpali_train_set"
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_dict = cast(DatasetDict, load_dataset(base_path + ds_path, num_proc=40))
    return ds_dict


def load_train_set_detailed() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_train_set_with_tabfquad() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = cast(Dataset, concatenate_datasets(ds_tot))
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docmatix_ir_negs() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(Dataset, load_dataset(base_path + "docmatix-ir", split="train"))
    # dataset = dataset.select(range(100500))

    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(
        Dataset, load_dataset(base_path + "Docmatix", "images", split="train")
    )

    return ds_dict, anchor_ds, "docmatix"


def load_wikiss() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "Tevatron/"
    dataset = cast(
        Dataset,
        load_dataset(base_path + "wiki-ss-nq", data_files="train.jsonl", split="train"),
    )
    # dataset = dataset.select(range(400500))
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    base_path = "./data_dir/" if USE_LOCAL_DATASET else "HuggingFaceM4/"
    anchor_ds = cast(Dataset, load_dataset(base_path + "wiki-ss-corpus", split="train"))

    return ds_dict, anchor_ds, "wikiss"


def load_train_set_ir_negs() -> Tuple[DatasetDict, Dataset, str]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "manu/"
    dataset = cast(Dataset, load_dataset(base_path + "colpali-queries", split="train"))

    print("Dataset size:", len(dataset))
    # filter out queries with "gold_in_top_100" == False
    dataset = dataset.filter(lambda x: x["gold_in_top_100"], num_proc=16)
    print("Dataset size after filtering:", len(dataset))

    # keep only top 20 negative passages
    dataset = dataset.map(lambda x: {"negative_passages": x["negative_passages"][:20]})

    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    anchor_ds = cast(Dataset, load_dataset(base_path + "colpali-corpus", split="train"))
    return ds_dict, anchor_ds, "vidore"


def load_train_set_with_docmatix() -> DatasetDict:
    ds_paths = [
        "infovqa_train",
        "docvqa_train",
        "arxivqa_train",
        "tatdqa_train",
        "tabfquad_train_subsampled",
        "syntheticDocQA_government_reports_train",
        "syntheticDocQA_healthcare_industry_train",
        "syntheticDocQA_artificial_intelligence_train",
        "syntheticDocQA_energy_train",
        "Docmatix_filtered_train",
    ]
    base_path = "./data_dir/" if USE_LOCAL_DATASET else "vidore/"
    ds_tot: List[Dataset] = []
    for path in ds_paths:
        cpath = base_path + path
        ds = cast(Dataset, load_dataset(cpath, split="train"))
        if "arxivqa" in path:
            # subsample 10k
            ds = ds.shuffle(42).select(range(10000))
        ds_tot.append(ds)

    dataset = concatenate_datasets(ds_tot)
    dataset = dataset.shuffle(seed=42)
    # split into train and test
    dataset_eval = dataset.select(range(500))
    dataset = dataset.select(range(500, len(dataset)))
    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})
    return ds_dict


def load_docvqa_dataset() -> DatasetDict:
    if USE_LOCAL_DATASET:
        dataset_doc = cast(
            Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="validation")
        )
        dataset_doc_eval = cast(
            Dataset, load_dataset("./data_dir/DocVQA", "DocVQA", split="test")
        )
        dataset_info = cast(
            Dataset,
            load_dataset("./data_dir/DocVQA", "InfographicVQA", split="validation"),
        )
        dataset_info_eval = cast(
            Dataset, load_dataset("./data_dir/DocVQA", "InfographicVQA", split="test")
        )
    else:
        dataset_doc = cast(
            Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
        )
        dataset_doc_eval = cast(
            Dataset, load_dataset("lmms-lab/DocVQA", "DocVQA", split="test")
        )
        dataset_info = cast(
            Dataset,
            load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="validation"),
        )
        dataset_info_eval = cast(
            Dataset, load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="test")
        )

    # concatenate the two datasets
    dataset = concatenate_datasets([dataset_doc, dataset_info])
    dataset_eval = concatenate_datasets([dataset_doc_eval, dataset_info_eval])
    # sample 100 from eval dataset
    dataset_eval = dataset_eval.shuffle(seed=42).select(range(200))

    # rename question as query
    dataset = dataset.rename_column("question", "query")
    dataset_eval = dataset_eval.rename_column("question", "query")

    # create new column image_filename that corresponds to ucsf_document_id if not None, else image_url
    dataset = dataset.map(
        lambda x: {
            "image_filename": (
                x["ucsf_document_id"]
                if x["ucsf_document_id"] is not None
                else x["image_url"]
            )
        }
    )
    dataset_eval = dataset_eval.map(
        lambda x: {
            "image_filename": (
                x["ucsf_document_id"]
                if x["ucsf_document_id"] is not None
                else x["image_url"]
            )
        }
    )

    ds_dict = DatasetDict({"train": dataset, "test": dataset_eval})

    return ds_dict


def load_mixed_multiL_train_set() -> DatasetDict:

    spanish_dataset = load_dataset(
        "llamaindex/vdr-multilingual-train", "es", split="train", num_proc=50
    ).select(range(58738))
    italian_dataset = load_dataset(
        "llamaindex/vdr-multilingual-train", "it", split="train", num_proc=50
    ).select(range(54942))
    german_dataset = load_dataset(
        "llamaindex/vdr-multilingual-train", "de", split="train", num_proc=50
    ).select(range(58217))
    french_dataset = load_dataset(
        "llamaindex/vdr-multilingual-train", "fr", split="train", num_proc=50
    ).select(range(55270))
    multi_ling_tr = concatenate_datasets(
        [spanish_dataset, italian_dataset, german_dataset, french_dataset]
    )
    multi_ling_tr = multi_ling_tr.select_columns(["image", "query"])

    shard_list = [f"data/train-{i:05d}-of-00326.parquet" for i in range(35)]
    visrag_synth_orig = cast(
        DatasetDict,
        load_dataset(
            "openbmb/VisRAG-Ret-Train-Synthetic-data",
            data_files=shard_list,
            split="train",
            verification_mode=VerificationMode.NO_CHECKS,
            num_proc=50,
        ),
    )
    visrag_synth = visrag_synth_orig.select_columns(["image", "query"])
    visrag_synth_eval = visrag_synth.select(range(250))
    visrag_synth_tr = visrag_synth.select(range(250, 25250))

    visrag_vqa_orig = cast(
        DatasetDict,
        load_dataset(
            "openbmb/VisRAG-Ret-Train-In-domain-data", split="train", num_proc=50
        ),
    )
    visrag_vqa = visrag_vqa_orig.select_columns(["image", "query"])
    visrag_vqa_eval = visrag_vqa.select(range(250))
    visrag_vqa_tr = visrag_vqa.select(range(250, len(visrag_vqa)))

    docmatix_orig = cast(
        DatasetDict,
        load_dataset(
            "Metric-AI/rag_docmatix_100k",
            data_files="data/train-*.parquet",
            split="train",
            num_proc=50,
        ),
    )
    docmatix = docmatix_orig.select_columns(["image", "query"])
    docmatix_eval = docmatix.select(range(250))
    docmatix_tr = docmatix.select(range(250, 25250))

    colpali_orig = cast(
        DatasetDict,
        load_dataset("vidore/colpali_train_set", split="train", num_proc=50),
    )
    colpali = colpali_orig.select_columns(["image", "query"])
    colpali_eval = colpali.select(range(250))
    colpali_tr = colpali.select(range(250, len(colpali)))

    english_dataset = load_dataset(
        "llamaindex/vdr-multilingual-train", "en", split="train", num_proc=50
    ).select(range(53512))
    english = english_dataset.select_columns(["image", "query"])
    english_eval = english.select(range(250))
    english_tr = english.select(range(250, len(english)))

    tabfquad_dataset = load_dataset(
        "Metric-AI/tabfquad_train_set", split="train", num_proc=50
    )
    tabfquad = tabfquad_dataset.select_columns(["image", "query"])

    # french_dataset = load_dataset("llamaindex/vdr-multilingual-train", "fr", split="train",num_proc=15).select(range(55270))
    # french = french_dataset.select_columns(['image','query'])
    # french_eval = french.select(range(250))
    # french_tr = french.select(range(250, len(french)))

    train_set = concatenate_datasets(
        [visrag_synth_tr, visrag_vqa_tr, docmatix_tr, colpali_tr, english_tr, tabfquad]
    ).shuffle(seed=42)
    full_train_set = concatenate_datasets(
        [
            multi_ling_tr,
            train_set,
            train_set.shuffle(seed=35),
            train_set.shuffle(seed=28),
            train_set.shuffle(seed=21),
            train_set.shuffle(seed=14),
            train_set.shuffle(seed=7),
            train_set.shuffle(seed=49),
            train_set.shuffle(seed=56),
            train_set.shuffle(seed=63),
            train_set.shuffle(seed=70),
        ]
    )

    test_set = concatenate_datasets(
        [visrag_synth_eval, visrag_vqa_eval, docmatix_eval, colpali_eval, english_eval]
    ).shuffle(seed=42)

    ds_dict = DatasetDict({"train": full_train_set, "test": test_set})

    return ds_dict


def load_test_sets():
    energy = load_dataset("vidore/syntheticDocQA_energy_test", num_proc=40)
    healthcare = load_dataset(
        "vidore/syntheticDocQA_healthcare_industry_test", num_proc=40
    )
    artificial_intelligence = load_dataset(
        "vidore/syntheticDocQA_artificial_intelligence_test", num_proc=40
    )
    gov_reports = load_dataset(
        "vidore/syntheticDocQA_government_reports_test", num_proc=40
    )
    infovqa = load_dataset("vidore/infovqa_test_subsampled", num_proc=40)
    docvqa = load_dataset("vidore/docvqa_test_subsampled", num_proc=40)
    arxiv = load_dataset("vidore/arxivqa_test_subsampled", num_proc=40)
    tabfquad = load_dataset("vidore/tabfquad_test_subsampled", num_proc=40)
    tatdqa = load_dataset("vidore/tatdqa_test", num_proc=40)
    shift_project = load_dataset("vidore/shiftproject_test", num_proc=40)


class TestSetFactory:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def __call__(self, *args, **kwargs):
        dataset = load_dataset(self.dataset_path, split="test", num_proc=40)
        return dataset


if __name__ == "__main__":
    ds = TestSetFactory("vidore/tabfquad_test_subsampled")()
    print(ds)
    # load_mixed_multiL_train_set()
    # load_test_sets()
