# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets
from datasets import Dataset
import pandas as pd

from llama_recipes.datasets.utils import Concatenator

def get_preprocessed_samsum(dataset_config, tokenizer, split):

    if split == "train":
        train_df= pd.read_csv("/sensei-fs/users/mihirp/project_summary/data/train.csv")
        dataset = Dataset.from_pandas(train_df)
    else:
        test_df= pd.read_csv("/sensei-fs/users/mihirp/project_summary/data/test.csv")
        dataset = Dataset.from_pandas(test_df)

    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n{{summary}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                dialog=sample["document"],
                summary=sample["model_summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
        
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset
