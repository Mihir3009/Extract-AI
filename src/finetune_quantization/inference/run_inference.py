# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional
from tqdm.auto import tqdm
import os
from os.path import join
import pandas as pd

import torch
import datasets
from datasets import load_dataset
import transformers
from transformers.pipelines.pt_utils import KeyDataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from peft.tuners.lora import LoraLayer
from peft import PeftModel, get_peft_config
from trl import SFTTrainer


########################################################################
# This is a fully working simple example to use trl's RewardTrainer.
#
# This example fine-tunes any causal language model (GPT-2, GPT-Neo, etc.)
# by using the RewardTrainer from trl, we will leverage PEFT library to finetune
# adapters on the model.
#
########################################################################


# Define and parse arguments.


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=8)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=2)
    num_train_epochs: Optional[int] = field(default=3)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=2048)
    save_strategy: Optional[str] = field(default="epoch")
    model_name: Optional[str] = field(
        default="tiiuae/falcon-7b",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    new_model: Optional[str] = field(
        default="/sensei-fs/users/mihirp/project_summary/models/falcon_40b/feedback_data",
        metadata={
            "help": "Path to trained models"
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The preference dataset to use."},
    )
    train_file: Optional[str] = field(
        default="mihir_project/data/raw_data/train.csv",
        metadata={"help": "path for the train file."},
    )
    test_file: Optional[str] = field(
        default="mihir_project/data/raw_data/test.csv",
        metadata={"help": "path for the test file."},
    )
    output_dir: Optional[str] = field(
        default="/sensei-fs/users/dernonco/interns/2023/mihir/summarization/models",
        metadata={"help": "path for the output."},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def create_and_prepare_model(args):

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="auto", trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, args.new_model)
    model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side='left', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                )

    return pipeline

pipe = create_and_prepare_model(script_args)

# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub).
#
# For CSV/JSON files this script will use the first column for the full texts and the second column for the
# summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
if script_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    datasets = load_dataset(script_args.dataset_name)
else:
    data_files = {}
    if script_args.train_file is not None:
        data_files["train"] = script_args.train_file
        extension = script_args.train_file.split(".")[-1]
    if script_args.test_file is not None:
        data_files["test"] = script_args.test_file
        extension = script_args.test_file.split(".")[-1]
    datasets = load_dataset(extension, data_files=data_files)
# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.

if not os.path.exists(script_args.output_dir):
    os.makedirs(script_args.output_dir)

    
#Evaluation of Model on test data

all_results=[]
remaining=[]
for index, out in enumerate(tqdm(pipe(KeyDataset(datasets["test"], "text"), batch_size=1, max_new_tokens=512), total=len(datasets["test"]))):
    try:
        summary= out[0]['generated_text'].split("<Summary>")[-1]
        summary= summary.strip()
        new_data= {'prediction': summary}
        all_results.append(new_data)
    except:
        print(out)
        summary= out[0]['generated_text']
        summary= summary.strip()
        new_data= {'prediction': summary}
        all_results.append(new_data)
        remaining.append(new_data)
    
    pred_df= pd.DataFrame(all_results)
    pred_df.to_csv(join(script_args.output_dir, 'predictions.csv'), index=False)
        
    if len(remaining)>0:
        remaining_df= pd.DataFrame(remaining)
        remaining_df.to_csv(join(script_args.output_dir, 'remaining.csv'), index=False)
        

