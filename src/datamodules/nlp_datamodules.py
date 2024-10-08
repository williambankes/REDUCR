# -*- coding: utf-8 -*-

from pathlib import Path

import torch
import pandas as pd
from src.utils import utils
from src.datamodules.datamodules import TestDataloader
import pytorch_lightning as pl
from torch.utils.data import DataLoader

#NLP Imports
import datasets
from datasets import DatasetDict, concatenate_datasets, Dataset
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).parent.absolute()
print(f"cwd: {SCRIPT_DIR}")

log = utils.get_logger(__name__)

#Define the patch_getitem class:
class _(datasets.arrow_dataset.Dataset):
            
    def __getitem__(self, key):
        
        output_dict = super().__getitem__(key)
                    
        global_index = output_dict.pop("idx")
        
        data = output_dict
        
        target = output_dict["labels"]
        
        categories = output_dict['labels']
        
        return global_index, data, target, categories  

def patch_getitem(instance):
    
    """
    Hacky way of allowing the HuggingFace Datasets classes of returning an 
    additional category class as well as an index, sentence and label.
    
    If a cleaner solution exists would be intersted to know...
    """
       
    #ensure the class is not already an instance of _:
    if not isinstance(instance, _):
        instance.__class__ = _
    
    #Return the instance to avoid side-effects    
    return instance
    
    

class GLUEDataModule(pl.LightningDataModule):
    
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "idx",
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        no_test_set_avail,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        batch_size: int = 32,
        shuffle = None,
        sequence = None,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = batch_size
        self.eval_batch_size = batch_size

        self.shuffle=shuffle
        self.no_test_set_avail = no_test_set_avail

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,
                                                       use_fast=True)
                                                       #,cache_dir='.')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.sequence = sequence
        self.num_classes = self.glue_task_num_labels[task_name]

    def setup(self, stage: str, sampler_factory=None, test=False):
        
        self.test = test        
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self, sample=False, keep_test_on:bool=True):
        if self.no_test_set_avail:
            train_set = self.dataset["train"].select(range(0, len(self.dataset["train"]), 2))
        else:
            train_set = self.dataset["train"]
                       
        #Patch the dataset get item method
        #train_set = patch_getitem(train_set) removed horrible patch code        

        # if there is a sequence, i.e. in the main/large model training, override the dataset instance to only return elements from the core set
        if self.sequence is not None:
            setattr(train_set, "sequence", self.sequence)
            setattr(train_set, "idx", train_set["idx"])

            #Not sure I like this code ... why can't you just wrap it?
            def patch_instance(instance):
                """Create a new class derived from instance, override its relevant method.
                Then set instance type to the new class. 
                I have to implement it like this because you can't directly monkey patch magic methods on the instance level."""
                class _(type(instance)):
                    def __getitem__(self, key):
                        """Can be used to index columns (by string names) or rows (by integer index or iterable of indices or bools)."""
                        if not isinstance(key, str):
                            key_temp = torch.tensor(self.sequence[key])
                            key = [loc_idx for loc_idx, idx in enumerate(train_set.idx) if idx in key_temp]

                            if len(key) == 1:
                                key = key[0]

                        return self._getitem(
                            key,
                        )
                    def __len__(self):
                        return len(self.sequence)

                instance.__class__ = _
                
                return

            patch_instance(train_set)
        
        if self.test and keep_test_on:
            return TestDataloader(
                DataLoader(
                train_set, batch_size=self.train_batch_size, 
                shuffle=self.shuffle), 
                n_step=32)
        
        else:
            return DataLoader(train_set, batch_size=self.train_batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            if self.no_test_set_avail:
                val_set = self.dataset["train"].select(range(1, 
                                     len(self.dataset["train"]), 2))
            else:
                val_set = self.dataset["validation"]
            
            
            if self.test:
                return TestDataloader(
                    DataLoader(val_set,#patch_getitem(val_set), 
                               batch_size=self.eval_batch_size, 
                               shuffle=True), 
                    n_step=32)
            else:
                return DataLoader(val_set,#patch_getitem(val_set), 
                                  batch_size=self.eval_batch_size, 
                                  shuffle=True)
        
        elif len(self.eval_splits) > 1:
            return [DataLoader(val_set,#patch_getitem(self.dataset[x]),
                               batch_size=self.eval_batch_size, 
                               shuffle=True) for x in self.eval_splits]

    def test_dataloader(self):
        
        #We do not patch_getitem here as we do not log the per class loss in the test set
        #Maybe we should...
        
        if len(self.eval_splits) == 1:
            
            if self.no_test_set_avail:
                data = self.dataset['validation']#patch_getitem(self.dataset['validation'])
            else:
                data = self.dataset['test']#patch_getitem(self.dataset['test'])
                
            if self.test:
                return TestDataloader(
                        DataLoader(data, batch_size=self.eval_batch_size),
                        n_step=1)
            
            else:
                return DataLoader(data, batch_size=self.eval_batch_size)
                
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x],#patch_getitem(self.dataset[x]),
                               batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features
    
    def _get_set_of_labels(self):
        
        return self.num_classes
    
    def percentage_corrupted(self, *args, **kwargs):
        
        return 0
    
    
class QQPDataModule(GLUEDataModule):
    
    def __init__(self, model_name_or_path:str, small_val=False, **kwargs):
        
        #Setup the GLUE Datamodule but with mnli as the given task:
        super().__init__(model_name_or_path=model_name_or_path,
                         no_test_set_avail=False,
                         task_name='qqp', **kwargs)
        
        #This is new:
        self.small_val = small_val
        self.eval_splits = ['validation']
        
        
    def setup(self, stage: str, sampler_factory=None, test=False, cache_dir=None):
        
        self.test = test
        self.dataset = datasets.load_dataset("glue", "qqp", cache_dir=cache_dir)
        
        #Convert features from text to token's and attention maps
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)
    
        #combine the datasets:
        split_labels = [x for x in self.dataset.keys() if ("validation" in x) or\
                                                             ("train" in x)]
        data = [self.dataset[label] for label in split_labels]
        
        self.dataset = concatenate_datasets(data)   
        
        #set a global index here: -> this needs to be set for each of the three datasets...
        self.dataset = self.dataset.\
            remove_columns('idx').\
            map(lambda example, idx: {"idx":idx}, with_indices=True)
        setattr(self.dataset, "idx", list(range(len(self.dataset))))
        
        self.dataset = self.train_test_val_split(self.dataset)
        
    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            if self.no_test_set_avail:
                val_set = self.dataset["train"].select(range(1, 
                                     len(self.dataset["train"]), 2))
            else:
                val_set = self.dataset["validation"]
                
            if self.small_val:
                positive_examples = val_set.filter(lambda x:x['labels'] == 1)
                negative_examples = val_set.filter(lambda x:x['labels'] == 0)
                
                neg_indices = list(range(len(positive_examples)))
                negative_examples = negative_examples.select(neg_indices)
                
                val_set = concatenate_datasets([positive_examples, negative_examples])
            
            return DataLoader(patch_getitem(val_set), 
                              batch_size=self.eval_batch_size, 
                              shuffle=True)
        
        elif len(self.eval_splits) > 1:
            return [DataLoader(patch_getitem(self.dataset[x]),
                               batch_size=self.eval_batch_size, 
                               shuffle=True) for x in self.eval_splits]
        
   
    def train_test_val_split(self, dataset):
        
        #Calculate the relevant train test splits
        size = len(dataset)
        train_end_index = size//2
        val_end_index = int(train_end_index + size//2.5)
        
        idx = list(range(size))
        train_idx = idx[:train_end_index]
        val_idx = idx[train_end_index:val_end_index]
        test_idx = idx[val_end_index:]
        
        #Select the test idx:
        test_dataset = dataset.select(test_idx)

        #Remove positive samples from the train and val dataset:
        train_val_dataset = dataset.select(train_idx + val_idx)
               
        #filter out examples:
        positive_examples = train_val_dataset.filter(lambda x:x['labels'] == 1)
        negative_examples = train_val_dataset.filter(lambda x:x['labels'] == 0)

        pos_idx = list(range(len(positive_examples)))
        pos_idx = pos_idx[:len(positive_examples)//2]
        positive_examples = positive_examples.select(pos_idx)
        
        train_val_dataset = concatenate_datasets([positive_examples, negative_examples])

        #split into train and val dataset
        size = len(train_val_dataset)
        train_idx = list(range(size))[::2]
        val_idx = list(range(size))[1::2]
        
        train = train_val_dataset.select(train_idx).filter(lambda x:x['idx'] != 1)
        val = train_val_dataset.select(val_idx).filter(lambda x:x['idx'] != 0)
        
        #reset index:        
        train = train.remove_columns('idx').\
            map(lambda example, idx: {"idx":idx}, with_indices=True)
        val = val.remove_columns('idx').\
            map(lambda example, idx: {"idx":idx}, with_indices=True)
        test = test_dataset.remove_columns('idx').\
            map(lambda example, idx: {"idx":idx}, with_indices=True)
        
        dataset = DatasetDict({
            'train': train,
            'test': test,
            'validation': val
            })
        
        return dataset
        
class MNLIDataModule(GLUEDataModule):
    
    def __init__(self, model_name_or_path: str, **kwargs):
        """
        A dataloader class specifically designed for the MultiNLI dataset, the
        validation datasets are combined with the training datasets and separated
        into a labelled, train, test and validation split.
        
        no_test_set_avail = True
        task_name = 'mnli'

        Parameters
        ----------
        model_name_or_path : str
            DESCRIPTION.
        **kwargs : TYPE
            any parameters passed to the GLUEDataModule

        Returns
        -------
        None.

        """
        
        #Setup the GLUE Datamodule but with mnli as the given task:
        super().__init__(model_name_or_path=model_name_or_path,
                         no_test_set_avail=False,
                         task_name='mnli', **kwargs)
        
        self.eval_splits = ['validation']
        
    def setup(self, stage: str, sampler_factory=None, test=False):
        
        self.test = test
        self.dataset = datasets.load_dataset("nyu-mll/glue", "mnli", cache_dir=".")
        
        #Convert features from text to token's and attention maps
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        #combine the datasets:
        split_labels = [x for x in self.dataset.keys() if ("validation" in x) or\
                                                             ("train" in x)]
        data = [self.dataset[label] for label in split_labels]
        
        self.dataset = concatenate_datasets(data)  
        
        #set a global index here:-> needs to be done for each of the datasets
        self.dataset = self.dataset.\
            remove_columns('idx').\
            map(lambda example, idx: {"idx":idx}, with_indices=True)
        setattr(self.dataset, "idx", list(range(len(self.dataset))))
        
        self.dataset = self.train_test_val_split(self.dataset)
        
    def train_test_val_split(self, dataset):
        
        #Calculate the relevant train test splits
        size = len(dataset)
        train_end_index = size//2
        val_end_index = int(train_end_index + size//2.5)
        
        idx = list(range(size))
        train_idx = idx[:train_end_index]
        val_idx = idx[train_end_index:val_end_index]
        test_idx = idx[val_end_index:]
        
        train = dataset.select(train_idx).remove_columns('idx').\
            map(lambda example, idx: {"idx":idx}, with_indices=True)
        val = dataset.select(val_idx).remove_columns('idx').\
            map(lambda example, idx: {"idx":idx}, with_indices=True)
        test = dataset.select(test_idx).remove_columns('idx').\
            map(lambda example, idx: {"idx":idx}, with_indices=True)
                
        dataset = DatasetDict({
            'train': train,
            'test': test,
            'validation': val
            })
        
        return dataset


if __name__ == '__main__': 

    datamodule = MNLIDataModule(model_name_or_path='openaccess-ai-collective/tiny-mistral')
    datamodule.setup(stage=None, test=True)
        
    loader = datamodule.train_dataloader(sample=False)
    outputs = list()
    for batch in loader:
        idx, x, target, category = batch
        outputs.append(idx)

#%% Analyse the code directly to avoid multiple repeats:

import datasets
from transformers import AutoTokenizer
from datasets import DatasetDict, concatenate_datasets, Dataset
from src.datamodules.datamodules import TestDataloader
from torch.utils.data import DataLoader
    
################################ METHODS AND CLASSES ##########################

def convert_to_features(example_batch, indices=None):

    text_fields = ["premise", "hypothesis"] 
    tokenizer = AutoTokenizer.from_pretrained('openaccess-ai-collective/tiny-mistral',
                                                   use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Either encode single sentence or sentence pairs
    if len(text_fields) > 1:
        texts_or_text_pairs = list(zip(example_batch[text_fields[0]], example_batch[text_fields[1]]))
    else:
        texts_or_text_pairs = example_batch[text_fields[0]]

    # Tokenize the text/text pairs
    features = tokenizer.batch_encode_plus(
        texts_or_text_pairs, max_length=256, pad_to_max_length=True, truncation=True
    )

    # Rename label to labels to make it easier to pass to model forward
    features["labels"] = example_batch["label"]
    
    

    return features    

#Define the patch_getitem class:
class _(datasets.arrow_dataset.Dataset):
            
    def __getitem__(self, key):
        
        output_dict = super().__getitem__(key)
                    
        global_index = output_dict.pop("idx")
        
        data = output_dict
        
        target = output_dict["labels"]
        
        categories = output_dict['labels']
        
        print(type(global_index))
        
        return global_index, data, target, categories  

def patch_getitem(instance):
    
    """
    Hacky way of allowing the HuggingFace Datasets classes of returning an 
    additional category class as well as an index, sentence and label.
    
    If a cleaner solution exists would be intersted to know...
    """
       
    #ensure the class is not already an instance of _:
    if not isinstance(instance, _):
        instance.__class__ = _
    
    #Return the instance to avoid side-effects    
    return instance

def train_test_val_split(dataset):
    
    #Calculate the relevant train test splits
    size = len(dataset)
    train_end_index = size//2
    val_end_index = int(train_end_index + size//2.5)
    
    idx = list(range(size))
    train_idx = idx[:train_end_index]
    val_idx = idx[train_end_index:val_end_index]
    test_idx = idx[val_end_index:]
    
    train = dataset.select(train_idx).remove_columns('idx').\
        map(lambda example, idx: {"idx":idx}, with_indices=True)
    val = dataset.select(val_idx).remove_columns('idx').\
        map(lambda example, idx: {"idx":idx}, with_indices=True)
    test = dataset.select(test_idx).remove_columns('idx').\
        map(lambda example, idx: {"idx":idx}, with_indices=True)
            
    dataset = DatasetDict({
        'train': train,
        'test': test,
        'validation': val
        })
    
    return dataset

###############################  HYPERPARAMETERS ##############################
test = True
dataset = datasets.load_dataset("nyu-mll/glue", "mnli", cache_dir=".")
text_fields = ["premise", "hypothesis"]
loader_columns = [
    "idx",
    "datasets_idx",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "start_positions",
    "end_positions",
    "labels",
]
no_test_set_avail=False


###########################  DATASET SETUP FUNCTION  ##########################
#Convert features from text to token's and attention maps
for split in dataset.keys():
    dataset[split] = dataset[split].map(
        convert_to_features,
        batched=True,
        remove_columns=["label"],
    )
    columns = [c for c in dataset[split].column_names if c in loader_columns]
    dataset[split].set_format(type="torch", columns=columns)

#combine the datasets:
split_labels = [x for x in dataset.keys() if ("validation" in x) or\
                                                     ("train" in x)]
data = [dataset[label] for label in split_labels]

dataset = datasets.concatenate_datasets(data)  

#set a global index here:-> needs to be done for each of the datasets
dataset = dataset.\
    remove_columns('idx').\
    map(lambda example, idx: {"idx":idx}, with_indices=True)
setattr(dataset, "idx", list(range(len(dataset))))

dataset = train_test_val_split(dataset)

############################# DATA LOADER ####################################

if no_test_set_avail:
    train_set = dataset["train"].select(range(0, len(dataset["train"]), 2))
else:
    train_set = dataset["train"]
                  
#Patch the dataset get item method
#train_set = patch_getitem(train_set)        

if test:
    loader = TestDataloader(
        DataLoader(
        train_set, batch_size=32, 
        shuffle=True), 
        n_step=32)

else:
    loader = DataLoader(train_set, batch_size=32, shuffle=True)
    
#%% Load sample from the dataset
#loader2 = DataLoader(dataset['train'], batch_size=32, shuffle=True)

sample = next(iter(loader))



