from typing import Callable, Union
from src.utils import utils

import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torchvision

log = utils.get_logger(__name__)

class ImbalancedSamplerFactory:
    
    def __init__(self,
                 num_classes: int,
                 imbalanced_class: Union[None, int] = None,
                 class_percentage: Union[None, float] = None,
                 weights: list = None):
                    
        if imbalanced_class is not None:
            assert class_percentage is not None,\
                "If class_imbalance is not None, class_percentage must also be specified"
            assert (class_percentage <= 1) and (class_percentage >= 0),\
                "class_percentage must be in domain [0,1]"
            assert weights is None,\
                "Only one of imbalanced_class or weights can be defined in ImbalancedSamplerFactory"
        
            #create weights:
            weights = np.ones(num_classes)
            weights *= (1. - class_percentage)/(num_classes - 1)
            weights[imbalanced_class] = class_percentage
            
            log.info(f"ImbalancedSamplerFactory weights: {weights}")
                    
        elif weights is not None:
            
            assert len(weights) == num_classes,\
                f'length of weights {len(weights)} must be the same as num_classes {num_classes}'
            
            #ensure weights are numpy array:
            weights = np.array(weights)
            
        self.weights = weights
            

    def train(self, dataset):
        
        if self.weights is None: 
            return None
        else:            
            return ImbalancedDatasetSampler(dataset, weightings=self.weights)

    def val(self, dataset):
        
        if self.weights is None:
            return None
        else:
            return ImbalancedDatasetSampler(dataset, weightings=self.weights)
    
    def test(self, dataset):
        
        if self.weights is None:
            return None
        else:
            return ImbalancedDatasetSampler(dataset)
       
  
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        indices: list = None,
        num_samples: int = None,
        weightings: list = None, 
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        
        #Adjust this...
        if weightings is None:   
            
            #Just weight classes according to their frequency in the dataset...
            weights = [1.0] * len(label_to_count[df["label"]])
            
            #Balances the dataset to sample points across classes equally:
            #weights = 1.0 / label_to_count[df["label"]]
            
        else:
                        
            assert len(weightings) == df['label'].nunique(),\
                f"weightings (length {len(weightings)}) should have same number of elements as label {df.label.nunique()} "
            if sum(weightings) != 1.0:
                weightings = np.array([w/sum(weightings) for w in weightings])               
            weights = pd.Series(weightings[df['label']])
                                
        self.weights = torch.DoubleTensor(weights)


    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            #return dataset.dataset.imgs[:][1]
            return np.array(dataset.dataset.targets)[dataset.indices].tolist()
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.targets
        else:
            raise NotImplementedError

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples
