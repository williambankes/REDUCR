import torch


def top_x_indices(vec, x, largest):
    """
    Returns the indices of the x largest/smallest entries in vec.

    Args:
        vec: tensor, number of samples to be selected
        x: int, number of indices to be returned
        smallest: bool, if true, the x largest entries are selected; if false,
        the x smallest entries are selected
    Returns:
        top_x_indices: tensor, top x indices, sorted
        other_indices: tensor, the indices that were not selected
    """

    sorted_idx = torch.argsort(vec, descending=largest)

    top_x_indices = sorted_idx[:x]
    other_indices = sorted_idx[x:]

    return top_x_indices, other_indices

def online_batch_selection_indices(vec, x):
    """
    Implementation of the stochastic sampling from (Loshchilov&Hutter,2016).

    Parameters
    ----------
    vec : torch.tensor
        vector input of selection scores
    x : int
        number of points to pick

    Returns
    -------
    sampled_indices : torch.tensor
        selected indices sampled accordingly
    
    other_indices : torch.tensor
        non selected indices

    """

    #hard coding the selection ratio to 100:
    se = torch.tensor(100)
    ratio = torch.exp(torch.log(se)/len(vec))
    ratio = ratio.to(device=vec.device)
    
    #indices:
    indices = torch.arange(len(vec)).to(device=vec.device)
        
    #calculate probabilities and sample:
    ps = 1/torch.pow(ratio, indices)
    ps /= ps.sum()
    selected_index = torch.multinomial(ps, num_samples=x, replacement=False)
    
    #Match the sampled indices to the sorted indices:
    sorted_idx = torch.argsort(vec, descending=True)
    sampled_indices = sorted_idx[selected_index]
    
    return sampled_indices, indices[~torch.isin(indices, sampled_indices)]
    
    

def power_random_indices(vec, x, beta):
    """
    Returns a random selection of indices where points are sampled proportionally
    to their rho loss

    Parameters
    ----------
    vec : torch.tensor, values of indices to be sorted
    x : int, number of samples to select

    Returns
    -------
    sampled_indices: tensor, indices selected
    other_indices: tensor, indices not-selected
    """
    vec += vec.min().abs() #Add minimal value to ensure >=0
    indices = torch.arange(len(vec)).to(device=vec.device)
    beta = beta.to(device=vec.device)
    
    vec = vec.pow(beta)
    sampled_indices = torch.multinomial(vec, num_samples=x, replacement=False)
    
    return sampled_indices, indices[~torch.isin(indices, sampled_indices)]

def softmax_random_indices(vec, x, beta):
    
    #Define indices and ensure data on correct device:
    indices = torch.arange(len(vec)).to(device=vec.device)
    beta = beta.to(device=vec.device)
    
    #Calculate the normalised log probability vectors: 
    vec = vec * beta
    normalised_logits = vec - vec.logsumexp(dim=-1, keepdim=True)
    
    #Sample indices from the distribution: 
    sampled_indices = torch.multinomial(normalised_logits.exp(), #exponentiate for multinomial
                                        num_samples=x,
                                        replacement=False)
    
    return sampled_indices, indices[~torch.isin(indices, sampled_indices)]
    
    
def rank_random_indices(vec, x, beta):
    
    #Define indices and ensure data on correct device:
    indices = torch.arange(len(vec)).to(device=vec.device)
    beta = beta.to(device=vec.device)
    
    #Sort the input arguments
    sorted_idx = torch.argsort(vec, descending=True)

    #Calculate the normalised log probability vectors:    
    logits = - beta * (indices + 1).log()
    normalised_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    
    #Sample the relevant indices:
    sampled_ranks = torch.multinomial(normalised_logits.exp(), 
                                      num_samples=x,
                                      replacement=False)

    sampled_indices = sorted_idx[sampled_ranks]
    return sampled_indices, sorted_idx[~torch.isin(sorted_idx, sampled_indices)]       
    

def top_x_payoff_indices(payoff_matrix, x):
        
    #first we minimise and reduce the columns:
    min_values = torch.min(payoff_matrix, dim=1)[0]
    
    #select top x indices from values minimised:
    sorted_idx = torch.argsort(min_values, descending=True)

    top_x_indices = sorted_idx[:x]
    other_indices = sorted_idx[x:]

    return top_x_indices, other_indices


def create_logging_dict(variables_to_log, selected_minibatch, not_selected_minibatch):
    """
    Create the dictionary for logging, in which, for each variable/metric, the
    selected and the not selected entries are logged separately as
    "selected_<var_name>" and "not_selected_<var_name>".

    Args:
        variables_to-log: dict, with key:var_name to be logged, value: tensor of values to be logger.
    """

    metrics_to_log = {}
    for name, metric in variables_to_log.items():
        metrics_to_log["selected_" + name] = metric[selected_minibatch].cpu().numpy()
        metrics_to_log["not_selected_" + name] = (
            metric[not_selected_minibatch].cpu().numpy()
        )
    return metrics_to_log
