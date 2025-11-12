import random
from collections import defaultdict

def random_sampling(data, sample_size):
    """
    Perform random sampling on the given data.

    Parameters:
    data (list): The dataset to sample from.
    sample_size (int): The number of samples to draw.

    Returns:
    list: A list containing the randomly sampled elements.
    """
    if sample_size > len(data):
        raise ValueError("Sample size cannot be greater than the dataset size.")
    return random.sample(data, sample_size)

def stratified_sampling(data, strata_key, sample_size_per_stratum):
    """
    Perform stratified sampling on the given data.

    Parameters:
    data (list of dict): The dataset to sample from, where each element is a dictionary.
    strata_key (str): The key in the dictionary used to define strata.
    sample_size_per_stratum (int): The number of samples to draw from each stratum.

    Returns:
    list: A list containing the stratified sampled elements.
    """
    strata = defaultdict(list)
    
    # Group data by strata
    for item in data:
        key = item[strata_key]
        strata[key].append(item)
    
    sampled_data = []
    
    # Sample from each stratum
    for stratum, items in strata.items():
        if sample_size_per_stratum > len(items):
            raise ValueError(f"Sample size per stratum cannot be greater than the size of stratum '{stratum}'.")
        sampled_data.extend(random.sample(items, sample_size_per_stratum))
    
    return sampled_data

def cluster_sampling(data, cluster_key, num_clusters):
    """
    Perform cluster sampling on the given data.

    Parameters:
    data (list of dict): The dataset to sample from, where each element is a dictionary.
    cluster_key (str): The key in the dictionary used to define clusters.
    num_clusters (int): The number of clusters to sample.

    Returns:
    list: A list containing the cluster sampled elements.
    """
    clusters = defaultdict(list)
    
    # Group data by clusters
    for item in data:
        key = item[cluster_key]
        clusters[key].append(item)
    
    if num_clusters > len(clusters):
        raise ValueError("Number of clusters to sample cannot be greater than the total number of clusters.")
    
    sampled_data = []
    
    # Sample clusters
    sampled_clusters = random.sample(list(clusters.values()), num_clusters)
    
    for cluster in sampled_clusters:
        sampled_data.extend(cluster)
    
    return sampled_data

def reservoir_sampling(data, sample_size):
    """
    Perform reservoir sampling on the given data.

    Parameters:
    data (iterable): The dataset to sample from.
    sample_size (int): The number of samples to draw.

    Returns:
    list: A list containing the reservoir sampled elements.
    """
    reservoir = []
    
    for i, item in enumerate(data):
        if i < sample_size:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = item
    
    return reservoir