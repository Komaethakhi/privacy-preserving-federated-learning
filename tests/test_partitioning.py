import pytest
import pandas as pd
from data.generate_data import generate_synthetic_healthcare_data
from data.partitioning import partition_dataset

def test_iid_partitioning():
    df = generate_synthetic_healthcare_data(num_samples=500, random_seed=42)
    names = ["Hospital A", "Hospital B", "Hospital C", "Hospital D", "Hospital E"]
    partitions = partition_dataset(df, hospital_names=names, partition_type="iid", random_seed=42)
    
    assert len(partitions) == 5
    total_partitioned = sum(len(p) for p in partitions.values())
    assert total_partitioned == 500

def test_non_iid_partitioning():
    df = generate_synthetic_healthcare_data(num_samples=500, random_seed=42)
    names = ["Hospital A", "Hospital B", "Hospital C", "Hospital D", "Hospital E"]
    partitions = partition_dataset(df, hospital_names=names, partition_type="non_iid", dirichlet_alpha=0.5, random_seed=42)
    
    assert len(partitions) == 5
    total_partitioned = sum(len(p) for p in partitions.values())
    assert total_partitioned == 500

def test_demographic_partitioning():
    df = generate_synthetic_healthcare_data(num_samples=500, random_seed=42)
    names = ["Hospital A", "Hospital B", "Hospital C", "Hospital D", "Hospital E"]
    partitions = partition_dataset(df, hospital_names=names, partition_type="demographic", random_seed=42)
    
    assert len(partitions) == 5
    total_partitioned = sum(len(p) for p in partitions.values())
    assert total_partitioned == 500
    assert partitions["Hospital A"]["age"].mean() > partitions["Hospital C"]["age"].mean()
