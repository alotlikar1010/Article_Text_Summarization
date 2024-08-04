from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    # raw_file_dir: str
    train_file_path: str
    test_file_path: str
