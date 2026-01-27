from datasets import load_dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import requests
import zipfile
import io

class DatasetDownloader:
    def __init__(self, base_path="./data"):
        self.base_path = Path(base_path)
        for source in ['huggingface', 'openml', 'uci']:
            (self.base_path / source).mkdir(parents=True, exist_ok=True)
    
    def download_huggingface_with_splits(self, dataset_name: str, 
                                        config: str = None,
                                        has_splits: bool = True):
        print(f"Downloading HuggingFace: {dataset_name}...")
        
        if has_splits:
            try:
                if config:
                    train = load_dataset(dataset_name, config, split="train")
                    test = load_dataset(dataset_name, config, split="test")
                else:
                    train = load_dataset(dataset_name, split="train")
                    test = load_dataset(dataset_name, split="test")
                
                train_df = train.to_pandas()
                test_df = test.to_pandas()
                
                safe_name = dataset_name.replace('/', '_')
                if config:
                    safe_name = f"{safe_name}_{config}"
                
                train_df.to_parquet(
                    self.base_path / "huggingface" / f"{safe_name}_train.parquet"
                )
                test_df.to_parquet(
                    self.base_path / "huggingface" / f"{safe_name}_test.parquet"
                )
                print(f"  ✓ Saved train ({len(train_df)}) and test ({len(test_df)})")
                
            except Exception as e:
                print(f"Failed to load splits: {e}")
                self._download_and_split_hf(dataset_name, config)
        else:
            self._download_and_split_hf(dataset_name, config)
    
    def _download_and_split_hf(self, dataset_name: str, config: str = None):
        """Download HF dataset without predefined splits, create our own"""
        if config:
            dataset = load_dataset(dataset_name, config, split="train")
        else:
            dataset = load_dataset(dataset_name, split="train")
        
        df = dataset.to_pandas()
        
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42
        )
        
        safe_name = dataset_name.replace('/', '_')
        if config:
            safe_name = f"{safe_name}_{config}"
        
        train_df.to_parquet(
            self.base_path / "huggingface" / f"{safe_name}_train.parquet"
        )
        test_df.to_parquet(
            self.base_path / "huggingface" / f"{safe_name}_test.parquet"
        )
        print(f"Created splits: train ({len(train_df)}), test ({len(test_df)})")
    
    def download_openml_with_split(self, data_id: int, name: str):
        """
        Download from OpenML and create train/test split
        OpenML doesn't provide splits via fetch_openml
        """
        print(f"Downloading OpenML: {name} (ID: {data_id})...")
        try:
            dataset = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
            df = pd.concat([dataset.data, dataset.target], axis=1)
            
            # Create 80/20 split with stratification if possible
            try:
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=42, 
                    stratify=dataset.target if len(dataset.target.unique()) < 20 else None
                )
            except:
                # Fallback without stratification if it fails
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=42
                )
            
            train_df.to_parquet(
                self.base_path / "openml" / f"{name}_train.parquet"
            )
            test_df.to_parquet(
                self.base_path / "openml" / f"{name}_test.parquet"
            )
            print(f"Created splits: train ({len(train_df)}), test ({len(test_df)})")
            
        except Exception as e:
            print(f"Failed: {e}")
    
    def download_uci_with_split(self, url: str, name: str, data_filename: str = None, has_header: bool = False):
        """
        Download from UCI (new zip format) and create train/test split
        
        Args:
            url: The new UCI zip download URL
            name: Dataset name for saving
            data_filename: Specific .data or .csv file inside the zip (if None, uses first .data or .csv found)
            has_header: Whether the data file has a header row
        """
        print(f"Downloading UCI: {name}...")
        try:
            # Download the zip file
            response = requests.get(url)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                if data_filename:
                    target_file = data_filename
                else:
                    data_files = [f for f in z.namelist() if f.endswith(('.data', '.csv')) and not f.startswith('__MACOSX')]
                    if not data_files:
                        raise ValueError(f"No .data or .csv files found in {name}")
                    target_file = data_files[0]
                
                with z.open(target_file) as f:
                    if has_header:
                        df = pd.read_csv(f)
                    else:
                        df = pd.read_csv(f, header=None)
            
            # Create 80/20 split
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )
            
            train_df.to_csv(
                self.base_path / "uci" / f"{name}_train.csv", index=False
            )
            test_df.to_csv(
                self.base_path / "uci" / f"{name}_test.csv", index=False
            )
            print(f"  ✓ Created splits: train ({len(train_df)}), test ({len(test_df)})")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def main():
    downloader = DatasetDownloader()
    
    print("\n--- HuggingFace Datasets (10) ---\n")
    
    downloader.download_huggingface_with_splits("imdb", has_splits=True)
    downloader.download_huggingface_with_splits("cornell-movie-review-data/rotten_tomatoes", has_splits=True)
    downloader.download_huggingface_with_splits("ag_news", has_splits=True)
    downloader.download_huggingface_with_splits("glue", config="sst2", has_splits=True)
    downloader.download_huggingface_with_splits("glue", config="cola", has_splits=True)
    downloader.download_huggingface_with_splits("yelp_review_full", has_splits=True)
    downloader.download_huggingface_with_splits("amazon_polarity", has_splits=True)
    downloader.download_huggingface_with_splits("dbpedia_14", has_splits=True)
    downloader.download_huggingface_with_splits("yahoo_answers_topics", has_splits=True)
    downloader.download_huggingface_with_splits("tweet_eval", config="emotion", has_splits=True)
    
    print("\n--- OpenML Datasets (10) ---\n")
    
    downloader.download_openml_with_split(31, "credit_g")
    downloader.download_openml_with_split(1590, "adult")
    downloader.download_openml_with_split(554, "mnist_784")
    downloader.download_openml_with_split(40498, "wine_quality_white") 
    downloader.download_openml_with_split(1461, "bank_marketing") 
    downloader.download_openml_with_split(37, "diabetes")
    downloader.download_openml_with_split(40945, "titanic")   
    downloader.download_openml_with_split(44, "spam")                  
    downloader.download_openml_with_split(1489, "phoneme")             
    downloader.download_openml_with_split(1464, "blood_transfusion") 
    
    print("\n--- UCI Datasets (10) ---\n")
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/static/public/53/iris.zip",
        "iris",
        data_filename="iris.data"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/static/public/109/wine.zip",
        "wine",
        data_filename="wine.data"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/static/public/15/breast+cancer+wisconsin+original.zip",
        "breast_cancer_wisconsin",
        data_filename="breast-cancer-wisconsin.data"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/static/public/73/mushroom.zip",
        "mushroom",
        data_filename="agaricus-lepiota.data"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/static/public/19/car+evaluation.zip",
        "car_evaluation",
        data_filename="car.data"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip",
        "letter_recognition",
        data_filename="letter-recognition.data"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/static/public/1/abalone.zip",
        "abalone",
        data_filename="abalone.data"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/static/public/12/balance+scale.zip",
        "balance_scale",
        data_filename="balance-scale.data"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/static/public/39/ecoli.zip",
        "ecoli",
        data_filename="ecoli.data"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/static/public/33/dermatology.zip",
        "dermatology",
        data_filename="dermatology.data"
    )
    
    print("\nDownloaded:")
    print("10 HuggingFace datasets (text classification)")
    print("10 OpenML datasets (tabular)")
    print("10 UCI datasets (classic ML)")


if __name__ == "__main__":
    main()