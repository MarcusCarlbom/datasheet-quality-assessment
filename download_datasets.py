from datasets import load_dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

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
    
    def download_uci_with_split(self, url: str, name: str, has_header: bool = False):
        """
        Download from UCI and create train/test split
        UCI never provides splits
        """
        print(f"Downloading UCI: {name}...")
        try:
            if has_header:
                df = pd.read_csv(url)
            else:
                df = pd.read_csv(url, header=None)
            
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
            print(f"Created splits: train ({len(train_df)}), test ({len(test_df)})")
            
        except Exception as e:
            print(f"Failed: {e}")


def main():
    downloader = DatasetDownloader()
    
    print("\n--- HuggingFace Datasets (10) ---\n")
    
    downloader.download_huggingface_with_splits("imdb", has_splits=True)
    downloader.download_huggingface_with_splits("rotten_tomatoes", has_splits=True)
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
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "iris"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        "wine"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        "breast_cancer_wisconsin"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
        "mushroom"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
        "car_evaluation"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
        "letter_recognition"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
        "abalone"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data",
        "balance_scale"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
        "ecoli"
    )
    
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data",
        "dermatology"
    )
    
    print("\nDownloaded:")
    print("10 HuggingFace datasets (text classification)")
    print("10 OpenML datasets (tabular)")
    print("10 UCI datasets (classic ML)")


if __name__ == "__main__":
    main()