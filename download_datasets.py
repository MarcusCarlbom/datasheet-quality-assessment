import gc
from datasets import load_dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import requests
import zipfile
import io
from datasets import get_dataset_config_info

class DatasetDownloader:
    def __init__(self, base_path="./data"):
        self.base_path = Path(base_path)
        for source in ['huggingface', 'openml', 'uci']:
            (self.base_path / source).mkdir(parents=True, exist_ok=True)
    
    def download_huggingface_with_splits(self, dataset_name: str, 
                                        config: str = None,
                                        has_splits: bool = True,
                                        max_size_gb: float = 10.0):
        print(f"Downloading HuggingFace: {dataset_name}...")
        
        # Check dataset size before downloadingR
        try:
            if config:
                dataset_info = get_dataset_config_info(dataset_name, config_name=config)
            else:
                dataset_info = get_dataset_config_info(dataset_name)
            
            # Get total size in bytes and convert to GB
            if dataset_info.dataset_size:
                size_gb = dataset_info.dataset_size / (1024**3)
                if size_gb > max_size_gb:
                    print(f"SKIPPED: Dataset too large ({size_gb:.2f} GB > {max_size_gb} GB limit)")
                    return
                print(f"Dataset size: {size_gb:.2f} GB")
        except Exception as e:
            print(f"Warning: Could not check dataset size: {e}")
            # Continue anyway if size check fails
        
        if has_splits:
            try:
                if config:
                    train = load_dataset(dataset_name, config, split="train", streaming=True)
                    test = load_dataset(dataset_name, config, split="test", streaming=True)
                else:
                    train = load_dataset(dataset_name, split="train", streaming=True)
                    test = load_dataset(dataset_name, split="test", streaming=True)
                
                # Convert streaming dataset to list, then to pandas
                train_df = pd.DataFrame(list(train))
                test_df = pd.DataFrame(list(test))
                
                train_size = len(train_df)
                test_size = len(test_df)
            
                del train, test
                gc.collect()
                
                safe_name = dataset_name.replace('/', '_')
                if config:
                    safe_name = f"{safe_name}_{config}"
                
                train_df.to_parquet(
                    self.base_path / "huggingface" / f"{safe_name}_train.parquet"
                )
                test_df.to_parquet(
                    self.base_path / "huggingface" / f"{safe_name}_test.parquet"
                )
                del train_df, test_df
                gc.collect()
                
                print(f"Saved train ({train_size}) and test ({test_size})")
                
            except ValueError as e:
                # Handle datasets with only one split or different split names
                if "Bad split" in str(e) or "Unknown split" in str(e):
                    print(f"Note: Dataset doesn't have train/test splits, will create custom split")
                    self._download_and_split_hf(dataset_name, config, max_size_gb)
                else:
                    raise
            except Exception as e:
                print(f"Failed to load splits: {e}")
                self._download_and_split_hf(dataset_name, config, max_size_gb)
        else:
            self._download_and_split_hf(dataset_name, config, max_size_gb)

    def _download_and_split_hf(self, dataset_name: str, config: str = None, max_size_gb: float = 10.0):
        """Download HF dataset without predefined splits, create our own"""
        
        # Check size here too
        try:
            from datasets import get_dataset_config_info
            if config:
                dataset_info = get_dataset_config_info(dataset_name, config_name=config)
            else:
                dataset_info = get_dataset_config_info(dataset_name)
            
            if dataset_info.dataset_size:
                size_gb = dataset_info.dataset_size / (1024**3)
                if size_gb > max_size_gb:
                    print(f"SKIPPED: Dataset too large ({size_gb:.2f} GB > {max_size_gb} GB limit)")
                    return
        except Exception:
            pass  # Continue if size check fails
        
        try:
            # Try to load train split first
            if config:
                dataset = load_dataset(dataset_name, config, split="train", streaming=True)
            else:
                dataset = load_dataset(dataset_name, split="train", streaming=True)
        except ValueError:
            # If train doesn't exist, try test split
            try:
                if config:
                    dataset = load_dataset(dataset_name, config, split="test", streaming=True)
                else:
                    dataset = load_dataset(dataset_name, split="test", streaming=True)
            except ValueError:
                # If neither works, load all available data
                if config:
                    dataset = load_dataset(dataset_name, config, streaming=True)
                else:
                    dataset = load_dataset(dataset_name, streaming=True)
                # Convert entire dataset
                dataset = dataset['train'] if 'train' in dataset else list(dataset.values())[0]
        
        df = pd.DataFrame(list(dataset))
        del dataset
        gc.collect()
        
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42
        )
        del df 
        
        train_size = len(train_df)
        test_size = len(test_df)
        
        safe_name = dataset_name.replace('/', '_')
        if config:
            safe_name = f"{safe_name}_{config}"
        
        train_df.to_parquet(
            self.base_path / "huggingface" / f"{safe_name}_train.parquet"
        )
        test_df.to_parquet(
            self.base_path / "huggingface" / f"{safe_name}_test.parquet"
        )
        del train_df, test_df
        gc.collect()
        print(f"Created splits: train ({train_size}), test ({test_size})")
    
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
            del df
            train_size = len(train_df)
            test_size = len(test_df)
            train_df.to_parquet(
                self.base_path / "openml" / f"{name}_train.parquet"
            )
            test_df.to_parquet(
                self.base_path / "openml" / f"{name}_test.parquet"
            )
            del train_df, test_df
            del dataset 
            print(f"Created splits: train ({train_size}), test ({test_size})")
            
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
            
            train_df_size = len(train_df)
            test_df_size = len(test_df)
            
            del train_df, test_df
            print(f"Created splits: train ({train_df_size}), test ({test_df_size})")
            
        except Exception as e:
            print(f"Failed: {e}")


def main():
    downloader = DatasetDownloader()

    print("\n--- HuggingFace Datasets (30) ---\n")
    #--
    downloader.download_huggingface_with_splits("imdb", has_splits=True)
    downloader.download_huggingface_with_splits("cornell-movie-review-data/rotten_tomatoes", has_splits=True)
    downloader.download_huggingface_with_splits("ag_news", has_splits=True)
    gc.collect()
    downloader.download_huggingface_with_splits("glue", config="sst2", has_splits=True)
    downloader.download_huggingface_with_splits("glue", config="cola", has_splits=True)
    downloader.download_huggingface_with_splits("yelp_review_full", has_splits=True)
    gc.collect()
    downloader.download_huggingface_with_splits("amazon_polarity", has_splits=True)
    downloader.download_huggingface_with_splits("dbpedia_14", has_splits=True)
    downloader.download_huggingface_with_splits("yahoo_answers_topics", has_splits=True)
    gc.collect()
    downloader.download_huggingface_with_splits("tweet_eval", config="emotion", has_splits=True)
    #--
    downloader.download_huggingface_with_splits("allenai/openbookqa",config= "additional",has_splits=True)
    # downloader.download_huggingface_with_splits("RogersPyke/robocoin_10K_20260121", has_splits=True)
    gc.collect()
    downloader.download_huggingface_with_splits("aps/super_glue", config="axb",has_splits=True)
    downloader.download_huggingface_with_splits("nyu-mll/glue", config="mnli_matched", has_splits=True)
    downloader.download_huggingface_with_splits("nyu-mll/glue", config="qnli", has_splits=True)
    gc.collect()
    downloader.download_huggingface_with_splits("baber/piqa", has_splits=True)
    #downloader.download_huggingface_with_splits("FDlalala/tranS", has_splits=True)
    downloader.download_huggingface_with_splits("yairschiff/qm9", has_splits=True)
    gc.collect()
    downloader.download_huggingface_with_splits("MathArena/aime_2025", has_splits=True)
    # downloader.download_huggingface_with_splits("darius-tang/peg_in_hole", has_splits=True)
    #--
    # downloader.download_huggingface_with_splits("oolongbench/oolong-synth",has_splits=True)
    gc.collect()
    downloader.download_huggingface_with_splits("OpenAssistant/oasst1", has_splits=True)
    downloader.download_huggingface_with_splits("josancamon/paperbench",has_splits=True)
    downloader.download_huggingface_with_splits("jaredfern/codah", config="codah", has_splits=True)
    gc.collect()
    # downloader.download_huggingface_with_splits("nlerobot/pusht", has_splits=True)
    downloader.download_huggingface_with_splits("livebench/math", has_splits=True)
    downloader.download_huggingface_with_splits("zwhe99/amc23", has_splits=True)
    gc.collect()
    downloader.download_huggingface_with_splits("nlile/24-game", has_splits=True)
    downloader.download_huggingface_with_splits("ISdept/piper_arm", has_splits=True)
    downloader.download_huggingface_with_splits("alvations/c4p0", has_splits=True)
    gc.collect()
    #--
    
    print("\n--- OpenML Datasets (30) ---\n")
    #--
    downloader.download_openml_with_split(31, "credit_g")
    downloader.download_openml_with_split(1590, "adult")
    downloader.download_openml_with_split(554, "mnist_784")
    gc.collect()
    downloader.download_openml_with_split(40498, "wine_quality_white") 
    downloader.download_openml_with_split(1461, "bank_marketing") 
    downloader.download_openml_with_split(37, "diabetes")
    gc.collect()
    downloader.download_openml_with_split(40945, "titanic")   
    downloader.download_openml_with_split(44, "spam")                  
    downloader.download_openml_with_split(1489, "phoneme") 
    gc.collect()            
    downloader.download_openml_with_split(1464, "blood_transfusion") 
    #--
    downloader.download_openml_with_split(1120, "MagicTelescope")
    downloader.download_openml_with_split(1068, "pc1")
    gc.collect()
    downloader.download_openml_with_split(4134, "Bioresponse")
    downloader.download_openml_with_split(1510, "wdbc") 
    downloader.download_openml_with_split(57, "hypothyroid") 
    gc.collect()
    downloader.download_openml_with_split(534, "cps_85_wages")
    downloader.download_openml_with_split(43342, "German-House-Prices")   
    downloader.download_openml_with_split(46531, "dataset_china")   
    gc.collect()               
    downloader.download_openml_with_split(1104, "leukemia")             
    downloader.download_openml_with_split(42225, "diamonds")
    #--
    downloader.download_openml_with_split(44063, "Bike_Sharing_Demand")
    gc.collect()
    downloader.download_openml_with_split(43510, "UEFA-Champions-league-Player-Statistics")
    downloader.download_openml_with_split(50, "tic-tac-toe")
    downloader.download_openml_with_split(42, "soybean") 
    gc.collect()
    downloader.download_openml_with_split(3, "kr-vs-kp") 
    downloader.download_openml_with_split(334, "monks-problems-2")
    downloader.download_openml_with_split(54, "vehicle")   
    gc.collect()
    downloader.download_openml_with_split(2, "anneal")                  
    downloader.download_openml_with_split(23517, "numerai28.6")             
    downloader.download_openml_with_split(40672, "fars")
    gc.collect()
    #--
    
    print("\n--- UCI Datasets (30) ---\n")
    #--
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

 #--
    downloader.download_uci_with_split(
        "https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip",
        "parkinsons_telemonitoring",
        data_filename="parkinsons_updrs.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/183/communities%2Band%2Bcrime.zip",
        "communities_and_crime",
        data_filename="communities.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/144/statlog%2Bgerman%2Bcredit%2Bdata.zip",
        "german_credit",
        data_filename="german.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/9/auto%2Bmpg.zip",
        "auto_mpg",
        data_filename="auto-mpg.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/94/spambase.zip",
        "spambase",
        data_filename="spambase.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/159/magic%2Bgamma%2Btelescope.zip",
        "magic_gamma_telescope",
        data_filename="magic04.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/14/breast%2Bcancer.zip",
        "breast_cancer",
        data_filename="breast-cancer.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/27/credit%2Bapproval.zip",
        "credit_approval",
        data_filename="crx.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/20/census%2Bincome.zip",
        "census_income",
        data_filename="adult.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/102/thyroid%2Bdisease.zip",
        "thyroid_disease",
        data_filename="allbp.data"
    )
    #--

    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/102/thyroid%2Bdisease.zip",
        "thyroid_disease",
        data_filename="allhyper.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/102/thyroid%2Bdisease.zip",
        "thyroid_disease",
        data_filename="allhypo.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/102/thyroid%2Bdisease.zip",
        "thyroid_disease",
        data_filename="allrep.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/102/thyroid%2Bdisease.zip",
        "thyroid_disease",
        data_filename="dis.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/102/thyroid%2Bdisease.zip",
        "thyroid_disease",
        data_filename="hypothyroid.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/102/thyroid%2Bdisease.zip",
        "thyroid_disease",
        data_filename="new-thyroid.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/102/thyroid%2Bdisease.zip",
        "thyroid_disease",
        data_filename="sick.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/174/parkinsons.zip",
        "parkinsons",
        data_filename="parkinsons.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/42/glass%2Bidentification.zip",
        "glass_identification",
        data_filename="glass.data"
    )
    
    downloader.download_uci_with_split(
        "https://cdn.uci-ics-mlr-prod.aws.uci.edu/111/zoo.zip",
        "zoo",
        data_filename="zoo.data"
    )
    #--

    print("\nDownloaded:")
    print("30 HuggingFace datasets (text classification)")
    print("30 OpenML datasets (tabular)")
    print("30 UCI datasets (classic ML)")


if __name__ == "__main__":
    main()