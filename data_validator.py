import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
from collections import Counter, defaultdict
import re

@dataclass
class LintResult:
    feature_name: str
    lint_type: str
    severity: str
    description: str
    recommendation: str
    sample_values: List[Any]
    
@dataclass
class DatasetQualityReport:
    dataset_name: str
    source: str
    total_rows: int
    total_columns: int
    lints: List[LintResult]
    missing_rate: float
    duplicate_rate: float
    number_as_string_count: int
    enum_as_numeric_count: int
    unnormalized_feature_count: int
    tailed_distribution_count: int
    duplicate_rows_count: int
    empty_examples_count: int
    excessive_missing_count: int
    mixed_types_count: int
    constant_features_count: int
    duplicate_columns_count: int
    ambiguous_labels_count: int
    train_test_contamination: bool


class DataLoader:
    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
    
    def load_dataset(self, filepath: Path) -> pd.DataFrame:
        if filepath.suffix == '.csv':
            return pd.read_csv(filepath, low_memory=False)
        elif filepath.suffix == '.parquet':
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")
    
    def discover_datasets(self) -> List[tuple]:
        datasets = []
        for source in ['huggingface', 'openml', 'uci']:
            source_path = self.base_path / source
            if source_path.exists():
                for ext in ['*.csv', '*.parquet']:
                    datasets.extend([(source, f) for f in source_path.glob(ext)])
        return datasets


class DataQualityValidator:
    def __init__(self):
        self.missing_threshold = 0.5
        self.normalization_scale_factor = 1000
        
    def validate(self, df: pd.DataFrame, dataset_name: str, source: str) -> DatasetQualityReport:
        lints = []
        lints.extend(self._check_numbers_as_strings(df))
        lints.extend(self._check_enum_as_numeric(df))
        lints.extend(self._check_unnormalized_features(df))
        lints.extend(self._check_tailed_distributions(df))
        lints.extend(self._check_duplicates(df))
        lints.extend(self._check_empty_examples(df))
        lints.extend(self._check_missing_values(df))
        lints.extend(self._check_type_inconsistencies(df))
        lints.extend(self._check_constant_features(df))
        lints.extend(self._check_duplicate_columns(df))
        lints.extend(self._check_ambiguous_labels(df))
        
        missing_rate = df.isnull().mean().mean()
        duplicate_rate = df.duplicated().sum() / len(df)
        
        return DatasetQualityReport(
            dataset_name=dataset_name,
            source=source,
            total_rows=len(df),
            total_columns=len(df.columns),
            lints=lints,
            missing_rate=missing_rate,
            duplicate_rate=duplicate_rate,
            number_as_string_count=len([l for l in lints if l.lint_type == 'number_as_string']),
            enum_as_numeric_count=len([l for l in lints if l.lint_type == 'enum_as_numeric']),
            unnormalized_feature_count=len([l for l in lints if l.lint_type == 'unnormalized_feature']),
            tailed_distribution_count=len([l for l in lints if l.lint_type == 'tailed_distribution']),
            duplicate_rows_count=1 if any(l.lint_type == 'duplicate_rows' for l in lints) else 0,
            empty_examples_count=1 if any(l.lint_type == 'empty_examples' for l in lints) else 0,
            excessive_missing_count=len([l for l in lints if l.lint_type == 'excessive_missing']),
            mixed_types_count=len([l for l in lints if l.lint_type == 'mixed_types']),
            constant_features_count=len([l for l in lints if l.lint_type == 'constant_feature']),
            duplicate_columns_count=len([l for l in lints if l.lint_type == 'duplicate_columns']),
            ambiguous_labels_count=1 if any(l.lint_type == 'ambiguous_labels' for l in lints) else 0,
            train_test_contamination=any(l.lint_type == 'train_test_contamination' for l in lints),
        )

    def _check_numbers_as_strings(self, df: pd.DataFrame) -> List[LintResult]:
        lints = []
        for col in df.select_dtypes(include=['object']).columns:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            
            numeric_converted = pd.to_numeric(non_null, errors='coerce')
            numeric_ratio = numeric_converted.notna().sum() / len(non_null)
            
            if numeric_ratio > 0.8:
                lints.append(LintResult(
                    feature_name=col,
                    lint_type='number_as_string',
                    severity='warning',
                    description=f'{numeric_ratio*100:.1f}% of values are numeric but stored as strings',
                    recommendation='Convert to numeric type: pd.to_numeric(df[col])',
                    sample_values=self._safe_sample(non_null.head(3))
                ))
        return lints
    
    def _check_enum_as_numeric(self, df: pd.DataFrame) -> List[LintResult]:
        lints = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col.lower() in ['label', 'target', 'class', 'y', 'category']:
                continue
                
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            
            if unique_count < 10 and total_count > 50:
                is_all_int = df[col].dropna().apply(lambda x: x == int(x)).all()
                if is_all_int:
                    lints.append(LintResult(
                        feature_name=col,
                        lint_type='enum_as_numeric',
                        severity='warning',
                        description=f'Only {unique_count} unique integer values - likely categorical',
                        recommendation='Consider using embeddings or one-hot encoding',
                        sample_values=self._safe_sample(sorted(df[col].dropna().unique())[:5])
                    ))
        return lints
    
    def _check_unnormalized_features(self, df: pd.DataFrame) -> List[LintResult]:
        lints = []
        for col in df.select_dtypes(include=[np.number]).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max == col_min:
                continue
            
            if abs(col_max - col_min) > self.normalization_scale_factor:
                lints.append(LintResult(
                    feature_name=col,
                    lint_type='unnormalized_feature',
                    severity='info',
                    description=f'Wide range: [{col_min:.2f}, {col_max:.2f}]',
                    recommendation='Consider normalization or standardization',
                    sample_values=self._safe_sample(df[col].dropna().head(2))
                ))
        return lints
    
    def _check_tailed_distributions(self, df: pd.DataFrame) -> List[LintResult]:
        lints = []
        for col in df.select_dtypes(include=[np.number]).columns:
            skewness = df[col].skew()
            if abs(skewness) > 3:
                lints.append(LintResult(
                    feature_name=col,
                    lint_type='tailed_distribution',
                    severity='info',
                    description=f'High skewness: {skewness:.2f}',
                    recommendation='Consider log transform or remove outliers',
                    sample_values=[df[col].min(), df[col].median(), df[col].max()]
                ))
        return lints

    def _check_duplicates(self, df: pd.DataFrame) -> List[LintResult]:
        lints = []
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            dup_ratio = dup_count / len(df)
            lints.append(LintResult(
                feature_name='__all__',
                lint_type='duplicate_rows',
                severity='error' if dup_ratio > 0.05 else 'warning',
                description=f'{dup_count} duplicate rows ({dup_ratio*100:.2f}%)',
                recommendation='Remove duplicates: df.drop_duplicates()',
                sample_values=[f'{dup_count} duplicates found']
            ))
        return lints
    
    def _check_empty_examples(self, df: pd.DataFrame) -> List[LintResult]:
        lints = []
        empty_count = df.isnull().all(axis=1).sum()
        if empty_count > 0:
            lints.append(LintResult(
                feature_name='__all__',
                lint_type='empty_examples',
                severity='error',
                description=f'{empty_count} completely empty rows',
                recommendation='Remove empty rows: df.dropna(how="all")',
                sample_values=[f'{empty_count} empty rows']
            ))
        return lints
    
    def _check_missing_values(self, df: pd.DataFrame) -> List[LintResult]:
        lints = []
        for col in df.columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > self.missing_threshold:
                lints.append(LintResult(
                    feature_name=col,
                    lint_type='excessive_missing',
                    severity='error',
                    description=f'{missing_ratio*100:.1f}% missing values',
                    recommendation='Consider removing feature or imputing',
                    sample_values=[f'{missing_ratio*100:.1f}% missing']
                ))
        return lints
    
    def _check_type_inconsistencies(self, df: pd.DataFrame) -> List[LintResult]:
        lints = []
        for col in df.columns:
            type_counts = Counter(type(x).__name__ for x in df[col].dropna())
            problematic = any(t in type_counts for t in ['str', 'bool']) and \
                         any(t in type_counts for t in ['int', 'float'])
            if problematic:
                lints.append(LintResult(
                    feature_name=col,
                    lint_type='mixed_types',
                    severity='error',
                    description=f'Multiple types detected: {dict(type_counts)}',
                    recommendation='Standardize to single type',
                    sample_values=list(dict(type_counts).keys())
                ))
        return lints
    
    def _check_constant_features(self, df: pd.DataFrame) -> List[LintResult]:
        lints = []
        for col in df.columns:
            if df[col].nunique() == 1:
                lints.append(LintResult(
                    feature_name=col,
                    lint_type='constant_feature',
                    severity='error',
                    description='Zero variance feature',
                    recommendation='Remove feature: df.drop(columns=[col])',
                    sample_values=[df[col].iloc[0]]
                ))
        return lints
    
    def _check_duplicate_columns(self, df: pd.DataFrame) -> List[LintResult]:
        lints = []
        checked = set()
        for i, col1 in enumerate(df.columns):
            if col1 in checked:
                continue
            for col2 in df.columns[i+1:]:
                if col2 in checked:
                    continue
                if df[col1].equals(df[col2]):
                    lints.append(LintResult(
                        feature_name=f'{col1}, {col2}',
                        lint_type='duplicate_columns',
                        severity='info',
                        description=f'Identical columns',
                        recommendation='Investigate if intentional or accidental',
                        sample_values=[f'{col1} == {col2}']
                    ))
                    checked.add(col2)
        return lints
    
    def _check_ambiguous_labels(self, df: pd.DataFrame) -> List[LintResult]:
        lints = []
        text_cols = []
        label_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['text', 'sentence', 'review', 'comment', 'question', 'content']):
                text_cols.append(col)
            if col_lower in ['label', 'target', 'class', 'category']:
                label_col = col
        
        if not text_cols or not label_col:
            return lints
        
        for text_col in text_cols:
            grouped = df.groupby(text_col)[label_col].nunique()
            ambiguous = grouped[grouped > 1]
            
            if len(ambiguous) > 0:
                examples = []
                for text in ambiguous.index[:3]:
                    labels = df[df[text_col] == text][label_col].unique()
                    text_preview = str(text)[:50] + "..." if len(str(text)) > 50 else str(text)
                    examples.append(f'"{text_preview}" → labels: {sorted(labels.tolist())}')
                
                lints.append(LintResult(
                    feature_name=f'{text_col}→{label_col}',
                    lint_type='ambiguous_labels',
                    severity='error',
                    description=f'{len(ambiguous)} text samples have multiple labels',
                    recommendation='Investigate data labeling process',
                    sample_values=examples
                ))
        return lints
    
    def check_train_test_contamination(self, train_path: Path, test_path: Path) -> LintResult:
        train_df = pd.read_parquet(train_path) if train_path.suffix == '.parquet' else pd.read_csv(train_path)
        test_df = pd.read_parquet(test_path) if test_path.suffix == '.parquet' else pd.read_csv(test_path)
        
        train_set = set(train_df.itertuples(index=False, name=None))
        test_set = set(test_df.itertuples(index=False, name=None))
        
        overlap = train_set & test_set
        contamination_rate = len(overlap) / len(test_set) if len(test_set) > 0 else 0
        
        if contamination_rate == 0:
            return None
            
        if contamination_rate < 0.005:
            severity = 'info'
            description = f'{contamination_rate*100:.2f}% test set overlap (negligible)'
            recommendation = 'Low contamination typical of duplicate rows'
        elif contamination_rate < 0.05:
            severity = 'warning'
            description = f'{contamination_rate*100:.1f}% test set overlap (minor)'
            recommendation = 'Deduplicate source data and regenerate splits'
        elif contamination_rate < 0.2:
            severity = 'error'
            description = f'{contamination_rate*100:.1f}% test set overlap (moderate)'
            recommendation = 'Model evaluation metrics unreliable - deduplicate immediately'
        else:
            severity = 'error'
            description = f'{contamination_rate*100:.1f}% test set overlap (severe)'
            recommendation = 'Model evaluation invalid - stop use until fixed'
            
        return LintResult(
            feature_name='__train_test_split__',
            lint_type='train_test_contamination',
            severity=severity,
            description=description,
            recommendation=recommendation,
            sample_values=[
                f'{len(overlap)} contaminated of {len(test_set)} test examples',
                f'Train: {len(train_df)} rows, Test: {len(test_df)} rows'
            ]
        )
    
    def _safe_sample(self, values, n=5):
        try:
            if hasattr(values, 'tolist'):
                return values[:n].tolist()
            elif isinstance(values, (list, np.ndarray)):
                return list(values[:n])
            else:
                return [str(v) for v in list(values)[:n]]
        except:
            return ["<unable to display>"]


class ReportGenerator:
    def print_report(self, report: DatasetQualityReport):
        print(f"{report.source}/{report.dataset_name}: {report.total_rows}x{report.total_columns}")
        
        if not report.lints:
            print("✓ No issues")
            return
        
        errors = [l for l in report.lints if l.severity == 'error']
        warnings = [l for l in report.lints if l.severity == 'warning']
        
        if errors:
            print(f"  {len(errors)} errors")
        if warnings:
            print(f"  {len(warnings)} warnings")
    
    def generate_summary_csv(self, reports: List[DatasetQualityReport], output_path: str):
        summary_data = []
        for report in reports:
            row = {
                'dataset': report.dataset_name,
                'source': report.source,
                'rows': report.total_rows,
                'columns': report.total_columns,
                'missing_rate_pct': round(report.missing_rate * 100, 2),
                'duplicate_rate_pct': round(report.duplicate_rate * 100, 2),
                'number_as_string': report.number_as_string_count,
                'enum_as_numeric': report.enum_as_numeric_count,
                'unnormalized_features': report.unnormalized_feature_count,
                'tailed_distributions': report.tailed_distribution_count,
                'has_duplicate_rows': report.duplicate_rows_count,
                'has_empty_rows': report.empty_examples_count,
                'excessive_missing_cols': report.excessive_missing_count,
                'mixed_types_cols': report.mixed_types_count,
                'constant_features': report.constant_features_count,
                'duplicate_columns': report.duplicate_columns_count,
                'has_ambiguous_labels': report.ambiguous_labels_count,
                'has_train_test_contamination': 1 if report.train_test_contamination else 0,
            }
            summary_data.append(row)
        
        pd.DataFrame(summary_data).to_csv(output_path, index=False)


def find_train_test_pairs(datasets: List[tuple]) -> List[tuple]:
    pairs = []
    by_basename = defaultdict(dict)
    
    for source, filepath in datasets:
        name = filepath.stem
        if name.endswith('_train'):
            basename = name[:-6]
            by_basename[f"{source}/{basename}"]['train'] = (source, filepath)
        elif name.endswith('_test'):
            basename = name[:-5]
            by_basename[f"{source}/{basename}"]['test'] = (source, filepath)
    
    for key, files in by_basename.items():
        if 'train' in files and 'test' in files:
            pairs.append((files['train'], files['test']))
    
    return pairs


def main():
    loader = DataLoader(base_path="./data")
    validator = DataQualityValidator()
    reporter = ReportGenerator()
    
    datasets = loader.discover_datasets()
    train_test_pairs = find_train_test_pairs(datasets)
    
    # Get datasets that are NOT part of train/test pairs
    paired_files = set()
    for (_, train_path), (_, test_path) in train_test_pairs:
        paired_files.add(train_path)
        paired_files.add(test_path)
    
    standalone_datasets = [(s, f) for s, f in datasets if f not in paired_files]
    
    print(f"Found {len(train_test_pairs)} train/test pairs and {len(standalone_datasets)} standalone datasets\n")
    
    all_reports = []
    contamination_results = []
    
    # Check train/test pairs
    print("Validating train/test pairs...")
    for (train_source, train_path), (test_source, test_path) in train_test_pairs:
        base_name = train_path.stem.replace('_train', '')
        print(f"Checking {train_source}/{base_name}...")
        
        try:
            # Only validate the training set for data quality
            train_df = loader.load_dataset(train_path)
            report = validator.validate(train_df, base_name, train_source)
            
            # Check contamination
            contamination_lint = validator.check_train_test_contamination(train_path, test_path)
            if contamination_lint:
                report.lints.append(contamination_lint)
                report.train_test_contamination = True
                contamination_results.append({
                    'dataset': base_name,
                    'source': train_source,
                    'contamination_pct': float(re.search(r'([\d.]+)%', contamination_lint.description).group(1))
                })
            
            reporter.print_report(report)
            all_reports.append(report)
            
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
    
    # Validate standalone datasets
    if standalone_datasets:
        print("\nValidating standalone datasets...")
        for source, filepath in standalone_datasets:
            print(f"Checking {source}/{filepath.stem}...")
            try:
                df = loader.load_dataset(filepath)
                report = validator.validate(df, filepath.stem, source)
                reporter.print_report(report)
                all_reports.append(report)
            except Exception as e:
                print(f"Error processing {filepath.stem}: {e}")
    
    # Save results
    print("\nSaving results...")
    reporter.generate_summary_csv(all_reports, "quality_summary.csv")
    
    if contamination_results:
        pd.DataFrame(contamination_results).to_csv('contamination_details.csv', index=False)
    
    print("Done!")


if __name__ == "__main__":
    main()