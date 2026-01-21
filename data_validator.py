import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
from collections import Counter
import re

@dataclass
class LintResult:
    feature_name: str
    lint_type: str
    severity: str  # 'error', 'warning', 'info'
    description: str
    recommendation: str
    sample_values: List[Any]
    
@dataclass
class DatasetQualityReport:
    """Complete quality report for one dataset"""
    dataset_name: str
    source: str
    total_rows: int
    total_columns: int
    
    # All detected lints
    lints: List[LintResult]
    
    # Summary statistics
    missing_rate: float
    duplicate_rate: float
    constant_features_count: int

class DataLoader:
    """Handle loading CSV and Parquet from local folders"""
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
        """
        Find all datasets in folder structure
        Returns: List of (source, filepath) tuples
        """
        datasets = []
        for source in ['huggingface', 'openml', 'uci']:
            source_path = self.base_path / source
            if source_path.exists():
                for ext in ['*.csv', '*.parquet']:
                    datasets.extend([
                        (source, f) for f in source_path.glob(ext)
                    ])
        return datasets

class DataQualityValidator:
    """
    Main validation engine
    """
    def __init__(self):
        # Thresholds configuration
        self.missing_threshold = 0.5
        self.normalization_scale_factor = 1000
    
        
    def validate(self, df: pd.DataFrame, dataset_name: str, source: str) -> DatasetQualityReport:
        """
        Run all validation checks on a dataset
        """
        lints = []
        # Miscoding Lints
        lints.extend(self._check_numbers_as_strings(df))
        lints.extend(self._check_enum_as_numeric(df))
        
        # Scaling/Outlier Lints
        lints.extend(self._check_unnormalized_features(df))
        lints.extend(self._check_tailed_distributions(df))
        
        # Packaging Lints
        lints.extend(self._check_duplicates(df))
        lints.extend(self._check_empty_examples(df))
        
        # TensorFlow single-batch checks
        lints.extend(self._check_missing_values(df))
        lints.extend(self._check_type_inconsistencies(df))
        lints.extend(self._check_constant_features(df))
        lints.extend(self._check_duplicate_columns(df))
        
        # gzip-knn-paper issue
        lints.extend(self._check_ambiguous_labels(df))
        
        # Calculate summary metrics
        missing_rate = df.isnull().mean().mean()
        duplicate_rate = df.duplicated().sum() / len(df)
        constant_count = len([l for l in lints if l.lint_type == 'constant_feature'])
        
        return DatasetQualityReport(
            dataset_name=dataset_name,
            source=source,
            total_rows=len(df),
            total_columns=len(df.columns),
            lints=lints,
            missing_rate=missing_rate,
            duplicate_rate=duplicate_rate,
            constant_features_count=constant_count,
        )

    def _check_numbers_as_strings(self, df: pd.DataFrame) -> List[LintResult]:
        """
        CHECK: Numbers stored as strings
        DETECTS: "123", "45.6" in string columns
        RECOMMENDATION: Convert to numeric type
        """
        lints = []
        
        for col in df.select_dtypes(include=['object']).columns:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            
            # Try converting to numeric
            numeric_converted = pd.to_numeric(non_null, errors='coerce')
            numeric_ratio = numeric_converted.notna().sum() / len(non_null)
            
            # If >80% can be converted to numbers, guessing that it is probably miscoded
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
        """
        CHECK: Categorical values stored as integers/floats
        DETECTS: Small set of integer values that are actually categories
        RECOMMENDATION: Use embeddings or one-hot encoding
        """
        lints = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            # Auto check common label names to skip and not warn about a bunch of 0/1 labels
            if col_lower in ['label', 'target', 'class', 'y', 'category']:
                continue
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            
            # If low cardinality + all integers, likely categorical
            if unique_count < 10 and total_count > 50:
                # Check if all values are integers (even if stored as float)
                is_all_int = df[col].dropna().apply(lambda x: x == int(x)).all()
                
                if is_all_int:
                    lints.append(LintResult(
                        feature_name=col,
                        lint_type='enum_as_numeric',
                        severity='warning',
                        description=f'Only {unique_count} unique integer values - likely categorical',
                        recommendation='Consider using embeddings or one-hot encoding instead of raw integers',
                        sample_values=self._safe_sample(sorted(df[col].dropna().unique())[:5])
                    ))
        return lints
    
    def _check_unnormalized_features(self, df: pd.DataFrame) -> List[LintResult]:
        """
        CHECK: Features on widely different scales
        DETECTS: Large range or very different min/max across features
        RECOMMENDATION: Normalize or standardize
        """
        lints = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_max == col_min:
                continue
            
            # Check if scale is very large
            if abs(col_max - col_min) > self.normalization_scale_factor:
                lints.append(LintResult(
                    feature_name=col,
                    lint_type='unnormalized_feature',
                    severity='info',
                    description=f'Wide range: [{col_min:.2f}, {col_max:.2f}]',
                    recommendation='For neural networks, consider normalization: (x - min) / (max - min) or standardization: (x - mean) / std',
                    sample_values=self._safe_sample(df[col].dropna().head(2))
                ))
        return lints
    
    def _check_tailed_distributions(self, df: pd.DataFrame) -> List[LintResult]:
        """
        CHECK: Extreme outliers affecting distribution
        DETECTS: High skewness
        RECOMMENDATION: Log transform or remove outliers
        """
        lints = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            skewness = df[col].skew()
            
            # Highly skewed distributions
            if abs(skewness) > 3:
                lints.append(LintResult(
                    feature_name=col,
                    lint_type='tailed_distribution',
                    severity='info',
                    description=f'High skewness: {skewness:.2f}',
                    recommendation='Consider log transform: np.log1p(x) or remove outliers',
                    sample_values=[df[col].min(), df[col].median(), df[col].max()]
                ))
        
        return lints

    def _check_duplicates(self, df: pd.DataFrame) -> List[LintResult]:
        """
        CHECK: Duplicate rows
        DETECTS: Identical rows
        RECOMMENDATION: Remove duplicates
        """
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
        """
        CHECK: Completely empty rows
        DETECTS: Rows where all values are NaN
        RECOMMENDATION: Remove empty rows
        """
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
        """
        CHECK: Missing value detection
        DETECTS: High percentage of NaN/NULL values
        RECOMMENDATION: Impute or remove feature
        """
        lints = []
        
        for col in df.columns:
            missing_ratio = df[col].isnull().sum() / len(df)
            
            if missing_ratio > self.missing_threshold:
                lints.append(LintResult(
                    feature_name=col,
                    lint_type='excessive_missing',
                    severity='error',
                    description=f'{missing_ratio*100:.1f}% missing values (>{self.missing_threshold*100}%)',
                    recommendation='Consider removing feature or imputing: df[col].fillna(df[col].mean())',
                    sample_values=[f'{missing_ratio*100:.1f}% missing']
                ))
        return lints
    
    def _check_type_inconsistencies(self, df: pd.DataFrame) -> List[LintResult]:
        """
        CHECK: Mixed types within a column
        DETECTS: Same column having int, float, string values
        RECOMMENDATION: Standardize type
        """
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
                    recommendation='Standardize to single type or investigate data corruption',
                    sample_values=list(dict(type_counts).keys())
                ))
        return lints
    
    def _check_constant_features(self, df: pd.DataFrame) -> List[LintResult]:
        """
        CHECK: Zero-variance features
        DETECTS: Columns with only one unique value
        RECOMMENDATION: Remove feature (useless for ML)
        """
        lints = []
        
        for col in df.columns:
            if df[col].nunique() == 1:
                lints.append(LintResult(
                    feature_name=col,
                    lint_type='constant_feature',
                    severity='error',
                    description='Only one unique value (zero variance)',
                    recommendation='Remove feature: df.drop(columns=[col])',
                    sample_values=[df[col].iloc[0]]
                ))
        return lints
    
    def _check_duplicate_columns(self, df: pd.DataFrame) -> List[LintResult]:
        """
        CHECK: Duplicate columns (identical values)
        DETECTS: Different column names but same values
        RECOMMENDATION: Remove duplicates
        """
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
                        description=f'Columns "{col1}" and "{col2}" are identical',
                        recommendation='Investigate if intentional (border pixels, symmetry) or accidental',
                        sample_values=[f'{col1} == {col2}']
                    ))
                    checked.add(col2)
        return lints
    
    def _safe_sample(self, values, n=5):
        """Safely convert values to list of samples"""
        try:
            if hasattr(values, 'tolist'):
                return values[:n].tolist()
            elif isinstance(values, (list, np.ndarray)):
                return list(values[:n])
            else:
                return [str(v) for v in list(values)[:n]]
        except:
            return ["<unable to display>"]

    def check_train_test_contamination(self, 
                                   train_path: Path, 
                                   test_path: Path) -> LintResult:
        """
        CHECK: Train/test overlap
        DETECTS: Test examples that appear in training set  
        RETURNS: Contamination rate with graduated severity
        """
        # Load both datasets
        train_df = pd.read_parquet(train_path) if train_path.suffix == '.parquet' else pd.read_csv(train_path)
        test_df = pd.read_parquet(test_path) if test_path.suffix == '.parquet' else pd.read_csv(test_path)
        
        # Convert to hashable tuples
        train_set = set(train_df.itertuples(index=False, name=None))
        test_set = set(test_df.itertuples(index=False, name=None))
        
        overlap = train_set & test_set
        contamination_rate = len(overlap) / len(test_set) if len(test_set) > 0 else 0
        
        if contamination_rate == 0:
            return None
        # Graduated severity based on research and real-world impact
        if contamination_rate < 0.005:
            severity = 'info'
            description = f'{contamination_rate*100:.2f}% test set overlap (negligible - likely duplicate rows)'
            recommendation = 'Low contamination typical of duplicate rows in source data. Consider deduplicating source before splitting.'
            
        elif contamination_rate < 0.05:
            severity = 'warning'
            description = f'{contamination_rate*100:.1f}% test set overlap (minor contamination)'
            recommendation = 'Small but noticeable contamination. Deduplicate source data and regenerate splits to ensure valid evaluation.'
        elif contamination_rate < 0.2:
            severity = 'error'
            description = f'{contamination_rate*100:.1f}% test set overlap (moderate contamination)'
            recommendation = 'MODERATE contamination - model evaluation metrics are unreliable. Critical: Deduplicate and regenerate splits immediately.'
        else:
            severity = 'error'
            description = f'{contamination_rate*100:.1f}% test set overlap (severe contamination)'
            recommendation = 'SEVERE contamination - model evaluation is invalid. CRITICAL: Stop all use of this dataset until deduplication and proper splitting is performed.'
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

    def _check_ambiguous_labels(self, df: pd.DataFrame) -> List[LintResult]:
        """
        CHECK: Same input text with different labels
        DETECTS: Data labeling errors or collection issues
        RETURNS: Examples of ambiguous text
        """
        lints = []
        
        # Try to detect text and label columns
        text_cols = []
        label_col = None
        
        # Look for common text column names
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['text', 'sentence', 'review', 'comment', 'question', 'content']):
                text_cols.append(col)
            if col_lower in ['label', 'target', 'class', 'category']:
                label_col = col
        
        # If we can't find obvious columns, skip this check
        if not text_cols or not label_col:
            return lints
        
        # Check each text column
        for text_col in text_cols:
            # Group by text, count unique labels
            grouped = df.groupby(text_col)[label_col].nunique()
            ambiguous = grouped[grouped > 1]
            
            if len(ambiguous) > 0:
                # Get specific examples
                examples = []
                for text in ambiguous.index[:3]:  # Show up to 3 examples
                    labels = df[df[text_col] == text][label_col].unique()
                    # Truncate long text
                    text_preview = str(text)[:50] + "..." if len(str(text)) > 50 else str(text)
                    examples.append(f'"{text_preview}" → labels: {sorted(labels.tolist())}')
                
                lints.append(LintResult(
                    feature_name=f'{text_col}→{label_col}',
                    lint_type='ambiguous_labels',
                    severity='error',
                    description=f'{len(ambiguous)} text samples have multiple different labels',
                    recommendation='CRITICAL: Investigate data labeling process - same input should not have different outputs',
                    sample_values=examples
                ))
        return lints

class ReportGenerator:
    """Generate readable reports"""
    
    def print_report(self, report: DatasetQualityReport):
        print(f"Dataset: {report.dataset_name}")
        print(f"Source: {report.source}")
        print(f"Shape: {report.total_rows} rows x {report.total_columns} columns")
        
        if not report.lints:
            print("No issues found!")
            return
        
        # Group by severity
        errors = [l for l in report.lints if l.severity == 'error']
        warnings = [l for l in report.lints if l.severity == 'warning']
        infos = [l for l in report.lints if l.severity == 'info']
        
        if errors:
            print(f"ERRORS ({len(errors)}):")
            for lint in errors:
                self._print_lint(lint)
        
        if warnings:
            print(f"\nWARNINGS ({len(warnings)}):")
            for lint in warnings:
                self._print_lint(lint)
        
        if infos:
            print(f"\nINFO ({len(infos)}):")
            for lint in infos:
                self._print_lint(lint)
    
    def _print_lint(self, lint: LintResult):
        print(f"[{lint.lint_type}] {lint.feature_name}")
        print(f"Issue: {lint.description}")
        print(f"Fix: {lint.recommendation}")
        print(f"Examples: {lint.sample_values}")
    
    def generate_summary_csv(self, reports: List[DatasetQualityReport], output_path: str):
        """
        Generate CSV summary
        """
        summary_data = []
        
        for report in reports:
            metrics = self._extract_lint_metrics(report.lints)
            
            row = {
                # Basic info
                'dataset': report.dataset_name,
                'source': report.source,
                'rows': report.total_rows,
                'columns': report.total_columns,
                
                # Error counts by severity
                'errors': len([l for l in report.lints if l.severity == 'error']),
                'warnings': len([l for l in report.lints if l.severity == 'warning']),
                'infos': len([l for l in report.lints if l.severity == 'info']),
                
                # Basic metrics
                'missing_rate_pct': round(report.missing_rate * 100, 2),
                'duplicate_rate_pct': round(report.duplicate_rate * 100, 2),
                
                # Critical issues
                'has_contamination': 1 if metrics.get('train_test_contamination_pct', 0) > 0 else 0,
                'contamination_pct': metrics.get('train_test_contamination_pct', 0),
                'has_ambiguous_labels': 1 if metrics.get('ambiguous_labels_count', 0) > 0 else 0,
                'ambiguous_labels_count': metrics.get('ambiguous_labels_count', 0),
                
                # Data quality issues
                'constant_features': metrics.get('constant_features_count', 0),
                'excessive_missing_columns': metrics.get('excessive_missing_columns', 0),
                'empty_rows': metrics.get('empty_rows_count', 0),
                'duplicate_columns': metrics.get('duplicate_columns_count', 0),
                'mixed_types_columns': metrics.get('mixed_types_count', 0),
                
                # Representation issues
                'enum_as_numeric_features': metrics.get('enum_as_numeric_count', 0),
                'unnormalized_features': metrics.get('unnormalized_count', 0),
            }
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
        print(f"\nDetailed summary saved to {output_path}")
    
    def _extract_lint_metrics(self, lints: List[LintResult]) -> Dict[str, Any]:
        """
        Extract quantitative metrics from lints
        Returns dict with specific measurements for each lint type
        
        Parses lint descriptions to extract concrete numbers like:
        - Contamination percentage
        - Number of ambiguous texts
        - Counts of problematic features
        """
        metrics = {}
        
        for lint in lints:
            lint_type = lint.lint_type
            
            # Train/test contamination - extract percentage
            if lint_type == 'train_test_contamination':
                match = re.search(r'([\d.]+)%', lint.description)
                if match:
                    metrics['train_test_contamination_pct'] = float(match.group(1))
            
            # Ambiguous labels - count how many texts are ambiguous
            elif lint_type == 'ambiguous_labels':
                match = re.search(r'(\d+) text samples', lint.description)
                if match:
                    metrics['ambiguous_labels_count'] = int(match.group(1))
            
            # Constant features - count them
            elif lint_type == 'constant_feature':
                metrics['constant_features_count'] = metrics.get('constant_features_count', 0) + 1
            
            # Excessive missing - count columns
            elif lint_type == 'excessive_missing':
                metrics['excessive_missing_columns'] = metrics.get('excessive_missing_columns', 0) + 1
            
            # Duplicate columns - count pairs
            elif lint_type == 'duplicate_columns':
                metrics['duplicate_columns_count'] = metrics.get('duplicate_columns_count', 0) + 1
            
            # Empty rows - extract count
            elif lint_type == 'empty_examples':
                match = re.search(r'(\d+) completely empty', lint.description)
                if match:
                    metrics['empty_rows_count'] = int(match.group(1))
            
            # Duplicate rows - extract count
            elif lint_type == 'duplicate_rows':
                match = re.search(r'(\d+) duplicate rows', lint.description)
                if match:
                    metrics['duplicate_rows_count'] = int(match.group(1))
            
            # Mixed types - count features
            elif lint_type == 'mixed_types':
                metrics['mixed_types_count'] = metrics.get('mixed_types_count', 0) + 1
            
            # Enum as numeric - count features
            elif lint_type == 'enum_as_numeric':
                metrics['enum_as_numeric_count'] = metrics.get('enum_as_numeric_count', 0) + 1
            
            # Numbers as strings - count features
            elif lint_type == 'number_as_string':
                metrics['numbers_as_strings_count'] = metrics.get('numbers_as_strings_count', 0) + 1
            
            # Unnormalized - count features
            elif lint_type == 'unnormalized_feature':
                metrics['unnormalized_count'] = metrics.get('unnormalized_count', 0) + 1
            
            # Tailed distribution - count features
            elif lint_type == 'tailed_distribution':
                metrics['tailed_distribution_count'] = metrics.get('tailed_distribution_count', 0) + 1
            
        return metrics
    
    def generate_repository_comparison(self, reports: List[DatasetQualityReport], output_path: str = "repository_comparison.csv"):
        """
        Compare repositories by error frequency and percentage
        Shows: Which repos have which problems and how often
        """
        from collections import defaultdict
        
        by_source = defaultdict(list)
        for report in reports:
            by_source[report.source].append(report)
        
        comparison = []
        
        for source, source_reports in by_source.items():
            num_datasets = len(source_reports)
            
            # Count datasets with each error type
            error_stats = {}
            
            # Critical errors
            contamination = [r for r in source_reports if any(l.lint_type == 'train_test_contamination' for l in r.lints)]
            error_stats['contamination_count'] = len(contamination)
            error_stats['contamination_pct'] = round(len(contamination) / num_datasets * 100, 1)
            
            ambiguous = [r for r in source_reports if any(l.lint_type == 'ambiguous_labels' for l in r.lints)]
            error_stats['ambiguous_labels_count'] = len(ambiguous)
            error_stats['ambiguous_labels_pct'] = round(len(ambiguous) / num_datasets * 100, 1)
            
            # Data quality errors
            constant = [r for r in source_reports if any(l.lint_type == 'constant_feature' for l in r.lints)]
            error_stats['constant_features_count'] = len(constant)
            error_stats['constant_features_pct'] = round(len(constant) / num_datasets * 100, 1)
            
            excessive_missing = [r for r in source_reports if any(l.lint_type == 'excessive_missing' for l in r.lints)]
            error_stats['excessive_missing_count'] = len(excessive_missing)
            error_stats['excessive_missing_pct'] = round(len(excessive_missing) / num_datasets * 100, 1)
            
            empty_rows = [r for r in source_reports if any(l.lint_type == 'empty_examples' for l in r.lints)]
            error_stats['empty_rows_count'] = len(empty_rows)
            error_stats['empty_rows_pct'] = round(len(empty_rows) / num_datasets * 100, 1)
            
            high_duplicates = [r for r in source_reports if any(l.lint_type == 'duplicate_rows' and l.severity == 'error' for l in r.lints)]
            error_stats['high_duplicates_count'] = len(high_duplicates)
            error_stats['high_duplicates_pct'] = round(len(high_duplicates) / num_datasets * 100, 1)
            
            mixed_types = [r for r in source_reports if any(l.lint_type == 'mixed_types' for l in r.lints)]
            error_stats['mixed_types_count'] = len(mixed_types)
            error_stats['mixed_types_pct'] = round(len(mixed_types) / num_datasets * 100, 1)
            
            dup_columns = [r for r in source_reports if any(l.lint_type == 'duplicate_columns' for l in r.lints)]
            error_stats['duplicate_columns_count'] = len(dup_columns)
            error_stats['duplicate_columns_pct'] = round(len(dup_columns) / num_datasets * 100, 1)
            
            # Format/representation warnings (less critical)
            enum_numeric = [r for r in source_reports if any(l.lint_type == 'enum_as_numeric' for l in r.lints)]
            error_stats['enum_as_numeric_count'] = len(enum_numeric)
            error_stats['enum_as_numeric_pct'] = round(len(enum_numeric) / num_datasets * 100, 1)
            
            unnormalized = [r for r in source_reports if any(l.lint_type == 'unnormalized_feature' for l in r.lints)]
            error_stats['unnormalized_count'] = len(unnormalized)
            error_stats['unnormalized_pct'] = round(len(unnormalized) / num_datasets * 100, 1)
            
            # Calculate average metrics
            avg_missing = np.mean([r.missing_rate for r in source_reports]) * 100
            avg_duplicates = np.mean([r.duplicate_rate for r in source_reports]) * 100
            
            comparison.append({
                'repository': source,
                'num_datasets': num_datasets,
                
                # Averages
                'avg_missing_rate_pct': round(avg_missing, 2),
                'avg_duplicate_rate_pct': round(avg_duplicates, 2),
                
                # Critical errors (count + %)
                **error_stats
            })
        
        df = pd.DataFrame(comparison)
        
        # Sort by most problematic (most critical errors)
        df['critical_score'] = (
            df['contamination_count'] * 10 +  # Most critical
            df['ambiguous_labels_count'] * 10 +
            df['constant_features_count'] * 5 +
            df['excessive_missing_count'] * 3 +
            df['high_duplicates_count'] * 3
        )
        df = df.sort_values('critical_score', ascending=True)
        df = df.drop('critical_score', axis=1)
        
        df.to_csv(output_path, index=False)
        print(f"Repository comparison saved to {output_path}")
        
        return df

def main():
    """
    Main execution: validate all datasets and generate reports
    """
    # Initialize components
    loader = DataLoader(base_path="./data")
    validator = DataQualityValidator()
    reporter = ReportGenerator()
    
    # Discover all datasets
    datasets = loader.discover_datasets()
    print(f"Found {len(datasets)} datasets to validate\n")
    
    all_reports = []
    contamination_results = []
    train_test_pairs = find_train_test_pairs(datasets)
    
    for (train_source, train_path), (test_source, test_path) in train_test_pairs:
        try:
            contamination_lint = validator.check_train_test_contamination(
                train_path, test_path
            )
            
            if contamination_lint:
                contamination_results.append({
                    'dataset': train_path.stem.replace('_train', ''),
                    'source': train_source,
                    'lint': contamination_lint
                })
                print(f"!{contamination_lint.description}!")
        
        except Exception as e:
            print(f"Error checking contamination: {e}")
    
    for source, filepath in datasets:
        print("="*40)
        print(f"Processing: {source}/{filepath.stem}...")
        print("="*40)
        try:
            df = loader.load_dataset(filepath)
            
            report = validator.validate(
                df=df,
                dataset_name=filepath.stem,
                source=source
            )
            
            # ADD contamination lint if this is a train or test file
            for contam in contamination_results:
                if filepath.stem.startswith(contam['dataset']):
                    report.lints.append(contam['lint'])
            
            reporter.print_report(report)
            all_reports.append(report)
            
        except Exception as e:
            print(f"ERROR processing {filepath}: {e}\n")
    
    # Generate summary
    reporter.generate_summary_csv(all_reports, "quality_summary.csv")
    reporter.generate_repository_comparison(all_reports, "repository_comparison.csv")
    
    #Generate contamination report
    if contamination_results:
        print("TRAIN/TEST CONTAMINATION SUMMARY")
        for result in contamination_results:
            print(f"{result['source']}/{result['dataset']}: {result['lint'].description}")


def find_train_test_pairs(datasets: List[tuple]) -> List[tuple]:
    """
    Find matching train/test file pairs
    Returns: List of ((train_source, train_path), (test_source, test_path))
    """
    pairs = []
    
    # Group by base name
    from collections import defaultdict
    by_basename = defaultdict(dict)
    
    for source, filepath in datasets:
        name = filepath.stem
        
        if name.endswith('_train'):
            basename = name[:-6]
            by_basename[f"{source}/{basename}"]['train'] = (source, filepath)
        elif name.endswith('_test'):
            basename = name[:-5]
            by_basename[f"{source}/{basename}"]['test'] = (source, filepath)
    
    # Match pairs
    for key, files in by_basename.items():
        if 'train' in files and 'test' in files:
            pairs.append((files['train'], files['test']))
    
    return pairs


if __name__ == "__main__":
    main()