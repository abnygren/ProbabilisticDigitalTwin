"""
Experiment and Recommendation persistence for Cu₃VS₄ Bayesian Optimization

ExperimentStore  — manages all experiments with source tracking and JSON I/O
RecommendationStore — tracks recommendation lifecycle with prediction snapshots
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json


# =============================================================================
# EXPERIMENT STORE
# =============================================================================

class ExperimentStore:
    """
    Manages all experiments with source tracking and JSON persistence.

    Each experiment has:
    - Unique ID (EXP_001, EXP_002, ...)
    - Source: 'imported', 'recommendation', or 'manual'
    - Link to recommendation ID (if source='recommendation')
    - Synthesis conditions and results
    - Timestamp
    """

    def __init__(self, json_path: Path = None):
        if json_path is None:
            raise ValueError("json_path must be provided")
        self.json_path = json_path
        self.experiments: List[Dict[str, Any]] = []
        self._next_id = 1
        if self.json_path.exists():
            self.load()

    def _generate_id(self) -> str:
        exp_id = f"EXP_{self._next_id:03d}"
        self._next_id += 1
        return exp_id

    def import_from_csv(self, csv_path: Path, source: str = 'imported') -> int:
        """Bulk import experiments from CSV file. Returns count imported."""
        df = pd.read_csv(csv_path)
        count = 0
        for _, row in df.iterrows():
            exp = {
                'exp_id': self._generate_id(),
                'source': source,
                'recommendation_id': None,
                'timestamp': datetime.now().isoformat(),
                'conditions': {
                    'Temp': float(row['Temp']),
                    'Time': float(row['Time']),
                    'VOacac': float(row['VOacac']),
                    'DDT': float(row['DDT']),
                    'OAm': float(row['OAm']),
                },
                'results': {
                    'Size': float(row['Size']) if pd.notna(row.get('Size')) else None,
                    'GSD': float(row['GSD']) if pd.notna(row.get('GSD')) else None,
                    'Squareness': float(row['Squareness']) if pd.notna(row.get('Squareness')) else None,
                    'HasProduct': int(row.get('HasProduct', 0)),
                    'PhasePure': int(row.get('PhasePure', 0)),
                    'Polymorph': str(row.get('Polymorph', '')) if pd.notna(row.get('Polymorph')) else None,
                },
                'run_id_original': int(row.get('Run_ID', 0)) if pd.notna(row.get('Run_ID')) else None,
            }
            self.experiments.append(exp)
            count += 1
        self.save()
        return count

    def add_experiment(
        self,
        conditions: Dict[str, float],
        results: Dict[str, Any],
        source: str = 'manual',
        recommendation_id: str = None
    ) -> str:
        """Add a single experiment. Returns experiment ID."""
        exp_id = self._generate_id()
        exp = {
            'exp_id': exp_id,
            'source': source,
            'recommendation_id': recommendation_id,
            'timestamp': datetime.now().isoformat(),
            'conditions': {
                'Temp': float(conditions.get('Temp', 0)),
                'Time': float(conditions.get('Time', 0)),
                'VOacac': float(conditions.get('VOacac', 0)),
                'DDT': float(conditions.get('DDT', 0)),
                'OAm': float(conditions.get('OAm', 0)),
            },
            'results': {
                'Size': float(results['Size']) if results.get('Size') is not None else None,
                'GSD': float(results['GSD']) if results.get('GSD') is not None else None,
                'Squareness': float(results['Squareness']) if results.get('Squareness') is not None else None,
                'HasProduct': int(results.get('HasProduct', 0)),
                'PhasePure': int(results.get('PhasePure', 0)),
                'Polymorph': results.get('Polymorph'),
            },
        }
        self.experiments.append(exp)
        self.save()
        return exp_id

    def get_all(self) -> pd.DataFrame:
        """Return all experiments as DataFrame."""
        if not self.experiments:
            return pd.DataFrame()
        rows = []
        for exp in self.experiments:
            row = {
                'exp_id': exp['exp_id'],
                'source': exp['source'],
                'recommendation_id': exp.get('recommendation_id'),
                'timestamp': exp['timestamp'],
                **exp['conditions'],
                **exp['results'],
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def get_by_source(self, source: str) -> pd.DataFrame:
        df = self.get_all()
        if df.empty:
            return df
        return df[df['source'] == source]

    def get_by_id(self, exp_id: str) -> Optional[Dict]:
        for exp in self.experiments:
            if exp['exp_id'] == exp_id:
                return exp
        return None

    def get_training_data(self) -> pd.DataFrame:
        """Return data formatted for model training (HasProduct=1 with valid measurements)."""
        df = self.get_all()
        if df.empty:
            return df
        df = df[df['HasProduct'] == 1].copy()
        df = df.dropna(subset=['Size', 'GSD', 'Squareness'])
        df['IsCubic'] = (df['Polymorph'].fillna('').str.lower().str.strip() == 'cubic').astype(int)
        return df

    def count(self) -> Dict[str, int]:
        df = self.get_all()
        if df.empty:
            return {'total': 0, 'imported': 0, 'recommendation': 0, 'manual': 0}
        counts = df['source'].value_counts().to_dict()
        counts['total'] = len(df)
        return counts

    def save(self):
        data = {'next_id': self._next_id, 'experiments': self.experiments}
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            self._next_id = data.get('next_id', 1)
            self.experiments = data.get('experiments', [])

    def __len__(self):
        return len(self.experiments)

    def __repr__(self):
        counts = self.count()
        return (f"ExperimentStore({counts['total']} experiments: "
                f"{counts.get('imported', 0)} imported, "
                f"{counts.get('recommendation', 0)} from recs, "
                f"{counts.get('manual', 0)} manual)")


# =============================================================================
# RECOMMENDATION STORE
# =============================================================================

class RecommendationStore:
    """
    Tracks recommendation lifecycle with full prediction snapshots.

    Each recommendation has:
    - Unique ID (REC_001, REC_002, ...)
    - Status: 'pending', 'completed', or 'skipped'
    - Target parameters, suggested conditions, predictions (frozen snapshot)
    - Actual results and computed errors (when completed)
    """

    def __init__(self, json_path: Path = None):
        if json_path is None:
            raise ValueError("json_path must be provided")
        self.json_path = json_path
        self.recommendations: List[Dict[str, Any]] = []
        self._next_id = 1
        if self.json_path.exists():
            self.load()

    def _generate_id(self) -> str:
        rec_id = f"REC_{self._next_id:03d}"
        self._next_id += 1
        return rec_id

    def save_recommendation(
        self,
        target_size: float,
        size_tolerance: float,
        conditions: Dict[str, float],
        predictions: Dict[str, float],
        rank: int = 1
    ) -> str:
        """Save a new recommendation with prediction snapshot. Returns recommendation ID."""
        rec_id = self._generate_id()
        rec = {
            'rec_id': rec_id,
            'status': 'pending',
            'rank': rank,
            'timestamp': datetime.now().isoformat(),
            'target': {'size': target_size, 'tolerance': size_tolerance},
            'conditions': {
                'Temp': float(conditions.get('Temp', 0)),
                'Time': float(conditions.get('Time', 0)),
                'VOacac': float(conditions.get('VOacac', 0)),
                'DDT': float(conditions.get('DDT', 0)),
                'OAm': float(conditions.get('OAm', 0)),
            },
            'predictions': {
                'size_mu': float(predictions.get('size_mu', 0)),
                'size_std': float(predictions.get('size_std', 0)),
                'gsd_mu': float(predictions.get('gsd_mu', 0)),
                'gsd_std': float(predictions.get('gsd_std', 0)),
                'sq_mu': float(predictions.get('sq_mu', 0)),
                'sq_std': float(predictions.get('sq_std', 0)),
                'p_feasible': float(predictions.get('p_feasible', 0)),
            },
            'actual_results': None,
            'completed_timestamp': None,
            'errors': None,
            'skip_reason': None,
            'skipped_timestamp': None,
        }
        self.recommendations.append(rec)
        self.save()
        return rec_id

    def complete(self, rec_id: str, actual_results: Dict[str, Any]) -> Dict[str, float]:
        """Mark recommendation as completed and compute errors."""
        rec = self.get_by_id(rec_id)
        if rec is None:
            raise ValueError(f"Recommendation {rec_id} not found")
        if rec['status'] != 'pending':
            raise ValueError(f"Recommendation {rec_id} is not pending (status: {rec['status']})")

        rec['actual_results'] = {
            'Size': float(actual_results['Size']) if actual_results.get('Size') is not None else None,
            'GSD': float(actual_results['GSD']) if actual_results.get('GSD') is not None else None,
            'Squareness': float(actual_results['Squareness']) if actual_results.get('Squareness') is not None else None,
            'HasProduct': int(actual_results.get('HasProduct', 0)),
            'PhasePure': int(actual_results.get('PhasePure', 0)),
            'Polymorph': actual_results.get('Polymorph'),
        }
        rec['completed_timestamp'] = datetime.now().isoformat()
        rec['status'] = 'completed'

        errors = self._compute_errors(rec['predictions'], rec['actual_results'])
        rec['errors'] = errors
        self.save()
        return errors

    def _compute_errors(self, predictions: Dict[str, float], actual: Dict[str, Any]) -> Dict[str, float]:
        errors = {}
        for prop, pred_key, std_key in [
            ('Size', 'size_mu', 'size_std'),
            ('GSD', 'gsd_mu', 'gsd_std'),
            ('Squareness', 'sq_mu', 'sq_std'),
        ]:
            actual_val = actual.get(prop)
            pred_val = predictions.get(pred_key)
            pred_std = predictions.get(std_key, 0.001)

            if actual_val is not None and pred_val is not None:
                error = actual_val - pred_val
                z_score = error / max(pred_std, 0.001)
                errors[f'{prop.lower()}_error'] = error
                errors[f'{prop.lower()}_z_score'] = z_score
                errors[f'{prop.lower()}_within_1sigma'] = abs(z_score) < 1.0
                errors[f'{prop.lower()}_within_2sigma'] = abs(z_score) < 2.0
        return errors

    def skip(self, rec_id: str, reason: str = None):
        rec = self.get_by_id(rec_id)
        if rec is None:
            raise ValueError(f"Recommendation {rec_id} not found")
        if rec['status'] != 'pending':
            raise ValueError(f"Recommendation {rec_id} is not pending")
        rec['status'] = 'skipped'
        rec['skip_reason'] = reason
        rec['skipped_timestamp'] = datetime.now().isoformat()
        self.save()

    def get_by_id(self, rec_id: str) -> Optional[Dict]:
        for rec in self.recommendations:
            if rec['rec_id'] == rec_id:
                return rec
        return None

    def get_pending(self) -> List[Dict]:
        return [r for r in self.recommendations if r['status'] == 'pending']

    def get_completed(self) -> List[Dict]:
        return [r for r in self.recommendations if r['status'] == 'completed']

    def get_all(self) -> List[Dict]:
        return self.recommendations

    def get_history_df(self) -> pd.DataFrame:
        if not self.recommendations:
            return pd.DataFrame()
        rows = []
        for rec in self.recommendations:
            row = {
                'rec_id': rec['rec_id'],
                'status': rec['status'],
                'rank': rec.get('rank', 1),
                'timestamp': rec['timestamp'],
                'target_size': rec['target']['size'],
                'target_tol': rec['target']['tolerance'],
                **{f"cond_{k}": v for k, v in rec['conditions'].items()},
                **{f"pred_{k}": v for k, v in rec['predictions'].items()},
            }
            if rec['actual_results']:
                row.update({f"actual_{k}": v for k, v in rec['actual_results'].items()})
            if rec['errors']:
                row.update(rec['errors'])
            rows.append(row)
        return pd.DataFrame(rows)

    def get_error_statistics(self) -> Dict[str, Any]:
        """Compute error statistics from completed recommendations (for calibration / bias correction)."""
        completed = self.get_completed()
        if len(completed) < 2:
            return {'n_completed': len(completed), 'sufficient_data': False}

        stats: Dict[str, Any] = {'n_completed': len(completed), 'sufficient_data': True}
        for prop in ['size', 'gsd', 'squareness']:
            errors = [r['errors'].get(f'{prop}_error') for r in completed
                     if r['errors'] and r['errors'].get(f'{prop}_error') is not None]
            z_scores = [r['errors'].get(f'{prop}_z_score') for r in completed
                       if r['errors'] and r['errors'].get(f'{prop}_z_score') is not None]
            within_1s = [r['errors'].get(f'{prop}_within_1sigma') for r in completed
                        if r['errors'] and r['errors'].get(f'{prop}_within_1sigma') is not None]
            within_2s = [r['errors'].get(f'{prop}_within_2sigma') for r in completed
                        if r['errors'] and r['errors'].get(f'{prop}_within_2sigma') is not None]

            if errors:
                stats[f'{prop}_mae'] = np.mean(np.abs(errors))
                stats[f'{prop}_mean_error'] = np.mean(errors)
                stats[f'{prop}_std_error'] = np.std(errors)
            if z_scores:
                stats[f'{prop}_mean_z'] = np.mean(np.abs(z_scores))
                stats[f'{prop}_calibration_factor'] = max(1.0, np.sqrt(np.mean(np.array(z_scores)**2)))
            if within_1s:
                stats[f'{prop}_within_1sigma_rate'] = np.mean(within_1s)
            if within_2s:
                stats[f'{prop}_within_2sigma_rate'] = np.mean(within_2s)
        return stats

    def count(self) -> Dict[str, int]:
        return {
            'total': len(self.recommendations),
            'pending': len(self.get_pending()),
            'completed': len(self.get_completed()),
            'skipped': len([r for r in self.recommendations if r['status'] == 'skipped']),
        }

    def save(self):
        data = {'next_id': self._next_id, 'recommendations': self.recommendations}
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            self._next_id = data.get('next_id', 1)
            self.recommendations = data.get('recommendations', [])

    def __repr__(self):
        counts = self.count()
        return (f"RecommendationStore({counts['total']} recs: "
                f"{counts['pending']} pending, "
                f"{counts['completed']} completed, "
                f"{counts['skipped']} skipped)")
