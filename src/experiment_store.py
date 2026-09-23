"""Experiment and recommendation persistence.

ExperimentStore holds all experiments with source tracking.
RecommendationStore holds recommendation lifecycle and frozen prediction snapshots.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import shutil


def _atomic_json_save(filepath: Path, data: dict):
    """Write JSON atomically: temp file + rename, with a ``.bak`` backup."""
    filepath = Path(filepath)
    tmp_path = filepath.parent / (filepath.name + '.tmp')
    bak_path = filepath.parent / (filepath.name + '.bak')
    try:
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        if filepath.exists():
            shutil.copy2(filepath, bak_path)
        tmp_path.replace(filepath)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


class ExperimentStore:
    """All experiments with source tracking and JSON persistence.

    Each experiment carries:
        exp_id            unique ID (``EXP_001``, ``EXP_002`` ...)
        source            ``'imported'`` / ``'recommendation'`` / ``'manual'``
        recommendation_id link to the originating recommendation (if any)
        conditions        synthesis parameters (Temp, Time, VOacac, DDT, OAm)
        precursors        Cu and Metal precursor identities
        results           Size, CV, Squareness, HasProduct, PhasePure, Polymorph
        timestamp         ISO-format wall-clock time
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

    def import_from_csv(
        self,
        csv_path: Path,
        source: str = 'imported',
        default_cu_precursor: str = 'CuI',
        default_metal_precursor: str = 'VO(acac)2',
    ) -> int:
        """Bulk-import experiments from a CSV. Returns the count imported.

        Deduplicates on ``Run_ID`` when present (preserving replicates with
        identical conditions but different outcomes), falling back to
        condition-key dedup only when ``Run_ID`` is missing.

        If the CSV has ``Cu_precursor`` / ``Metal_precursor`` columns those
        per-row values are stored; otherwise the supplied defaults are used.
        """
        df = pd.read_csv(csv_path)

        # Accept 'CV' or the older 'GSD' column; stored as 'CV'.
        has_cv_col = 'CV' in df.columns
        has_gsd_col = 'GSD' in df.columns
        if not has_cv_col and not has_gsd_col:
            raise ValueError("CSV must contain either 'CV' or 'GSD' column for the polydispersity metric.")

        existing_run_ids = set()
        existing_conditions = set()
        for exp in self.experiments:
            rid = exp.get('run_id_original')
            if rid is not None:
                existing_run_ids.add(rid)
            c = exp['conditions']
            prec = exp.get('precursors', {})
            cu_prec = prec.get('Cu_precursor', 'CuI')
            metal_prec = prec.get('Metal_precursor', 'VO(acac)2')
            existing_conditions.add((
                round(c['Temp'], 4), round(c['Time'], 4),
                round(c['VOacac'], 6), round(c['DDT'], 4),
                round(c['OAm'], 4), cu_prec, metal_prec,
            ))

        has_run_id_col = 'Run_ID' in df.columns

        count = 0
        skipped = 0
        for _, row in df.iterrows():
            run_id = (int(row['Run_ID'])
                      if has_run_id_col and pd.notna(row.get('Run_ID'))
                      else None)

            has_cu = ('Cu_precursor' in df.columns
                      and pd.notna(row.get('Cu_precursor')))
            row_cu_prec = str(row['Cu_precursor']) if has_cu else default_cu_precursor

            has_metal = ('Metal_precursor' in df.columns
                         and pd.notna(row.get('Metal_precursor')))
            row_metal_prec = str(row['Metal_precursor']) if has_metal else default_metal_precursor

            if run_id is not None and run_id in existing_run_ids:
                skipped += 1
                continue
            if run_id is None:
                cond_key = (
                    round(float(row['Temp']), 4), round(float(row['Time']), 4),
                    round(float(row['VOacac']), 6), round(float(row['DDT']), 4),
                    round(float(row['OAm']), 4), row_cu_prec, row_metal_prec,
                )
                if cond_key in existing_conditions:
                    skipped += 1
                    continue
                existing_conditions.add(cond_key)

            # Pull the polydispersity metric from whichever column exists.
            polydispersity_value = None
            if has_cv_col and pd.notna(row.get('CV')):
                polydispersity_value = float(row['CV'])
            elif has_gsd_col and pd.notna(row.get('GSD')):
                polydispersity_value = float(row['GSD'])

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
                'precursors': {
                    'Cu_precursor': row_cu_prec,
                    'Metal_precursor': row_metal_prec,
                },
                'results': {
                    'Size': float(row['Size']) if pd.notna(row.get('Size')) else None,
                    'CV': polydispersity_value,
                    'Squareness': float(row['Squareness']) if pd.notna(row.get('Squareness')) else None,
                    'HasProduct': int(row.get('HasProduct', 0)),
                    'PhasePure': int(row.get('PhasePure', 0)),
                    'Polymorph': str(row.get('Polymorph', '')) if pd.notna(row.get('Polymorph')) else None,
                },
                'run_id_original': run_id,
            }
            if run_id is not None:
                existing_run_ids.add(run_id)
            self.experiments.append(exp)
            count += 1

        if skipped > 0:
            print(f"[INFO] Skipped {skipped} duplicate experiments during import")
        self.save()
        return count

    def add_experiment(
        self,
        conditions: Dict[str, float],
        results: Dict[str, Any],
        source: str = 'manual',
        recommendation_id: str = None,
        cu_precursor: str = None,
        metal_precursor: str = None,
    ) -> str:
        """Add a single experiment. Returns the new experiment ID.

        Parameters
        ----------
        cu_precursor : str, optional
            Cu precursor used (``'CuI'``, ``'CuCl'`` ...). Stored for transfer learning.
        metal_precursor : str, optional
            Group-5 metal precursor used (``'VO(acac)2'``, ``'TaCl5'`` ...).
        """
        exp_id = self._generate_id()

        if results.get('CV') is not None:
            cv_value = float(results['CV'])
        elif results.get('GSD') is not None:
            cv_value = float(results['GSD'])
        else:
            cv_value = None

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
            'precursors': {
                'Cu_precursor': cu_precursor or 'CuI',
                'Metal_precursor': metal_precursor or 'VO(acac)2',
            },
            'results': {
                'Size': float(results['Size']) if results.get('Size') is not None else None,
                'CV': cv_value,
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
        """Return every experiment as a DataFrame.

        Adds ``Cu_precursor`` and ``Metal_precursor`` columns when precursor
        metadata is stored (used by transfer learning).
        """
        if not self.experiments:
            return pd.DataFrame()
        rows = []
        for exp in self.experiments:
            precursors = exp.get('precursors', {})
            row = {
                'exp_id': exp['exp_id'],
                'source': exp['source'],
                'recommendation_id': exp.get('recommendation_id'),
                'timestamp': exp['timestamp'],
                **exp['conditions'],
                'Cu_precursor': precursors.get('Cu_precursor', 'CuI'),
                'Metal_precursor': precursors.get('Metal_precursor', 'VO(acac)2'),
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
        """Return rows usable for model training: ``HasProduct == 1`` and all measurements present."""
        df = self.get_all()
        if df.empty:
            return df
        df = df[df['HasProduct'] == 1].copy()
        df = df.dropna(subset=['Size', 'CV', 'Squareness'])
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
        _atomic_json_save(self.json_path, data)

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

class RecommendationStore:
    """Tracks recommendation lifecycle with frozen prediction snapshots.

    Each recommendation carries:
        rec_id              ``REC_001``, ``REC_002`` ...
        status              ``'pending'`` / ``'completed'`` / ``'skipped'``
        target              target size, tolerance, squareness bin
        conditions          suggested raw lab parameters
        predictions         model snapshot at recommendation time
        actual_results      measured outcome (once completed)
        errors              prediction errors and z-scores (once completed)
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
        squareness_bin: Optional[str],
        conditions: Dict[str, float],
        predictions: Dict[str, float],
        feature_mode: Optional[str] = None,
        correction_applied: Optional[bool] = None,
        rank: int = 1,
        precursors: Optional[Dict[str, str]] = None,
    ) -> str:
        """Persist a new recommendation with a frozen prediction snapshot. Returns the new ``rec_id``."""
        rec_id = self._generate_id()
        rec = {
            'rec_id': rec_id,
            'status': 'pending',
            'rank': rank,
            'timestamp': datetime.now().isoformat(),
            'target': {
                'size': target_size,
                'tolerance': size_tolerance,
                'squareness_bin': squareness_bin,
            },
            'conditions': {
                'Temp': float(conditions.get('Temp', 0)),
                'Time': float(conditions.get('Time', 0)),
                'VOacac': float(conditions.get('VOacac', 0)),
                'DDT': float(conditions.get('DDT', 0)),
                'OAm': float(conditions.get('OAm', 0)),
            },
            'precursors': precursors or {},
            'predictions': {
                'size_mu': float(predictions.get('size_mu', 0)),
                'size_std': float(predictions.get('size_std', 0)),
                'cv_mu': float(predictions.get('cv_mu', 0)),
                'cv_std': float(predictions.get('cv_std', 0)),
                'sq_mu': float(predictions.get('sq_mu', 0)),
                'sq_std': float(predictions.get('sq_std', 0)),
                'p_feasible': float(predictions.get('p_feasible', 0)),
                'p_bin': float(predictions.get('p_bin', 0)),
                'correction_applied': bool(predictions.get('correction_applied', correction_applied)),
            },
            'model_context': {
                'feature_mode': feature_mode,
                'correction_applied': bool(predictions.get('correction_applied', correction_applied)),
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
        """Mark a recommendation completed, store actual results, and compute errors."""
        rec = self.get_by_id(rec_id)
        if rec is None:
            raise ValueError(f"Recommendation {rec_id} not found")
        if rec['status'] != 'pending':
            raise ValueError(f"Recommendation {rec_id} is not pending (status: {rec['status']})")

        def _float_or_none(key):
            v = actual_results.get(key)
            return float(v) if v is not None else None

        rec['actual_results'] = {
            'Size': _float_or_none('Size'),
            'CV': _float_or_none('CV'),
            'Squareness': _float_or_none('Squareness'),
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
            ('CV', 'cv_mu', 'cv_std'),
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

    def delete(self, rec_id: str):
        """Permanently remove a pending recommendation. Only ``'pending'`` rows can be deleted."""
        rec = self.get_by_id(rec_id)
        if rec is None:
            raise ValueError(f"Recommendation {rec_id} not found")
        if rec['status'] != 'pending':
            raise ValueError(
                f"Cannot delete {rec_id}: status is '{rec['status']}'. "
                f"Only pending recommendations can be deleted."
            )
        self.recommendations = [r for r in self.recommendations if r['rec_id'] != rec_id]
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
                'target_squareness_bin': rec.get('target', {}).get('squareness_bin'),
                'feature_mode': rec.get('model_context', {}).get('feature_mode'),
                'correction_applied': rec.get('model_context', {}).get('correction_applied'),
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
        """Aggregate error statistics from completed recommendations (used for calibration and bias correction)."""
        completed = self.get_completed()
        if len(completed) < 2:
            return {'n_completed': len(completed), 'sufficient_data': False}

        stats: Dict[str, Any] = {'n_completed': len(completed), 'sufficient_data': True}
        for prop in ['size', 'cv', 'squareness']:
            def _collect(key):
                return [r['errors'].get(f'{prop}_{key}') for r in completed
                        if r['errors'] and r['errors'].get(f'{prop}_{key}') is not None]

            errors = _collect('error')
            z_scores = _collect('z_score')
            within_1s = _collect('within_1sigma')
            within_2s = _collect('within_2sigma')

            if errors:
                stats[f'{prop}_mae'] = np.mean(np.abs(errors))
                stats[f'{prop}_mean_error'] = np.mean(errors)
                stats[f'{prop}_std_error'] = np.std(errors)
            if z_scores:
                stats[f'{prop}_mean_z'] = np.mean(np.abs(z_scores))
                rms_z = np.sqrt(np.mean(np.array(z_scores) ** 2))
                stats[f'{prop}_calibration_factor'] = max(0.5, rms_z)
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
        _atomic_json_save(self.json_path, data)

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
