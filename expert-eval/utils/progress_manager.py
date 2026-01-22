"""
Progress manager for expert evaluations.

Handles saving, loading, and resuming evaluation progress.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProgressManager:
    """Manages evaluation progress persistence."""
    
    def __init__(self, progress_dir: Optional[Path] = None):
        """
        Initialize the progress manager.
        
        Args:
            progress_dir: Directory for storing progress files.
                         Defaults to 'expert-eval/progress/' in project root.
        """
        if progress_dir is None:
            progress_dir = Path(__file__).parent.parent / "progress"
        
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        self._evaluations: Dict[str, Dict[str, Any]] = {}
        self._current_file: Optional[str] = None
    
    def _get_progress_path(self, results_filename: str) -> Path:
        """Get the progress file path for a given results file."""
        # Remove .jsonl extension and add _evaluations.json
        base_name = Path(results_filename).stem
        return self.progress_dir / f"{base_name}_evaluations.json"
    
    def load_progress(self, results_filename: str) -> Dict[str, Dict[str, Any]]:
        """
        Load existing evaluation progress for a results file.
        
        Args:
            results_filename: Name of the results JSONL file.
        
        Returns:
            Dictionary mapping sample keys to evaluation data.
        """
        progress_path = self._get_progress_path(results_filename)
        self._current_file = results_filename
        
        if progress_path.exists():
            try:
                with progress_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._evaluations = data.get("evaluations", {})
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load progress file: {e}")
                self._evaluations = {}
        else:
            self._evaluations = {}
        
        return self._evaluations
    
    def save_progress(self, results_filename: Optional[str] = None) -> None:
        """
        Save current evaluation progress.
        
        Args:
            results_filename: Name of the results JSONL file.
                            Uses current file if not specified.
        """
        if results_filename is None:
            results_filename = self._current_file
        
        if results_filename is None:
            raise ValueError("No results file specified")
        
        progress_path = self._get_progress_path(results_filename)
        
        data = {
            "results_file": results_filename,
            "last_updated": datetime.now().isoformat(),
            "total_evaluations": len(self._evaluations),
            "evaluations": self._evaluations,
        }
        
        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_evaluation(self, sample_key: str) -> Optional[Dict[str, Any]]:
        """
        Get evaluation for a specific sample.
        
        Args:
            sample_key: Unique key for the sample.
        
        Returns:
            Evaluation data or None if not evaluated.
        """
        return self._evaluations.get(sample_key)
    
    def save_evaluation(
        self,
        sample_key: str,
        reasoning_quality: int,
        reasoning_valid: bool,
        tool_interpretation_correct: Optional[bool],
        fabrication_suspected: bool,
        verdict_agree: bool,
        expert_verdict: str,
        notes: str,
        auto_save: bool = True,
    ) -> Dict[str, Any]:
        """
        Save an evaluation for a sample.
        
        Args:
            sample_key: Unique key for the sample.
            reasoning_quality: 1-5 rating (poor to excellent).
            reasoning_valid: Is the reasoning scientifically sound?
            tool_interpretation_correct: Were tool results correctly used?
            fabrication_suspected: Does reasoning appear made up?
            verdict_agree: Does expert agree with the verdict?
            expert_verdict: Expert's own verdict (real/fake/uncertain).
            notes: Free-form notes.
            auto_save: Whether to auto-save to disk.
        
        Returns:
            The saved evaluation data.
        """
        evaluation = {
            "sample_key": sample_key,
            "reasoning_quality": reasoning_quality,
            "reasoning_valid": reasoning_valid,
            "tool_interpretation_correct": tool_interpretation_correct,
            "fabrication_suspected": fabrication_suspected,
            "verdict_agree": verdict_agree,
            "expert_verdict": expert_verdict,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
        }
        
        self._evaluations[sample_key] = evaluation
        
        if auto_save and self._current_file:
            self.save_progress()
        
        return evaluation
    
    def delete_evaluation(self, sample_key: str, auto_save: bool = True) -> bool:
        """
        Delete an evaluation for a sample.
        
        Args:
            sample_key: Unique key for the sample.
            auto_save: Whether to auto-save to disk.
        
        Returns:
            True if evaluation was deleted, False if not found.
        """
        if sample_key in self._evaluations:
            del self._evaluations[sample_key]
            if auto_save and self._current_file:
                self.save_progress()
            return True
        return False
    
    def get_progress_stats(self, total_samples: int) -> Dict[str, Any]:
        """
        Get progress statistics.
        
        Args:
            total_samples: Total number of samples in the dataset.
        
        Returns:
            Dictionary with progress statistics.
        """
        evaluated = len(self._evaluations)
        return {
            "evaluated": evaluated,
            "total": total_samples,
            "remaining": total_samples - evaluated,
            "percent_complete": (evaluated / total_samples * 100) if total_samples > 0 else 0,
        }
    
    def get_evaluated_keys(self) -> List[str]:
        """Get list of all evaluated sample keys."""
        return list(self._evaluations.keys())
    
    def is_evaluated(self, sample_key: str) -> bool:
        """Check if a sample has been evaluated."""
        return sample_key in self._evaluations
    
    def export_to_csv(self, output_path: Path) -> None:
        """
        Export evaluations to CSV format.
        
        Args:
            output_path: Path for the output CSV file.
        """
        if not self._evaluations:
            return
        
        fieldnames = [
            "sample_key",
            "reasoning_quality",
            "reasoning_valid",
            "tool_interpretation_correct",
            "fabrication_suspected",
            "verdict_agree",
            "expert_verdict",
            "notes",
            "timestamp",
        ]
        
        output_path = Path(output_path)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for evaluation in self._evaluations.values():
                writer.writerow(evaluation)
    
    def get_all_evaluations(self) -> Dict[str, Dict[str, Any]]:
        """Get all evaluations."""
        return self._evaluations.copy()
