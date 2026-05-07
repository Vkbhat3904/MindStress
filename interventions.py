import json
import logging
import datetime
from typing import Optional

# Set up logging for interventions
logger = logging.getLogger("InterventionLogger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("interventions_log.jsonl")
fh.setFormatter(logging.Formatter('%(message)s'))
# Prevent adding multiple handlers if re-imported
if not logger.handlers:
    logger.addHandler(fh)

class InterventionManager:
    def __init__(self, threshold: float = 0.70, mild_threshold: float = 0.50, cooldown_steps: int = 5):
        """
        Manages stress interventions.
        Args:
            threshold (float): High stress threshold to trigger active interventions.
            mild_threshold (float): Mild stress threshold to trigger passive interventions.
            cooldown_steps (int): Minimum steps to wait before triggering a new intervention.
        """
        self.threshold = threshold
        self.mild_threshold = mild_threshold
        self.cooldown_steps = cooldown_steps
        self.steps_since_last_intervention = cooldown_steps  # Ready to fire immediately if needed
        
    def evaluate(self, stress_prob: float, physio_metrics: Optional[dict] = None) -> Optional[str]:
        """
        Evaluates current stress probability and physiological metrics to determine if an
        intervention is needed.
        
        Args:
            stress_prob (float): Predicted stress probability from the model [0, 1].
            physio_metrics (dict, optional): Dictionary with physiological indicators 
                                             e.g., {'lf_hf_ratio': 1.6, 'beta_power': 0.9}
        Returns:
            str or None: The intervention text if triggered, else None.
        """
        if physio_metrics is None:
            physio_metrics = {}
            
        self.steps_since_last_intervention += 1
        
        # Only trigger if we are above the mild threshold and cooling down period has passed
        if stress_prob >= self.mild_threshold:
            if self.steps_since_last_intervention >= self.cooldown_steps:
                intervention = self._select_intervention(stress_prob, physio_metrics)
                self._log_intervention(stress_prob, intervention, physio_metrics)
                self.steps_since_last_intervention = 0
                return intervention
        return None
        
    def _select_intervention(self, stress_prob: float, physio_metrics: dict) -> str:
        """Selects the most appropriate intervention based on stress level and physiological markers."""
        # High stress
        if stress_prob >= self.threshold:
            # Check physiological indicators for targeted interventions
            lf_hf = physio_metrics.get('lf_hf_ratio', 0.0)
            beta = physio_metrics.get('beta_power', 0.0)
            
            # High sympathetic arousal -> Grounding / Breathing
            if lf_hf > 1.5 or beta > 0.8:
                return "Box Breathing: Inhale 4s, Hold 4s, Exhale 4s, Hold 4s."
            else:
                return "Music Recommendation: Slow-tempo ambient music (60 BPM) or Binaural beats at 432Hz."
                
        # Mild stress
        elif stress_prob >= self.mild_threshold:
            return "Relaxation Prompt: 5-4-3-2-1 grounding technique. Name 5 things you can see..."
            
        return "Unknown state"
        
    def _log_intervention(self, stress_prob: float, intervention: str, physio_metrics: dict) -> None:
        """Logs the intervention event and metadata to a file and prints to console."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "stress_probability": round(float(stress_prob), 4),
            "intervention_type": intervention,
            "physio_metrics": physio_metrics
        }
        
        # Log to JSON Lines file
        logger.info(json.dumps(log_entry))
        
        # Print to console for realtime feedback
        print("\n" + "="*50)
        print(f"[INTERVENTION TRIGGERED] - {log_entry['timestamp']}")
        print(f"Stress Probability: {stress_prob:.2%}")
        if physio_metrics:
            metrics_str = ", ".join(f"{k}: {v:.2f}" for k, v in physio_metrics.items())
            print(f"Physio Metrics: {metrics_str}")
        print(f"Action: {intervention}")
        print("="*50 + "\n")
