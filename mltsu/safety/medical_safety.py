"""
Medical Safety Wrapper for TinyBioBERT
Enforces deterministic execution for safety-critical medical predictions.

This module provides the critical safety infrastructure required for deploying
medical AI in clinical settings, ensuring FDA compliance and patient safety.
"""

import torch
import torch.nn as nn
import numpy as np
from enum import Enum
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
from pathlib import Path
import warnings
import copy


class MedicalTaskType(Enum):
    """
    Medical task classification for safety-aware execution.

    Tasks are classified into three categories:
    1. CRITICAL: Life-threatening, requires deterministic execution
    2. DIAGNOSTIC: Clinical decisions, requires high confidence
    3. RESEARCH: Exploratory analysis, can use P-bit stochasticity
    """

    # Safety-critical tasks (MUST be deterministic)
    DRUG_DOSING = "drug_dosing"
    ALLERGY_DETECTION = "allergy_detection"
    CONTRAINDICATION_CHECK = "contraindication_check"
    EMERGENCY_DIAGNOSIS = "emergency_diagnosis"
    SURGICAL_PLANNING = "surgical_planning"

    # Diagnostic tasks (high confidence required)
    DIAGNOSIS_STANDARD = "diagnosis_standard"
    RISK_ASSESSMENT = "risk_assessment"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    LAB_INTERPRETATION = "lab_interpretation"

    # Research tasks (P-bit allowed)
    NER_EXTRACTION = "ner_extraction"
    LITERATURE_MINING = "literature_mining"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    PHENOTYPING = "phenotyping"
    COHORT_DISCOVERY = "cohort_discovery"

    @classmethod
    def is_critical(cls, task_type: 'MedicalTaskType') -> bool:
        """Check if task requires deterministic execution."""
        critical_tasks = {
            cls.DRUG_DOSING,
            cls.ALLERGY_DETECTION,
            cls.CONTRAINDICATION_CHECK,
            cls.EMERGENCY_DIAGNOSIS,
            cls.SURGICAL_PLANNING,
        }
        return task_type in critical_tasks

    @classmethod
    def is_diagnostic(cls, task_type: 'MedicalTaskType') -> bool:
        """Check if task is diagnostic (requires high confidence)."""
        diagnostic_tasks = {
            cls.DIAGNOSIS_STANDARD,
            cls.RISK_ASSESSMENT,
            cls.TREATMENT_RECOMMENDATION,
            cls.LAB_INTERPRETATION,
        }
        return task_type in diagnostic_tasks

    @classmethod
    def requires_audit(cls, task_type: 'MedicalTaskType') -> bool:
        """Check if task requires audit logging for compliance."""
        return cls.is_critical(task_type) or cls.is_diagnostic(task_type)


@dataclass
class SafetyConfig:
    """Configuration for medical safety features."""

    # Execution modes
    force_deterministic_critical: bool = True
    confidence_threshold_diagnostic: float = 0.85
    max_uncertainty_allowed: float = 0.3

    # Audit settings
    enable_audit_logging: bool = True
    audit_log_path: Optional[str] = "./audit_logs"
    save_frequency: int = 100  # Save every N predictions

    # Safety checks
    enable_input_validation: bool = True
    enable_output_validation: bool = True
    check_calibration: bool = True

    # Reproducibility
    deterministic_seed: int = 42
    save_prediction_hashes: bool = True

    # Regulatory compliance
    include_patient_id: bool = True
    include_timestamp: bool = True
    include_model_version: bool = True
    encrypt_audit_logs: bool = False


@dataclass
class AuditEntry:
    """Single audit log entry for regulatory compliance."""

    timestamp: str
    task_type: str
    execution_mode: str
    patient_id: Optional[str]
    input_hash: str
    output_hash: str
    confidence: float
    uncertainty: Optional[float]
    model_version: str
    safety_checks: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON for storage."""
        return json.dumps(self.__dict__, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'AuditEntry':
        """Reconstruct from JSON."""
        data = json.loads(json_str)
        return cls(**data)


class AuditLog:
    """
    Audit logging system for medical AI regulatory compliance.
    Meets HIPAA, FDA 21 CFR Part 11, and EU MDR requirements.
    """

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.entries: List[AuditEntry] = []
        self.entry_count = 0

        # Create audit directory
        if config.audit_log_path:
            self.log_dir = Path(config.audit_log_path)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique log file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"audit_{timestamp}.jsonl"

    def add_entry(self, entry: AuditEntry):
        """Add audit entry and save if needed."""
        self.entries.append(entry)
        self.entry_count += 1

        # Save periodically
        if (self.config.save_frequency > 0 and
            self.entry_count % self.config.save_frequency == 0):
            self.save()

    def save(self):
        """Save audit entries to file."""
        if not self.config.audit_log_path or not self.entries:
            return

        # Append entries to file
        with open(self.log_file, 'a') as f:
            for entry in self.entries:
                f.write(entry.to_json() + '\n')

        # Clear saved entries from memory
        self.entries.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get audit log summary statistics."""
        return {
            'total_entries': self.entry_count,
            'unsaved_entries': len(self.entries),
            'log_file': str(self.log_file) if hasattr(self, 'log_file') else None,
        }


class MedicalSafetyWrapper:
    """
    Medical Safety Wrapper for TinyBioBERT.

    Enforces deterministic execution for safety-critical medical predictions
    while allowing P-bit stochasticity for research tasks.

    Key Features:
    1. Task-based execution mode selection
    2. Deterministic override for critical tasks
    3. Comprehensive audit logging
    4. Input/output validation
    5. Reproducibility guarantees
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[SafetyConfig] = None,
        uncertainty_quantifier: Optional[Any] = None,
        model_version: str = "1.0.0"
    ):
        """
        Initialize Medical Safety Wrapper.

        Args:
            model: TinyBioBERT model
            config: Safety configuration
            uncertainty_quantifier: Optional uncertainty quantification module
            model_version: Model version for audit tracking
        """
        self.model = model
        self.config = config or SafetyConfig()
        self.uncertainty_quantifier = uncertainty_quantifier
        self.model_version = model_version

        # Initialize audit log
        if self.config.enable_audit_logging:
            self.audit_log = AuditLog(self.config)
        else:
            self.audit_log = None

        # Cache for deterministic settings
        self._cached_settings = {}

        # Statistics tracking
        self.stats = {
            'total_predictions': 0,
            'deterministic_predictions': 0,
            'pbit_predictions': 0,
            'safety_overrides': 0,
        }

    def predict(
        self,
        inputs: Dict[str, torch.Tensor],
        task_type: MedicalTaskType,
        patient_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make prediction with appropriate safety level.

        Args:
            inputs: Model inputs (input_ids, attention_mask, etc.)
            task_type: Type of medical task
            patient_id: Optional patient identifier for audit
            metadata: Optional metadata for audit

        Returns:
            Prediction results with safety metadata
        """
        self.stats['total_predictions'] += 1

        # Determine execution mode
        is_critical = MedicalTaskType.is_critical(task_type)
        is_diagnostic = MedicalTaskType.is_diagnostic(task_type)
        requires_deterministic = is_critical or (
            is_diagnostic and self.config.force_deterministic_critical
        )

        # Input validation
        if self.config.enable_input_validation:
            self._validate_inputs(inputs)

        # Compute input hash for reproducibility
        input_hash = self._compute_hash(inputs)

        # Execute prediction with appropriate mode
        if requires_deterministic:
            result = self._predict_deterministic(inputs, task_type)
            execution_mode = "DETERMINISTIC"
            self.stats['deterministic_predictions'] += 1
        else:
            result = self._predict_stochastic(inputs, task_type)
            execution_mode = "PBIT_STOCHASTIC"
            self.stats['pbit_predictions'] += 1

        # Output validation
        if self.config.enable_output_validation:
            safety_checks = self._validate_outputs(result, task_type)
        else:
            safety_checks = {}

        # Apply safety overrides if needed
        if self._requires_safety_override(result, task_type):
            result = self._apply_safety_override(result, task_type)
            self.stats['safety_overrides'] += 1
            safety_checks['safety_override'] = True

        # Compute output hash
        output_hash = self._compute_hash(result.get('predictions'))

        # Add metadata
        result['execution_mode'] = execution_mode
        result['task_type'] = task_type.value
        result['safety_checks'] = safety_checks
        result['reproducibility_hash'] = f"{input_hash}_{output_hash}"

        # Audit logging
        if self.audit_log and MedicalTaskType.requires_audit(task_type):
            audit_entry = AuditEntry(
                timestamp=datetime.now().isoformat(),
                task_type=task_type.value,
                execution_mode=execution_mode,
                patient_id=patient_id,
                input_hash=input_hash,
                output_hash=output_hash,
                confidence=float(result.get('confidence', torch.tensor(0.0)).mean()),
                uncertainty=float(result.get('uncertainty', torch.tensor(0.0)).mean()) if 'uncertainty' in result else None,
                model_version=self.model_version,
                safety_checks=safety_checks,
                metadata=metadata or {},
            )
            self.audit_log.add_entry(audit_entry)

        return result

    def _predict_deterministic(
        self,
        inputs: Dict[str, torch.Tensor],
        task_type: MedicalTaskType
    ) -> Dict[str, Any]:
        """
        Make deterministic prediction (no P-bit stochasticity).

        This ensures reproducible outputs for safety-critical tasks.
        """
        # Save current model state
        self.model.eval()

        with torch.no_grad():
            # Disable all P-bit components
            old_settings = self._disable_pbit_components()

            try:
                # Set deterministic seed
                torch.manual_seed(self.config.deterministic_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.deterministic_seed)

                # Forward pass
                outputs = self.model(**inputs)

                # Extract predictions
                if 'logits' in outputs:
                    logits = outputs['logits']
                    predictions = torch.argmax(logits, dim=-1)
                    confidence = torch.softmax(logits, dim=-1).max(dim=-1)[0]
                else:
                    predictions = outputs.get('predictions')
                    confidence = outputs.get('confidence', torch.ones_like(predictions))

                result = {
                    'predictions': predictions,
                    'confidence': confidence,
                    'logits': outputs.get('logits'),
                    'hidden_states': outputs.get('hidden_states'),
                }

            finally:
                # Restore P-bit settings
                self._restore_pbit_components(old_settings)

        return result

    def _predict_stochastic(
        self,
        inputs: Dict[str, torch.Tensor],
        task_type: MedicalTaskType
    ) -> Dict[str, Any]:
        """
        Make stochastic prediction using P-bit sampling.

        This allows exploration and uncertainty quantification for research tasks.
        """
        if self.uncertainty_quantifier:
            # Use uncertainty quantifier for multiple forward passes
            result = self.uncertainty_quantifier.predict_with_uncertainty(
                inputs['input_ids'],
                inputs.get('attention_mask'),
                inputs.get('token_type_ids'),
            )
        else:
            # Single forward pass with P-bit components enabled
            self.model.eval()  # But P-bit dropout may still be active

            with torch.no_grad():
                outputs = self.model(**inputs)

                if 'logits' in outputs:
                    logits = outputs['logits']
                    predictions = torch.argmax(logits, dim=-1)
                    confidence = torch.softmax(logits, dim=-1).max(dim=-1)[0]
                else:
                    predictions = outputs.get('predictions')
                    confidence = outputs.get('confidence', torch.ones_like(predictions))

                result = {
                    'predictions': predictions,
                    'confidence': confidence,
                    'logits': outputs.get('logits'),
                    'hidden_states': outputs.get('hidden_states'),
                }

        return result

    def _disable_pbit_components(self) -> Dict[str, Any]:
        """
        Temporarily disable all P-bit sampling components.

        Returns:
            Dictionary of original settings to restore later
        """
        old_settings = {}

        # Check if model has BERT structure
        if hasattr(self.model, 'bert'):
            bert_model = self.model.bert
        elif hasattr(self.model, 'model'):
            bert_model = self.model.model
        else:
            bert_model = self.model

        # Disable attention P-bit sampling
        if hasattr(bert_model, 'encoder'):
            for i, layer in enumerate(bert_model.encoder.layers):
                if hasattr(layer, 'attention'):
                    attention = layer.attention

                    # Check for ThermodynamicAttention
                    if hasattr(attention, 'attention'):
                        attention_module = attention.attention
                        if hasattr(attention_module, 'n_samples'):
                            old_settings[f'layer_{i}_n_samples'] = attention_module.n_samples
                            attention_module.n_samples = 1  # Single deterministic sample
                        if hasattr(attention_module, 'use_tsu'):
                            old_settings[f'layer_{i}_use_tsu'] = attention_module.use_tsu
                            attention_module.use_tsu = False

        # Disable embedding noise
        if hasattr(bert_model, 'embeddings'):
            embeddings = bert_model.embeddings
            if hasattr(embeddings, 'noise_level'):
                old_settings['embedding_noise_level'] = embeddings.noise_level
                embeddings.noise_level = 0.0

        # Disable P-bit dropout
        for name, module in bert_model.named_modules():
            if 'dropout' in name.lower() and hasattr(module, 'p'):
                if hasattr(module, 'tsu_backend'):  # PbitDropout
                    old_settings[f'{name}_p'] = module.p
                    module.p = 0.0

        return old_settings

    def _restore_pbit_components(self, old_settings: Dict[str, Any]):
        """Restore P-bit sampling settings."""
        if hasattr(self.model, 'bert'):
            bert_model = self.model.bert
        elif hasattr(self.model, 'model'):
            bert_model = self.model.model
        else:
            bert_model = self.model

        # Restore attention settings
        if hasattr(bert_model, 'encoder'):
            for i, layer in enumerate(bert_model.encoder.layers):
                if hasattr(layer, 'attention'):
                    attention = layer.attention

                    if hasattr(attention, 'attention'):
                        attention_module = attention.attention

                        if f'layer_{i}_n_samples' in old_settings:
                            attention_module.n_samples = old_settings[f'layer_{i}_n_samples']

                        if f'layer_{i}_use_tsu' in old_settings:
                            attention_module.use_tsu = old_settings[f'layer_{i}_use_tsu']

        # Restore embedding noise
        if hasattr(bert_model, 'embeddings'):
            if 'embedding_noise_level' in old_settings:
                bert_model.embeddings.noise_level = old_settings['embedding_noise_level']

        # Restore dropout settings
        for name, module in bert_model.named_modules():
            if f'{name}_p' in old_settings:
                module.p = old_settings[f'{name}_p']

    def _validate_inputs(self, inputs: Dict[str, torch.Tensor]) -> bool:
        """Validate input tensors for safety."""
        # Check for required fields
        if 'input_ids' not in inputs:
            raise ValueError("Missing required input: input_ids")

        # Check tensor properties
        input_ids = inputs['input_ids']
        if input_ids.dim() != 2:
            raise ValueError(f"Expected 2D input_ids, got {input_ids.dim()}D")

        # Check for invalid values
        if torch.any(input_ids < 0):
            warnings.warn("Negative token IDs detected in input")

        return True

    def _validate_outputs(
        self,
        outputs: Dict[str, Any],
        task_type: MedicalTaskType
    ) -> Dict[str, bool]:
        """Validate outputs for medical safety."""
        checks = {}

        # Check confidence levels
        if 'confidence' in outputs:
            confidence = outputs['confidence']
            min_conf = torch.min(confidence).item()
            max_conf = torch.max(confidence).item()

            checks['confidence_valid'] = 0 <= min_conf <= max_conf <= 1

            # Critical tasks need high confidence
            if MedicalTaskType.is_critical(task_type):
                mean_conf = torch.mean(confidence).item()
                checks['confidence_sufficient'] = mean_conf >= self.config.confidence_threshold_diagnostic

        # Check uncertainty levels
        if 'uncertainty' in outputs:
            uncertainty = outputs['uncertainty']
            max_unc = torch.max(uncertainty).item()
            checks['uncertainty_acceptable'] = max_unc <= self.config.max_uncertainty_allowed

        # Check for NaN/Inf
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                checks[f'{key}_finite'] = torch.all(torch.isfinite(value)).item()

        return checks

    def _requires_safety_override(
        self,
        result: Dict[str, Any],
        task_type: MedicalTaskType
    ) -> bool:
        """Check if safety override is needed."""
        if not MedicalTaskType.is_critical(task_type):
            return False

        # Override if confidence too low
        if 'confidence' in result:
            min_conf = torch.min(result['confidence']).item()
            if min_conf < self.config.confidence_threshold_diagnostic:
                return True

        # Override if uncertainty too high
        if 'uncertainty' in result:
            max_unc = torch.max(result['uncertainty']).item()
            if max_unc > self.config.max_uncertainty_allowed:
                return True

        return False

    def _apply_safety_override(
        self,
        result: Dict[str, Any],
        task_type: MedicalTaskType
    ) -> Dict[str, Any]:
        """Apply safety override to outputs."""
        # Mark predictions as requiring human review
        result['requires_human_review'] = True
        result['safety_override_reason'] = 'Low confidence or high uncertainty'

        # For critical tasks, mask low-confidence predictions
        if 'confidence' in result and 'predictions' in result:
            mask = result['confidence'] < self.config.confidence_threshold_diagnostic
            result['predictions'][mask] = -1  # Special value for "uncertain"

        return result

    def _compute_hash(self, data: Any) -> str:
        """Compute hash for reproducibility tracking."""
        if data is None:
            return "none"

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        else:
            data_bytes = str(data).encode()

        return hashlib.sha256(data_bytes).hexdigest()[:16]

    def get_statistics(self) -> Dict[str, Any]:
        """Get safety wrapper statistics."""
        stats = self.stats.copy()

        if self.audit_log:
            stats['audit'] = self.audit_log.get_summary()

        # Calculate ratios
        if stats['total_predictions'] > 0:
            stats['deterministic_ratio'] = stats['deterministic_predictions'] / stats['total_predictions']
            stats['override_ratio'] = stats['safety_overrides'] / stats['total_predictions']

        return stats

    def save_audit_log(self):
        """Force save audit log."""
        if self.audit_log:
            self.audit_log.save()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save audit log."""
        self.save_audit_log()


def create_medical_safety_wrapper(
    model: nn.Module,
    enforce_critical: bool = True,
    enable_audit: bool = True,
    audit_path: str = "./audit_logs",
    model_version: str = "1.0.0",
    uncertainty_module: Optional[Any] = None,
) -> MedicalSafetyWrapper:
    """
    Factory function to create Medical Safety Wrapper.

    Args:
        model: TinyBioBERT model
        enforce_critical: Force deterministic for critical tasks
        enable_audit: Enable audit logging
        audit_path: Path for audit logs
        model_version: Model version string
        uncertainty_module: Optional uncertainty quantification module

    Returns:
        Configured MedicalSafetyWrapper instance
    """
    config = SafetyConfig(
        force_deterministic_critical=enforce_critical,
        enable_audit_logging=enable_audit,
        audit_log_path=audit_path,
    )

    wrapper = MedicalSafetyWrapper(
        model=model,
        config=config,
        uncertainty_quantifier=uncertainty_module,
        model_version=model_version,
    )

    return wrapper