"""
Medical Uncertainty Quantification for TinyBioBERT.
Provides calibrated uncertainty estimates for medical predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    from sklearn.preprocessing import label_binarize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. AUROC/AUPRC metrics will be disabled.")


@dataclass
class UncertaintyDecomposition:
    """Decomposed uncertainty components."""
    total_uncertainty: float
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    predictive_entropy: float
    mutual_information: float


class MedicalMetrics:
    """
    Medical-specific evaluation metrics including AUROC and AUPRC.
    Critical for evaluating medical classification performance.
    """

    @staticmethod
    def compute_auroc(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        multi_class: str = 'ovr',
        average: str = 'macro',
    ) -> Dict[str, Any]:
        """
        Compute Area Under ROC Curve (AUROC).

        Args:
            y_true: True labels (shape: [n_samples] or [n_samples, n_classes])
            y_scores: Predicted probabilities (shape: [n_samples, n_classes])
            multi_class: Strategy for multi-class ('ovr' or 'ovo')
            average: Averaging strategy ('macro', 'micro', 'weighted')

        Returns:
            Dictionary containing:
                - auroc: Overall AUROC score
                - auroc_per_class: Per-class AUROC scores (if multi-class)
                - roc_curves: ROC curve data for visualization
        """
        if not SKLEARN_AVAILABLE:
            warnings.warn("AUROC computation requires scikit-learn")
            return {'auroc': None, 'auroc_per_class': None, 'roc_curves': None}

        results = {}

        # Handle binary classification
        if y_scores.shape[1] == 2:
            # Use positive class probabilities
            auroc = roc_auc_score(y_true, y_scores[:, 1])
            fpr, tpr, thresholds = roc_curve(y_true, y_scores[:, 1])
            results['auroc'] = auroc
            results['roc_curves'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

        # Handle multi-class classification
        else:
            # Binarize labels for OvR computation
            n_classes = y_scores.shape[1]
            y_true_binarized = label_binarize(y_true, classes=range(n_classes))

            # Overall AUROC
            try:
                auroc = roc_auc_score(
                    y_true_binarized,
                    y_scores,
                    multi_class=multi_class,
                    average=average
                )
                results['auroc'] = auroc
            except ValueError as e:
                warnings.warn(f"Could not compute multi-class AUROC: {e}")
                results['auroc'] = None

            # Per-class AUROC
            auroc_per_class = {}
            roc_curves = {}

            for i in range(n_classes):
                try:
                    auroc_i = roc_auc_score(y_true_binarized[:, i], y_scores[:, i])
                    auroc_per_class[f'class_{i}'] = auroc_i

                    fpr, tpr, thresholds = roc_curve(y_true_binarized[:, i], y_scores[:, i])
                    roc_curves[f'class_{i}'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
                except:
                    auroc_per_class[f'class_{i}'] = None

            results['auroc_per_class'] = auroc_per_class
            results['roc_curves'] = roc_curves

        return results

    @staticmethod
    def compute_auprc(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        average: str = 'macro',
    ) -> Dict[str, Any]:
        """
        Compute Area Under Precision-Recall Curve (AUPRC).
        Critical for imbalanced medical datasets.

        Args:
            y_true: True labels (shape: [n_samples] or [n_samples, n_classes])
            y_scores: Predicted probabilities (shape: [n_samples, n_classes])
            average: Averaging strategy ('macro', 'micro', 'weighted')

        Returns:
            Dictionary containing:
                - auprc: Overall AUPRC score
                - auprc_per_class: Per-class AUPRC scores (if multi-class)
                - pr_curves: Precision-recall curve data
        """
        if not SKLEARN_AVAILABLE:
            warnings.warn("AUPRC computation requires scikit-learn")
            return {'auprc': None, 'auprc_per_class': None, 'pr_curves': None}

        results = {}

        # Handle binary classification
        if y_scores.shape[1] == 2:
            auprc = average_precision_score(y_true, y_scores[:, 1])
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores[:, 1])
            results['auprc'] = auprc
            results['pr_curves'] = {'precision': precision, 'recall': recall, 'thresholds': thresholds}

        # Handle multi-class classification
        else:
            n_classes = y_scores.shape[1]
            y_true_binarized = label_binarize(y_true, classes=range(n_classes))

            # Overall AUPRC
            try:
                auprc = average_precision_score(
                    y_true_binarized,
                    y_scores,
                    average=average
                )
                results['auprc'] = auprc
            except ValueError as e:
                warnings.warn(f"Could not compute multi-class AUPRC: {e}")
                results['auprc'] = None

            # Per-class AUPRC
            auprc_per_class = {}
            pr_curves = {}

            for i in range(n_classes):
                try:
                    auprc_i = average_precision_score(y_true_binarized[:, i], y_scores[:, i])
                    auprc_per_class[f'class_{i}'] = auprc_i

                    precision, recall, thresholds = precision_recall_curve(
                        y_true_binarized[:, i], y_scores[:, i]
                    )
                    pr_curves[f'class_{i}'] = {
                        'precision': precision,
                        'recall': recall,
                        'thresholds': thresholds
                    }
                except:
                    auprc_per_class[f'class_{i}'] = None

            results['auprc_per_class'] = auprc_per_class
            results['pr_curves'] = pr_curves

        return results

    @staticmethod
    def compute_sensitivity_specificity(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        per_class: bool = True,
    ) -> Dict[str, float]:
        """
        Compute sensitivity (recall) and specificity.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            per_class: Whether to compute per-class metrics

        Returns:
            Dictionary with sensitivity and specificity values
        """
        results = {}

        if per_class and len(np.unique(y_true)) > 2:
            # Multi-class case
            n_classes = len(np.unique(y_true))

            for class_idx in range(n_classes):
                # Treat as one-vs-rest
                y_true_binary = (y_true == class_idx).astype(int)
                y_pred_binary = (y_pred == class_idx).astype(int)

                tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
                tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
                fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
                fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                results[f'sensitivity_class_{class_idx}'] = sensitivity
                results[f'specificity_class_{class_idx}'] = specificity

            # Overall metrics (macro average)
            results['sensitivity_macro'] = np.mean([v for k, v in results.items() if 'sensitivity' in k])
            results['specificity_macro'] = np.mean([v for k, v in results.items() if 'specificity' in k])

        else:
            # Binary case
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            results['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
            results['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

        return results


class CalibrationMetrics:
    """
    Calibration metrics for medical predictions.
    """

    @staticmethod
    def expected_calibration_error(
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            predictions: Predicted labels
            labels: True labels
            confidences: Prediction confidences
            n_bins: Number of bins for calibration

        Returns:
            ECE value (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            # Find samples in this bin
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])

            if np.sum(in_bin) > 0:
                # Accuracy in bin
                bin_accuracy = np.mean(predictions[in_bin] == labels[in_bin])

                # Average confidence in bin
                bin_confidence = np.mean(confidences[in_bin])

                # Weighted difference
                ece += np.sum(in_bin) * np.abs(bin_accuracy - bin_confidence)

        return ece / len(predictions)

    @staticmethod
    def maximum_calibration_error(
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        Args:
            predictions: Predicted labels
            labels: True labels
            confidences: Prediction confidences
            n_bins: Number of bins

        Returns:
            MCE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])

            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(predictions[in_bin] == labels[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                mce = max(mce, np.abs(bin_accuracy - bin_confidence))

        return mce

    @staticmethod
    def brier_score(
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Compute Brier score.

        Args:
            probabilities: Predicted probabilities
            labels: True labels (one-hot encoded)

        Returns:
            Brier score (lower is better)
        """
        return np.mean((probabilities - labels) ** 2)


class MedicalUncertaintyQuantifier:
    """
    Uncertainty quantification for medical NLP predictions.
    Uses P-bit stochasticity for uncertainty estimation.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        temperature: float = 1.0,
        use_dropout_uncertainty: bool = True,
    ):
        """
        Initialize uncertainty quantifier.

        Args:
            model: TinyBioBERT model
            n_samples: Number of forward passes for uncertainty
            temperature: Temperature for calibration
            use_dropout_uncertainty: Use dropout at inference for uncertainty
        """
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.use_dropout_uncertainty = use_dropout_uncertainty

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs

        Returns:
            Dictionary containing:
                - predictions: Most likely predictions
                - confidences: Prediction confidences
                - uncertainties: Uncertainty estimates
                - logits_mean: Mean logits across samples
                - logits_var: Variance of logits
        """
        device = input_ids.device
        batch_size, seq_length = input_ids.shape

        # Enable dropout for uncertainty if requested
        if self.use_dropout_uncertainty:
            self.model.train()  # Enable dropout
        else:
            self.model.eval()

        # Multiple forward passes
        all_logits = []

        for _ in range(self.n_samples):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs['logits']
            all_logits.append(logits)

        # Stack all logits
        all_logits = torch.stack(all_logits)  # (n_samples, batch, seq, classes)

        # Compute mean and variance
        logits_mean = all_logits.mean(dim=0)
        logits_var = all_logits.var(dim=0)

        # Apply temperature scaling
        logits_scaled = logits_mean / self.temperature

        # Get probabilities
        probs = F.softmax(logits_scaled, dim=-1)

        # Get predictions and confidences
        confidences, predictions = torch.max(probs, dim=-1)

        # Compute uncertainty (entropy of mean distribution)
        uncertainties = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)

        return {
            'predictions': predictions,
            'confidences': confidences,
            'uncertainties': uncertainties,
            'logits_mean': logits_mean,
            'logits_var': logits_var,
            'probabilities': probs,
        }

    def decompose_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> UncertaintyDecomposition:
        """
        Decompose uncertainty into epistemic and aleatoric components.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Decomposed uncertainty components
        """
        # Get predictions with uncertainty
        results = self.predict_with_uncertainty(
            input_ids, attention_mask
        )

        probs = results['probabilities']  # Mean probabilities

        # Predictive entropy (total uncertainty)
        predictive_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()

        # Expected entropy (aleatoric uncertainty)
        # For medical NER, this represents inherent ambiguity in the data
        expected_entropy = results['uncertainties'].mean()

        # Mutual information (epistemic uncertainty)
        # Represents model uncertainty that can be reduced with more data
        mutual_information = predictive_entropy - expected_entropy

        # Ensure non-negative
        mutual_information = max(0, mutual_information.item())

        return UncertaintyDecomposition(
            total_uncertainty=predictive_entropy.item(),
            epistemic_uncertainty=mutual_information,
            aleatoric_uncertainty=expected_entropy.item(),
            predictive_entropy=predictive_entropy.item(),
            mutual_information=mutual_information,
        )

    def calibrate_temperature(
        self,
        val_loader: torch.utils.data.DataLoader,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """
        Calibrate temperature using validation data.

        Args:
            val_loader: Validation data loader
            max_iter: Maximum optimization iterations
            lr: Learning rate for temperature optimization

        Returns:
            Optimal temperature
        """
        print("Calibrating temperature for uncertainty...")

        # Temperature as optimizable parameter
        temperature = nn.Parameter(torch.ones(1) * self.temperature)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        def eval_calibration():
            """Evaluate calibration with current temperature."""
            total_loss = 0.0
            n_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    # Move to device
                    input_ids = batch['input_ids'].to(next(self.model.parameters()).device)
                    labels = batch['labels'].to(next(self.model.parameters()).device)
                    attention_mask = batch.get('attention_mask')

                    if attention_mask is not None:
                        attention_mask = attention_mask.to(input_ids.device)

                    # Get predictions
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']

                    # Apply temperature
                    logits_scaled = logits / temperature

                    # Compute NLL loss (good proxy for calibration)
                    loss = F.cross_entropy(
                        logits_scaled.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                    )

                    total_loss += loss
                    n_batches += 1

            return total_loss / n_batches

        # Optimize temperature
        def closure():
            optimizer.zero_grad()
            loss = eval_calibration()
            loss.backward()
            return loss

        optimizer.step(closure)

        # Update temperature
        self.temperature = temperature.item()
        print(f"Calibrated temperature: {self.temperature:.3f}")

        return self.temperature

    def evaluate_calibration(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate calibration metrics.

        Args:
            predictions: Model predictions
            labels: True labels
            confidences: Prediction confidences

        Returns:
            Dictionary of calibration metrics
        """
        ece = CalibrationMetrics.expected_calibration_error(
            predictions, labels, confidences
        )

        mce = CalibrationMetrics.maximum_calibration_error(
            predictions, labels, confidences
        )

        # Convert to one-hot for Brier score
        n_classes = len(np.unique(labels))
        labels_onehot = np.eye(n_classes)[labels]
        predictions_onehot = np.eye(n_classes)[predictions]

        brier = CalibrationMetrics.brier_score(
            predictions_onehot, labels_onehot
        )

        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier,
        }

    def evaluate_medical_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate comprehensive medical metrics including AUROC and AUPRC.

        Args:
            y_true: True labels
            y_scores: Predicted probabilities (from model)
            y_pred: Predicted labels (optional, will be computed from y_scores if not provided)

        Returns:
            Dictionary containing all medical metrics
        """
        if y_pred is None:
            y_pred = np.argmax(y_scores, axis=1)

        metrics = {}

        # AUROC metrics
        auroc_results = MedicalMetrics.compute_auroc(y_true, y_scores)
        metrics.update({f'auroc_{k}': v for k, v in auroc_results.items() if 'curves' not in k})

        # AUPRC metrics
        auprc_results = MedicalMetrics.compute_auprc(y_true, y_scores)
        metrics.update({f'auprc_{k}': v for k, v in auprc_results.items() if 'curves' not in k})

        # Sensitivity and Specificity
        sens_spec = MedicalMetrics.compute_sensitivity_specificity(y_true, y_pred)
        metrics.update(sens_spec)

        # Calibration metrics
        confidences = np.max(y_scores, axis=1)
        calibration = self.evaluate_calibration(y_pred, y_true, confidences)
        metrics.update(calibration)

        return metrics

    @torch.no_grad()
    def evaluate_on_dataset(
        self,
        data_loader: torch.utils.data.DataLoader,
        return_curves: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model on entire dataset with medical metrics.

        Args:
            data_loader: DataLoader for evaluation
            return_curves: Whether to return ROC/PR curves

        Returns:
            Comprehensive evaluation results
        """
        device = next(self.model.parameters()).device

        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_uncertainties = []

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask')

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Get predictions with uncertainty
            results = self.predict_with_uncertainty(
                input_ids, attention_mask
            )

            # Collect results
            all_predictions.append(results['predictions'].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(results['probabilities'].cpu().numpy())
            all_uncertainties.append(results['uncertainties'].cpu().numpy())

        # Concatenate all results
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        all_probabilities = np.concatenate(all_probabilities)
        all_uncertainties = np.concatenate(all_uncertainties)

        # Flatten for token-level evaluation
        mask = all_labels != -100  # Ignore padding
        all_predictions = all_predictions[mask]
        all_labels = all_labels[mask]
        all_probabilities = all_probabilities[mask]
        all_uncertainties = all_uncertainties[mask]

        # Compute medical metrics
        metrics = self.evaluate_medical_metrics(
            all_labels,
            all_probabilities,
            all_predictions
        )

        # Add uncertainty statistics
        metrics['mean_uncertainty'] = np.mean(all_uncertainties)
        metrics['std_uncertainty'] = np.std(all_uncertainties)

        # Optionally return curves for visualization
        if return_curves:
            auroc_results = MedicalMetrics.compute_auroc(all_labels, all_probabilities)
            auprc_results = MedicalMetrics.compute_auprc(all_labels, all_probabilities)
            metrics['roc_curves'] = auroc_results.get('roc_curves')
            metrics['pr_curves'] = auprc_results.get('pr_curves')

        return metrics

    def get_high_uncertainty_examples(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify high-uncertainty predictions for active learning.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            threshold: Uncertainty threshold

        Returns:
            Indices and uncertainty values of high-uncertainty examples
        """
        results = self.predict_with_uncertainty(input_ids, attention_mask)
        uncertainties = results['uncertainties']

        # Find high uncertainty positions
        high_uncertainty_mask = uncertainties > threshold

        # Get indices
        indices = torch.where(high_uncertainty_mask)

        return indices, uncertainties[high_uncertainty_mask]

    def compute_confidence_intervals(
        self,
        logits_samples: torch.Tensor,
        confidence_level: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute confidence intervals for predictions.

        Args:
            logits_samples: Samples of logits (n_samples, batch, seq, classes)
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Lower and upper bounds of confidence intervals
        """
        # Convert to probabilities
        probs_samples = F.softmax(logits_samples, dim=-1)

        # Compute percentiles
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = torch.percentile(probs_samples, lower_percentile, dim=0)
        upper_bound = torch.percentile(probs_samples, upper_percentile, dim=0)

        return lower_bound, upper_bound


class MedicalRiskAssessment:
    """
    Risk assessment for medical predictions.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.5,
        confidence_threshold: float = 0.8,
    ):
        """
        Initialize risk assessment.

        Args:
            uncertainty_threshold: Threshold for high uncertainty
            confidence_threshold: Threshold for high confidence
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold

    def assess_prediction_risk(
        self,
        predictions: torch.Tensor,
        confidences: torch.Tensor,
        uncertainties: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Assess risk level of predictions.

        Args:
            predictions: Model predictions
            confidences: Prediction confidences
            uncertainties: Uncertainty estimates

        Returns:
            Risk assessment results
        """
        # Risk categories
        high_risk = (uncertainties > self.uncertainty_threshold) | (
            confidences < self.confidence_threshold
        )
        medium_risk = (
            (uncertainties > 0.3) & (uncertainties <= self.uncertainty_threshold)
        ) | (
            (confidences >= 0.6) & (confidences < self.confidence_threshold)
        )
        low_risk = ~(high_risk | medium_risk)

        return {
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'risk_scores': uncertainties / confidences,  # Higher score = higher risk
        }

    def recommend_human_review(
        self,
        risk_assessment: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Recommend which predictions need human review.

        Args:
            risk_assessment: Risk assessment results

        Returns:
            Boolean mask for predictions requiring review
        """
        # Recommend review for high-risk predictions
        return risk_assessment['high_risk']


def create_medical_uncertainty_quantifier(
    model: nn.Module,
    n_samples: int = 10,
    calibrate: bool = True,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
) -> MedicalUncertaintyQuantifier:
    """
    Factory function to create medical uncertainty quantifier.

    Args:
        model: TinyBioBERT model
        n_samples: Number of samples for uncertainty
        calibrate: Whether to calibrate temperature
        val_loader: Validation data for calibration

    Returns:
        Configured uncertainty quantifier
    """
    quantifier = MedicalUncertaintyQuantifier(
        model=model,
        n_samples=n_samples,
    )

    # Calibrate if requested
    if calibrate and val_loader is not None:
        quantifier.calibrate_temperature(val_loader)

    return quantifier