# Embedding FFNN — Model Experiments

**Project:** Chewy Spot-the-Bot  
**Model:** Feed-Forward Neural Network (FFNN)  
**Feature Representation:** Sentence embeddings (384 dimensions)  
**Dataset:** Expanded HC3 (70/15/15 train/validation/test split)

## Objective

Evaluate how neural network architecture, activation functions, learning rates, and batch sizes affect classification performance using sentence embeddings.

Prioritize ROC-AUC, macro-F1, validation loss, and generalization rather than accuracy alone.

---

## Experiment 1 — Baseline (64 → 32 ReLU)

### Configuration
- Input dimensions: 384
- Architecture: 384 → 64 → 32 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: 0.3 after each hidden layer
- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 64
- Maximum epochs: 50
- Early stopping: `val_loss`, patience=2, restore_best_weights=True
- Loss: Binary cross-entropy
- Class balancing: Class-weighted training
- Random seed: 42
- Trainable parameters: 26,753

### Results
- Validation accuracy: 0.9150
- Macro precision: 0.8982
- Macro recall: 0.9105
- Macro F1: 0.9038
- ROC-AUC: 0.9717

**Class-Specific Results:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Human | 0.95 | 0.92 | 0.94 |
| ChatGPT | 0.85 | 0.90 | 0.87 |

**Best validation loss:** Pending training logs  
**Training epochs:** Pending training logs  
**Confusion matrix:** Pending

### Observation
The baseline embedding FFNN achieved 91.50% validation accuracy, 0.9038 macro-F1, and 0.9717 ROC-AUC on HC3.

Performance was reasonably balanced across classes, with human recall of 0.92 and ChatGPT recall of 0.90.

Compared with the selected TF-IDF FFNN, the embedding model achieved lower HC3 validation performance but used substantially fewer trainable parameters.

These results establish a baseline for evaluating architecture, activation functions, and optimization changes.

### Next Experiment
Increase hidden-layer capacity from 64 → 32 to 128 → 64 while keeping all other hyperparameters fixed.

## Experiment 2 — Increased Architecture (128 → 64)

### Change
Increased hidden-layer capacity from 64 → 32 to 128 → 64 to evaluate whether a larger embedding-based FFNN improves classification performance.

### Configuration
- Input dimensions: 384
- Architecture: 384 → 128 → 64 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: 0.3
- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 64
- Maximum epochs: 50
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Loss: Binary cross-entropy
- Class balancing: Class-weighted training
- Training seed: 42
- Trainable parameters: 57,601

### Results
- Validation accuracy: 0.9136
- Macro precision: 0.8961
- Macro recall: 0.9101
- Macro F1: 0.9025
- ROC-AUC: 0.9717
- Best validation loss: 0.2099 (epoch 23)
- Training duration: 25 epochs

**Confusion Matrix:**
```text
[[7889  689]
 [ 401 3632]]
```

### Observation
Increasing network capacity produced nearly identical ROC-AUC to the baseline (approximately 0.9717), with slightly lower validation accuracy and macro-F1.

Validation loss reached its minimum at epoch 23 before early stopping at epoch 25.

The additional model capacity did not demonstrate a meaningful improvement in HC3 validation performance.

### Conclusion
The original 64 → 32 architecture remains the preferred embedding FFNN baseline based on the available classification metrics and smaller parameter count.

Further experiments will investigate whether activation functions and learning rate affect optimization and generalization more significantly than increased network capacity.

## Experiment 3 — GELU Activation (64 → 32)

### Change
Replaced ReLU with GELU in both hidden layers of the embedding FFNN while retaining the original architecture and training configuration.

### Configuration
- Input dimensions: 384
- Architecture: 384 → 128 → 64 → 1
- Hidden activation: GELU
- Output activation: Sigmoid
- Dropout: 0.3
- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 64
- Maximum epochs: 80
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Loss: Binary cross-entropy
- Class balancing: Class-weighted training
- Training seed: 42

### Results
- Validation accuracy: 0.8575
- Macro precision: 0.8335
- Macro recall: 0.8611
- Macro F1: 0.8434
- ROC-AUC: 0.9371
- Best validation loss: 0.3227 (epoch 18)
- Training duration: 20 epochs

### Observation
GELU substantially underperformed the ReLU baseline, decreasing validation accuracy by approximately 5.75 percentage points and ROC-AUC from 0.9717 to 0.9371.

Validation loss improved slowly during later epochs, reaching its minimum at epoch 18 before early stopping.

At the current learning rate, GELU did not demonstrate an improvement in embedding-based classification performance.

### Conclusion
ReLU remains the stronger activation for the embedding FFNN under the current configuration. GELU may benefit from different optimization settings, but further tuning should be prioritized based on the broader experimental goals.

## Experiment 4 — Increased Learning Rate (0.001)

### Change
Increased Adam's learning rate from 0.0001 to 0.001 to evaluate whether faster optimization improves convergence or validation performance.

### Configuration
- Input dimensions: 384
- Architecture: 384 → 128 → 64 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: 0.3
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Maximum epochs: 50
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Loss: Binary cross-entropy
- Class balancing: Class-weighted training
- Training seed: 42

### Results
- Validation accuracy: 0.9140
- Macro precision: 0.8963
- Macro recall: 0.9115
- Macro F1: 0.9031
- ROC-AUC: 0.9717
- Training duration: 6 epochs
- Best validation loss: Pending training log

### Observation
Increasing the learning rate by 10× substantially reduced the number of training epochs while maintaining nearly identical validation ROC-AUC and macro-F1.

The higher learning rate slightly reduced accuracy and macro-F1 compared with the baseline, but increased macro recall.

This suggests the embedding FFNN can reach comparable HC3 validation performance with fewer optimization updates using a higher learning rate.

### Conclusion
A learning rate of 0.001 is a promising efficiency alternative, although it did not improve overall validation performance. Further comparison of validation loss and training stability is needed before selecting it over the original learning rate.

## Experiment 5 — ROC-AUC-Based Early Stopping

### Change
Changed early stopping from monitoring validation loss to monitoring validation ROC-AUC, increasing patience from 2 to 5 epochs.

The objective was to determine whether selecting model weights based on ranking performance improves ROC-AUC and macro-F1.

### Configuration
- Input dimensions: 384
- Architecture: 384 → 128 → 64 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: 0.3
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Maximum epochs: 30
- Early stopping: `val_auc`, mode=max, patience=5, restore_best_weights=True
- Loss: Binary cross-entropy
- Class balancing: Class-weighted training
- Training seed: 42

### Results
- Validation accuracy: 0.9164
- Macro precision: 0.8988
- Macro recall: 0.9147
- Macro F1: 0.9059
- ROC-AUC: 0.9737
- Best validation AUC (Keras): 0.9732 (epoch 6)
- Training duration: 11 epochs

### Observation
Monitoring validation ROC-AUC rather than validation loss improved the final model's ranking performance and macro-F1.

The original loss-based early stopping selected epoch 4, while AUC-based stopping selected epoch 6, allowing the model to retain weights with stronger ranking performance.

Compared with the original embedding ReLU baseline, ROC-AUC increased from approximately 0.9717 to 0.9737, while macro-F1 improved from 0.9038 to 0.9059.

### Conclusion
AUC-based early stopping is a promising model-selection approach for the embedding FFNN when ranking performance is a priority.

However, the experiment also changed early-stopping patience, and the observed improvements remain specific to HC3 validation data. Additional external evaluation is needed to determine whether they translate to modern text.

## Experiment 6 — Reduced Architecture with AUC-Based Early Stopping

### Change
Reduced the embedding FFNN architecture from 128 → 64 to 64 → 32 while preserving the improved training configuration from Experiment 5.

The objective was to determine whether a smaller network could maintain or improve validation performance with fewer trainable parameters.

### Configuration
- Input dimensions: 384
- Architecture: 384 → 64 → 32 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: 0.3
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Maximum epochs: 30
- Early stopping: `val_auc`, mode=max, patience=5, restore_best_weights=True
- Loss: Binary cross-entropy
- Class balancing: Class-weighted training
- Training seed: 42
- Trainable parameters: 26,753

### Results
- Validation accuracy: 0.9176
- Macro precision: 0.9013
- Macro recall: 0.9130
- Macro F1: 0.9067
- ROC-AUC: 0.9737
- Best logged validation AUC: ~0.9733
- Minimum validation loss: 0.2090 (epoch 10)
- Training duration: 17 epochs

### Observation
The smaller architecture slightly improved validation accuracy and macro-F1 compared with Experiment 5 while maintaining virtually identical ROC-AUC.

Reducing model capacity from 57,601 to 26,753 parameters did not reduce ranking performance on HC3.

Validation loss began increasing after epoch 10, although validation ROC-AUC remained relatively stable.

### Conclusion
The 64 → 32 ReLU architecture is the current preferred embedding FFNN configuration due to its comparable ROC-AUC, slightly stronger macro-F1, and reduced complexity.

External evaluation is required to determine whether these improvements translate to modern text.

## Experiment 7 — Reduced Learning Rate with AUC-Based Early Stopping

### Change
Reduced the learning rate from 0.001 to 0.0001 while retaining the 64 → 32 architecture and AUC-based early stopping to evaluate whether slower optimization improves embedding FFNN performance.

### Configuration
- Architecture: 384 → 64 → 32 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: 0.3
- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 64
- Maximum epochs: 80
- Early stopping: `val_auc`, mode=max, patience=5, restore_best_weights=True
- Loss: Binary cross-entropy
- Class balancing: Class-weighted training
- Training seed: 42
- Trainable parameters: 26,753

### Results
- Validation accuracy: 0.9215
- Macro precision: 0.9063
- Macro recall: 0.9160
- Macro F1: 0.9109
- ROC-AUC: 0.9745
- Best logged validation AUC: ~0.9741 (epoch 73)
- Minimum validation loss: 0.2069 (epoch 59)
- Training duration: 78 epochs

### Observation
Reducing the learning rate improved validation accuracy, macro-F1, and ROC-AUC compared with the 0.001 learning-rate experiment using the same architecture and early-stopping strategy.

Training required substantially more epochs, with validation ROC-AUC improving gradually before plateauing near 0.974.

Although ranking performance continued improving slightly, validation loss stabilized earlier, suggesting diminishing returns from continued training.

### Conclusion
The 64 → 32 ReLU architecture with learning rate 0.0001 and AUC-based early stopping is the strongest embedding FFNN configuration tested so far on HC3.

Further model selection should prioritize external generalization rather than additional small improvements in HC3 validation performance.
