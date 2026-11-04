# Neural Network Experiments

This document records the iterative development and tuning of the
feed-forward neural network (FFNN). Each experiment changes a limited
number of hyperparameters to evaluate their effect on validation
performance and overfitting.

## Experiment 1 — Baseline FFNN

### Configuration
- Architecture: 128 → 64 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: None
- Optimizer: Adam
- Learning rate: 0.001 (default)
- Loss: Binary cross-entropy
- Batch size: 128
- Early stopping: val_loss, patience=2

### Results
- Best epoch: 1
- Best validation accuracy: 0.9878
- Best validation loss: 0.0361

### Observation
Training performance approached perfection almost immediately while
validation loss increased after the first epoch, indicating rapid
overfitting.


## Experiment 2 — Add Dropout

### Change
Added 0.3 dropout after each hidden layer.

### Configuration
- Architecture: 128 → 64 → 1
- Dropout: 0.3
- Learning rate: 0.001
- Batch size: 128

### Results
- Best epoch: 1
- Validation accuracy: 0.9890
- Validation AUC: 0.9987
- Validation loss: 0.0317

### Observation
Dropout improved validation performance, but training performance still
approached perfection almost immediately and validation performance began
declining after the first epoch.


## Experiment 3 — Reduce Network Capacity

### Change
Reduced hidden layers from 128 → 64 to 64 → 32 while retaining dropout.

### Results
- Best epoch: 1
- Validation accuracy: 0.9885
- Validation AUC: 0.9988
- Validation loss: 0.0310

### Observation
The smaller network reduced parameter count and training time while
slightly improving validation loss and AUC. Rapid overfitting remained.


## Experiment 4 — Reduce Learning Rate

### Change
Reduced Adam learning rate from 0.001 to 0.0001.

### Configuration
- Architecture: 64 → 32 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: 0.3
- Optimizer: Adam
- Learning rate: 0.0001
- Loss: Binary cross-entropy
- Batch size: 128
- Early stopping: val_loss, patience=2, restore_best_weights=True

### Results
- Best epoch: 8
- Validation accuracy: 0.9912
- Macro precision: 0.9917
- Macro recall: 0.9881
- Macro F1: 0.9899
- ROC-AUC: 0.9993
- Validation loss: 0.0272
- Confusion matrix: [[5700, 19], [55, 2643]]

### Observation
The lower learning rate produced substantially more gradual convergence.
Validation loss continued improving through epoch 8 rather than beginning
to deteriorate after the first epoch. This configuration achieved the
strongest validation performance of the FFNN experiments.

## Experiment 5 — Updated Baseline After Dataset Expansion

### Change
Re-evaluated the existing FFNN configuration after expanding and updating the HC3 dataset with additional human Reddit responses.

The model architecture and hyperparameters remain unchanged from Experiment 4. Results from earlier experiments are retained for historical comparison but are not directly comparable because the dataset has changed.

### Configuration
- Architecture: 64 → 32 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: 0.3 after each hidden layer
- Optimizer: Adam
- Learning rate: 0.0001
- Loss: Binary cross-entropy
- Batch size: 128
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 with additional human Reddit responses
- Data split: 70% training / 15% validation / 15% test
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9895
- Macro precision: 0.9896
- Macro recall: 0.9863
- Macro F1: 0.9879
- ROC-AUC: 0.9992

**Confusion Matrix:**
```text
[[8538   40]
 [  92 3941]]
```

### Observation
The FFNN maintained strong validation performance after the HC3 dataset expansion, achieving 98.95% accuracy and a ROC-AUC of 0.9992.

The high ROC-AUC indicates excellent discrimination between human and AI-generated responses within the expanded HC3 validation dataset.

These results establish the updated baseline for subsequent hyperparameter experiments, including alternative hidden-layer activation functions.

### Next Steps
- Experiment 6: Leaky ReLU
- Experiment 7: ELU
- Experiment 8: GELU
- Experiment 9: Softplus

Keep the architecture, learning rate, dropout, batch size, and data splits consistent across activation experiments to isolate the effect of each activation function.

Evaluate selected models against the modern external dataset to assess generalization beyond HC3.

## Experiment 8 — GELU Activation

### Change
Replaced ReLU with GELU in both hidden layers of the TF-IDF FFNN to evaluate whether a smoother activation function improves performance.

### Configuration
- Architecture: 64 → 32 → 1
- Hidden activation: GELU
- Output activation: Sigmoid
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 128
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9891
- Macro precision: 0.9884
- Macro recall: 0.9864
- Macro F1: 0.9874
- ROC-AUC: 0.9992

**Confusion Matrix:**
```text
[[8525   53]
 [  85 3948]]
```

### Observation
GELU performed nearly identically to the ReLU baseline, with a slight decrease in accuracy and macro F1 while maintaining the same ROC-AUC of 0.9992.

GELU correctly identified 7 more AI responses but incorrectly classified 13 additional human responses compared to ReLU.

Overall, GELU did not provide a meaningful improvement over the existing ReLU configuration.

## Experiment 9 — Softplus Activation

### Change
Replaced GELU with Softplus in both hidden layers of the TF-IDF FFNN to evaluate its effect on classification performance.

### Configuration
- Architecture: 64 → 32 → 1
- Hidden activation: Softplus
- Output activation: Sigmoid
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 128
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9858
- Macro precision: 0.9840
- Macro recall: 0.9834
- Macro F1: 0.9837
- ROC-AUC: 0.9987
- Training duration: 20 epochs (maximum reached)

**Confusion Matrix:**
```text
[[8493   85]
 [  94 3939]]
```

### Observation
Softplus produced slightly lower validation performance than ReLU and GELU, but showed a different training pattern. It completed all 20 epochs, whereas ReLU and GELU typically stopped around epoch 12.

This suggests Softplus may converge more slowly or continue improving later in training. Its validation accuracy and macro-F1 were lower, and human false positives increased compared with the ReLU baseline.

Further examination of the training and validation loss curves would help determine whether additional epochs are worthwhile.

## Experiment 10 — Softplus Extended Training (35 Epochs)

### Change
Extended Softplus training from 20 to 35 maximum epochs to investigate whether additional training improves convergence and validation performance.

### Configuration
- Architecture: 64 → 32 → 1
- Hidden activation: Softplus
- Output activation: Sigmoid
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 128
- Maximum epochs: 35
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9883
- Macro precision: 0.9875
- Macro recall: 0.9857
- Macro F1: 0.9866
- ROC-AUC: 0.9990
- Final validation loss: 0.0328
- Training duration: 35 epochs (maximum reached)

**Confusion Matrix:**
```text
[[8518   60]
 [  87 3946]]
```

### Observation
Extending Softplus training to 35 epochs improved accuracy, macro-F1, and ROC-AUC compared with the 20-epoch experiment.

Validation loss decreased consistently throughout training, indicating that Softplus had not yet reached its lowest validation loss within the allotted epochs.

Although performance approached the ReLU baseline, Softplus still achieved slightly lower overall validation accuracy and ROC-AUC. Further training may help determine whether performance eventually plateaus.

## Experiment 11 — Softplus Extended Training (50 Epochs)

### Change
Extended Softplus training from 35 to 50 maximum epochs to examine convergence and whether additional training improves validation performance.

### Configuration
- Architecture: 64 → 32 → 1
- Hidden activation: Softplus
- Output activation: Sigmoid
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 128
- Maximum epochs: 50
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9890
- Macro precision: 0.9880
- Macro recall: 0.9866
- Macro F1: 0.9873
- ROC-AUC: 0.9991
- Final validation loss: 0.0306
- Training duration: 50 epochs (maximum reached)

**Confusion Matrix:**
```text
[[8519   59]
 [  80 3953]]
```

### Observation
Softplus continued improving with extended training, increasing validation accuracy to 98.90% and ROC-AUC to 0.9991.

Validation loss decreased throughout all 50 epochs but began leveling off toward the end. Performance approached the ReLU baseline, although Softplus required substantially more training time.

Softplus identified more AI-generated responses than ReLU but produced more human false positives. Overall, ReLU remains the more efficient baseline on HC3.

## Experiment 12 — Reduced Architecture and Random Seed Stability

### Change
Reduced the TF-IDF FFNN architecture from 64 → 32 to 32 → 16 hidden neurons, cutting trainable parameters approximately in half.

Tested three different training seeds (42, 123, 2026) to evaluate whether the smaller network produces consistent results.

### Configuration
- Architecture: 32 → 16 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Trainable parameters: 3,200,577 (previously 6,402,177)
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 128
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Dataset split: Fixed
- Training seeds: 42, 123, 2026
- Class balancing: Class-weighted training

### Results

| Metric | Seed 42 | Seed 123 | Seed 2026 |
|---|---|---|---|
| Validation accuracy | 0.9895 | 0.9896 | 0.9890 |
| Macro precision | 0.9896 | 0.9894 | 0.9883 |
| Macro recall | 0.9863 | 0.9867 | 0.9864 |
| Macro F1 | 0.9879 | 0.9880 | 0.9873 |
| ROC-AUC | 0.9992 | 0.9992 | 0.9992 |
| Epochs | 14 | 15 | 15 |

**Confusion Matrices:**

Seed 42:
```text
[[8538   40]
 [  92 3941]]
```

Seed 123:
```text
[[8533   45]
 [  86 3947]]
```

Seed 2026:
```text
[[8523   55]
 [  84 3949]]
```

### Observation
Reducing the architecture by approximately 50% maintained nearly identical HC3 validation performance compared with the original 64 → 32 ReLU baseline.

Results remained consistent across three training seeds, averaging 98.94% validation accuracy, 0.9877 macro-F1, and 0.9992 ROC-AUC.

All three runs converged within 14–15 epochs. Although the confusion matrices varied slightly, overall performance was stable.

The smaller architecture appears to be a promising, more parameter-efficient alternative.

## Experiment 13 — Increased Dropout (0.5)

### Change
Increased dropout from 0.3 to 0.5 in both hidden layers of the reduced TF-IDF FFNN architecture to evaluate the effect of stronger regularization.

### Configuration
- Architecture: 32 → 16 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: 0.5
- Learning rate: 0.0001
- Batch size: 128
- Maximum epochs: 20
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9891
- Macro precision: 0.9885
- Macro recall: 0.9863
- Macro F1: 0.9874
- ROC-AUC: 0.9991
- Best validation loss: 0.0313
- Training duration: 15 epochs

**Confusion Matrix:**
```text
[[8526   52]
 [  86 3947]]
```

### Observation
Increasing dropout to 0.5 slightly reduced validation accuracy and macro-F1 compared with dropout 0.3, while ROC-AUC remained nearly identical.

Training progressed more gradually, suggesting stronger regularization affected learning speed. However, it did not produce a meaningful improvement in HC3 validation performance.

Dropout 0.3 remains the stronger baseline for the reduced architecture.

## Experiment 14 — Reduced Dropout (0.2)

### Change
Reduced dropout from 0.3 to 0.2 in both hidden layers of the TF-IDF FFNN to evaluate whether less regularization improves learning and validation performance.

### Configuration
- Architecture: 32 → 16 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: 0.2
- Learning rate: 0.0001
- Batch size: 128
- Maximum epochs: 20
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9891
- Macro precision: 0.9888
- Macro recall: 0.9860
- Macro F1: 0.9874
- ROC-AUC: 0.9992
- Best validation loss: 0.0302 (epoch 12)
- Training duration: 14 epochs

**Confusion Matrix:**
```text
[[8530   48]
 [  90 3943]]
```

### Observation
Reducing dropout to 0.2 accelerated training but did not improve validation accuracy or macro-F1 compared with the 0.3 baseline.

Validation loss began increasing after epoch 12 while training loss continued decreasing, suggesting the model was beginning to overfit.

Overall, dropout 0.3 remains the preferred configuration based on the current HC3 validation experiments.

## Experiment 15 — Additional Hidden Layer (32 → 16 → 8)

### Change
Added a third hidden layer with 8 neurons to the reduced TF-IDF FFNN architecture to evaluate whether additional depth improves classification performance.

### Configuration
- Architecture: 32 → 16 → 8 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Trainable parameters: 3,200,705
- Dropout: 0.3 after each hidden layer
- Learning rate: 0.0001
- Batch size: 128
- Maximum epochs: 20
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9885
- Macro precision: 0.9878
- Macro recall: 0.9857
- Macro F1: 0.9868
- ROC-AUC: 0.9992
- Best validation loss: 0.0320 (epoch 8)
- Training duration: 10 epochs

**Confusion Matrix:**
```text
[[8522   56]
 [  89 3944]]
```

### Observation
Adding an additional hidden layer slightly reduced validation accuracy and macro-F1 compared with the 32 → 16 baseline, while ROC-AUC remained unchanged at 0.9992.

The model converged more quickly, reaching its lowest validation loss at epoch 8 before early stopping at epoch 10.

Additional depth did not improve HC3 validation performance, suggesting the simpler 32 → 16 architecture is sufficient for the current dataset.

## Experiment 16 — Reduced Learning Rate (0.00005)

### Change
Reduced the learning rate from 0.0001 to 0.00005 using the 32 → 16 TF-IDF FFNN architecture to examine its effect on convergence and validation performance.

### Configuration
- Architecture: 32 → 16 → 1
- Hidden activation: ReLU
- Output activation: Sigmoid
- Dropout: 0.3
- Optimizer: Adam
- Learning rate: 0.00005
- Batch size: 128
- Maximum epochs: 30
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9889
- Macro precision: 0.9888
- Macro recall: 0.9857
- Macro F1: 0.9872
- ROC-AUC: 0.9992
- Best validation loss: 0.0304 (epoch 22)
- Training duration: 24 epochs

**Confusion Matrix:**
```text
[[8532   46]
 [  94 3939]]
```

### Observation
Reducing the learning rate extended training from approximately 14 to 24 epochs while maintaining nearly identical validation performance.

The model achieved a slightly lower minimum validation loss but slightly lower accuracy and macro-F1 compared with the 0.0001 baseline.

The smaller learning rate slowed convergence without demonstrating a meaningful improvement in HC3 classification performance.

## Experiment 17 — GELU Activation (32 → 16)

### Change
Replaced ReLU with GELU in both hidden layers of the reduced TF-IDF FFNN to compare activation functions using the same architecture.

### Configuration
- Architecture: 32 → 16 → 1
- Hidden activation: GELU
- Output activation: Sigmoid
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 128
- Maximum epochs: 30
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9888
- Macro precision: 0.9884
- Macro recall: 0.9859
- Macro F1: 0.9871
- ROC-AUC: 0.9992
- Best validation loss: 0.0307 (epoch 14)
- Training duration: 16 epochs

**Confusion Matrix:**
```text
[[8527   51]
 [  90 3943]]
```

### Observation
GELU achieved nearly identical HC3 validation performance to ReLU, including the same ROC-AUC of 0.9992.

ReLU maintained slightly higher accuracy and macro-F1, with nine fewer total classification errors.

GELU converged in 16 epochs compared with approximately 14 for ReLU. The results do not demonstrate a meaningful advantage for GELU on HC3, but it remains a viable alternative for external evaluation.

## Experiment 18 — Softplus Activation (32 → 16)

### Change
Replaced ReLU with Softplus in both hidden layers of the reduced TF-IDF FFNN to compare activation functions using the same architecture.

### Configuration
- Architecture: 32 → 16 → 1
- Hidden activation: Softplus
- Output activation: Sigmoid
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 128
- Maximum epochs: 50
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9885
- Macro precision: 0.9880
- Macro recall: 0.9856
- Macro F1: 0.9868
- ROC-AUC: 0.9990
- Best validation loss: 0.0329 (epoch 50)
- Training duration: 50 epochs (maximum reached)

**Confusion Matrix:**
```text
[[8524   54]
 [  91 3942]]
```

### Observation
Softplus achieved strong HC3 validation performance but slightly underperformed ReLU and GELU on accuracy, macro-F1, and ROC-AUC.

Training progressed substantially

## Experiment 19 — Mixed ReLU + Softplus Activation (32 → 16)

### Change
Combined ReLU and Softplus activation functions in the reduced TF-IDF FFNN to investigate whether mixing activations improves validation performance compared with using either activation independently.

### Configuration
- Architecture: 32 → 16 → 1
- First hidden layer: ReLU (32 neurons)
- Second hidden layer: Softplus (16 neurons)
- Output activation: Sigmoid
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 128
- Maximum epochs: 30
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9891
- Macro precision: 0.9893
- Macro recall: 0.9857
- Macro F1: 0.9875
- ROC-AUC: 0.9991
- Best validation loss: 0.0316 (epoch 15)
- Training duration: 17 epochs

**Confusion Matrix:**
```text
[[8537   41]
 [  96 3937]]
```

### Observation
Combining ReLU and Softplus produced stronger validation accuracy and macro-F1 than Softplus alone, approaching the performance of the ReLU-only baseline.

Compared with the 32 → 16 ReLU baseline, the mixed-activation model produced nearly identical ROC-AUC (0.9991 vs. 0.9992) and only one additional human false positive (41 vs. 40).

Validation loss reached its minimum at epoch 15 before increasing slightly, triggering early stopping at epoch 17.

The hybrid demonstrated competitive HC3 performance, but did not outperform ReLU alone in accuracy, macro-F1, ROC-AUC, or minimum validation loss.

### Conclusion
The mixed ReLU + Softplus architecture is a viable alternative, demonstrating that different activation functions can be combined successfully within the same neural network.

However, the ReLU-only 32 → 16 architecture remains the preferred TF-IDF configuration based on validation performance and training efficiency. External evaluation is needed to determine whether mixed activations provide any generalization advantage.

## Experiment 20 — Reduced Batch Size (64)

### Change
Reduced batch size from 128 to 64 in the mixed-activation TF-IDF FFNN to evaluate the effect of more frequent gradient updates on convergence and validation performance.

### Configuration
- Architecture: 32 → 16 → 1
- Hidden activations: ReLU → Softplus
- Output activation: Sigmoid
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 64
- Maximum epochs: 30
- Early stopping: val_loss, patience=2, restore_best_weights=True
- Dataset: Expanded HC3 (70/15/15 split)
- Class balancing: Class-weighted training

### Results
- Validation accuracy: 0.9891
- Macro precision: 0.9888
- Macro recall: 0.9860
- Macro F1: 0.9874
- ROC-AUC: 0.9992
- Best validation loss: 0.0311 (epoch 10)
- Training duration: 12 epochs

**Confusion Matrix:**
```text
[[8530   48]
 [  90 3943]]
```

### Observation
Reducing batch size to 64 resulted in similar validation accuracy and macro-F1 compared with batch size 128.

ROC-AUC improved slightly from 0.9991 to 0.9992, while minimum validation loss decreased from 0.0316 to 0.0311.

The model converged in fewer epochs but required more optimizer steps overall. The results suggest batch size has a modest influence on optimization behavior without materially changing HC3 classification performance.

---

## TF-IDF FFNN — Top 3 Model Candidates

### Summary

After 20 experiments exploring activation functions, network architectures, dropout, learning rates, batch sizes, and random seed stability, three configurations were shortlisted for further evaluation.

Selection considered **ROC-AUC, macro-F1, validation loss, classification errors, model complexity, and training efficiency**, rather than accuracy alone.

### 1. Primary Candidate — ReLU (32 → 16)

**Best overall performance and efficiency.**

- Architecture: 32 → 16 → 1
- Activation: ReLU → ReLU → Sigmoid
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 128
- Training seed: 42
- Early stopping: `val_loss`, patience=2

**Validation Results**
- Accuracy: 0.9895
- Macro-F1: 0.9879
- ROC-AUC: 0.9992
- Best validation loss: ~0.0305
- Training epochs: 14
- Trainable parameters: 3,200,577

**Why selected:** Matched the original 64 → 32 architecture's performance with approximately half the parameters. Demonstrated consistent results across three training seeds (42, 123, 2026).

### 2. Secondary Candidate — ReLU + Softplus (32 → 16)

**Strong mixed-activation alternative.**

- Architecture: 32 → 16 → 1
- Activation: ReLU → Softplus → Sigmoid
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 64
- Training seed: 42
- Early stopping: `val_loss`, patience=2

**Validation Results**
- Accuracy: 0.9891
- Macro-F1: 0.9874
- ROC-AUC: 0.9992
- Best validation loss: 0.0311
- Training epochs: 12
- Trainable parameters: 3,200,577

**Why selected:** Matched the primary candidate's reported ROC-AUC and produced competitive macro-F1. Combining different activation functions provides an alternative architecture worth evaluating for generalization.

### 3. Third Candidate — GELU (64 → 32)

**Alternative activation with strong ranking performance.**

- Architecture: 64 → 32 → 1
- Activation: GELU → GELU → Sigmoid
- Dropout: 0.3
- Learning rate: 0.0001
- Batch size: 128
- Early stopping: `val_loss`, patience=2

**Validation Results**
- Accuracy: 0.9891
- Macro-F1: 0.9874
- ROC-AUC: 0.9992
- Training epochs: ~12
- Trainable parameters: ~6.4 million

**Why selected:** Achieved nearly identical ROC-AUC to ReLU and provides a larger-network GELU comparison. Additional testing is needed to determine whether it generalizes differently.

### Overall Observation

All three shortlisted models achieved approximately 0.9992 ROC-AUC on HC3 validation data, despite differences in activation functions, architecture, and batch size.

The 32 → 16 ReLU configuration remains the preferred baseline due to its combination of performance, simplicity, computational efficiency, and stability across training seeds.

However, strong HC3 validation performance does not guarantee generalization to newer AI-generated text.

### Future Experiments

- Evaluate shortlisted models on an independent modern text dataset.
- Compare ROC-AUC, macro-F1, confusion matrices, and class-specific recall.
- Examine false positives, especially human-written text incorrectly classified as AI.
- Investigate threshold selection using validation data.
- Compare the shortlisted TF-IDF models against embedding-based FFNN models.
- Preserve model configurations and fitted preprocessing artifacts for reproducibility.

**Current Status:** TF-IDF hyperparameter tuning paused. Moving to embedding-based neural network experiments. Final model selection pending external evaluation.

