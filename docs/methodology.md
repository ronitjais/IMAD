# Methodology

## Image baseline

Inspect the dataset before interpreting directory names as labels. Keep generator and source metadata when available. Generate a seeded train/validation/test manifest, group byte-identical files, and review the manifest for source leakage. The starter training script uses a pretrained ResNet-18 and selects its checkpoint by validation loss. It records training/validation loss and accuracy, but does not yet provide the complete evaluation report or calibration. Scores are uncalibrated model outputs.

Do not select the model or threshold using an external evaluation dataset. Preserve one internal test set and evaluate independent sources only after schema and label compatibility checks. Colab GPU experiments should save the resolved config, environment, seed, manifest, checkpoint, and history.
