# Training-Time Data Poisoning Experiments

Controlled simulation experiments evaluating training-time data poisoning attacks on image classification models using PyTorch.

## Implemented Experiments

- CIFAR-10 Backdoor Attack – Trigger-based poisoning with Attack Success Rate (ASR) evaluation.
- CIFAR-10 Label-Flip Poisoning – Random label corruption of a percentage of training samples.
- CIFAR-10 Federated Aggregation Poisoning – FedAvg-style simulation with one malicious client performing trigger-based poisoning.
- EuroSAT Backdoor Attack – Trigger-based poisoning applied to satellite imagery.

## Model

Lightweight CNN:
- 2 convolutional layers
- ReLU activations
- Max pooling
- 2 fully connected layers

## Metrics

- Clean Accuracy
- Macro F1 Score
- Attack Success Rate (ASR) for trigger-based attacks

## Requirements

```bash
pip install torch torchvision numpy scikit-learn
```

Datasets download automatically on first run.

## Run Experiments

```bash
python cifar10_backdoor_experiment.py
python cifar10_label_flip_experiment.py
python cifar10_fedavg_poisoning_experiment.py
python eurosat_backdoor_experiment.py
```

## Reproducibility

- Fixed random seed (seed = 42)
- Deterministic PyTorch configuration
- CPU execution

Academic research purposes only.