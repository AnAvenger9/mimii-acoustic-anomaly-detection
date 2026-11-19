# Notebook guide

Start with the three walkthroughs. Each includes a short runnable example and its recorded output.

1. [Reproduce the results on CPU](01_reproduce_results.ipynb)
2. [Inspect the five CNN architectures](02_model_architectures.ipynb)
3. [Transform audio into log-mel features](03_audio_to_logmel.ipynb)

The implementation notebooks mirror the Python modules, scripts and tests included in this repository. Sections follow the source definitions, so individual layers, split rules, metrics and training loops can be inspected on GitHub. Training commands and input requirements are in the [reproduction guide](../REPRODUCE.md).

## Preprocessing

- [download_mimii](code/preprocessing/download_mimii.ipynb) · [Python](../scripts/download_mimii.py)
- [verify_full_preprocessing](code/preprocessing/verify_full_preprocessing.ipynb) · [Python](../scripts/verify_full_preprocessing.py)
- [audio](code/preprocessing/audio.ipynb) · [Python](../src/mimii/audio.py)
- [config](code/preprocessing/config.ipynb) · [Python](../src/mimii/config.py)
- [dataset](code/preprocessing/dataset.ipynb) · [Python](../src/mimii/dataset.py)
- [features](code/preprocessing/features.ipynb) · [Python](../src/mimii/features.py)
- [prepare_full](code/preprocessing/prepare_full.ipynb) · [Python](../src/mimii/prepare_full.py)
- [prepare_multisnr](code/preprocessing/prepare_multisnr.ipynb) · [Python](../src/mimii/prepare_multisnr.py)
- [preprocess](code/preprocessing/preprocess.ipynb) · [Python](../src/mimii/preprocess.py)
- [training_data](code/preprocessing/training_data.ipynb) · [Python](../src/mimii/training_data.py)
- [visualization](code/preprocessing/visualization.ipynb) · [Python](../src/mimii/visualization.py)

## Architectures

- [acoustic_cnn](code/architectures/acoustic_cnn.ipynb) · [Python](../src/mimii/acoustic_cnn.py)
- [large_acoustic_cnn](code/architectures/large_acoustic_cnn.ipynb) · [Python](../src/mimii/large_acoustic_cnn.py)
- [mobilenet](code/architectures/mobilenet.ipynb) · [Python](../src/mimii/mobilenet.py)
- [models](code/architectures/models.ipynb) · [Python](../src/mimii/models.py)
- [transfer_models](code/architectures/transfer_models.ipynb) · [Python](../src/mimii/transfer_models.py)

## Training

- [extend_cnn_study](code/training/extend_cnn_study.ipynb) · [Python](../scripts/extend_cnn_study.py)
- [train_acoustic_cnn](code/training/train_acoustic_cnn.ipynb) · [Python](../scripts/train_acoustic_cnn.py)
- [train_cnn_optimization](code/training/train_cnn_optimization.ipynb) · [Python](../scripts/train_cnn_optimization.py)
- [train_large_acoustic_cnn](code/training/train_large_acoustic_cnn.ipynb) · [Python](../scripts/train_large_acoustic_cnn.py)
- [train_mobilenet](code/training/train_mobilenet.ipynb) · [Python](../scripts/train_mobilenet.py)
- [cnn_optimization](code/training/cnn_optimization.ipynb) · [Python](../src/mimii/cnn_optimization.py)
- [custom_cnn_experiment](code/training/custom_cnn_experiment.ipynb) · [Python](../src/mimii/custom_cnn_experiment.py)
- [finetune](code/training/finetune.ipynb) · [Python](../src/mimii/finetune.py)
- [large_cnn_experiment](code/training/large_cnn_experiment.ipynb) · [Python](../src/mimii/large_cnn_experiment.py)
- [train](code/training/train.ipynb) · [Python](../src/mimii/train.py)

## Evaluation

- [report_acoustic_fault_detection](code/evaluation/report_acoustic_fault_detection.ipynb) · [Python](../scripts/report_acoustic_fault_detection.py)
- [verify_mobilenet_results](code/evaluation/verify_mobilenet_results.ipynb) · [Python](../scripts/verify_mobilenet_results.py)
- [ensemble_xgb](code/evaluation/ensemble_xgb.ipynb) · [Python](../src/mimii/ensemble_xgb.py)
- [evaluation](code/evaluation/evaluation.ipynb) · [Python](../src/mimii/evaluation.py)
- [failure_analysis](code/evaluation/failure_analysis.ipynb) · [Python](../src/mimii/failure_analysis.py)
- [report_cnn_optimization](code/evaluation/report_cnn_optimization.ipynb) · [Python](../src/mimii/report_cnn_optimization.py)
- [report_multisnr](code/evaluation/report_multisnr.ipynb) · [Python](../src/mimii/report_multisnr.py)
- [report_transfer](code/evaluation/report_transfer.ipynb) · [Python](../src/mimii/report_transfer.py)
- [reproduce_results](code/evaluation/reproduce_results.ipynb) · [Python](../src/mimii/reproduce_results.py)

## Support

- [__init__](code/support/__init__.ipynb) · [Python](../src/mimii/__init__.py)
- [ae_experiment](code/support/ae_experiment.ipynb) · [Python](../src/mimii/ae_experiment.py)
- [ae_report](code/support/ae_report.ipynb) · [Python](../src/mimii/ae_report.py)
- [ae_windows](code/support/ae_windows.ipynb) · [Python](../src/mimii/ae_windows.py)

## Tests

- [test_acoustic_cnn](code/tests/test_acoustic_cnn.ipynb) · [Python](../tests/test_acoustic_cnn.py)
- [test_ae_windows](code/tests/test_ae_windows.ipynb) · [Python](../tests/test_ae_windows.py)
- [test_cnn_continuation](code/tests/test_cnn_continuation.ipynb) · [Python](../tests/test_cnn_continuation.py)
- [test_cnn_optimization](code/tests/test_cnn_optimization.ipynb) · [Python](../tests/test_cnn_optimization.py)
- [test_download_mimii](code/tests/test_download_mimii.ipynb) · [Python](../tests/test_download_mimii.py)
- [test_full_preprocessing](code/tests/test_full_preprocessing.ipynb) · [Python](../tests/test_full_preprocessing.py)
- [test_large_acoustic_cnn](code/tests/test_large_acoustic_cnn.ipynb) · [Python](../tests/test_large_acoustic_cnn.py)
- [test_mobilenet](code/tests/test_mobilenet.ipynb) · [Python](../tests/test_mobilenet.py)
- [test_multisnr](code/tests/test_multisnr.ipynb) · [Python](../tests/test_multisnr.py)
- [test_preprocessing](code/tests/test_preprocessing.ipynb) · [Python](../tests/test_preprocessing.py)
- [test_training](code/tests/test_training.ipynb) · [Python](../tests/test_training.py)
