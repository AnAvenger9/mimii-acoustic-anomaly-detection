# Notebook guide

Start with the walkthroughs. Each includes a short runnable example and its recorded output.

1. [Reproduce the results on CPU](01_reproduce_results.ipynb)
2. [Inspect the five CNN architectures](02_model_architectures.ipynb)
3. [Transform audio into log-mel features](03_audio_to_logmel.ipynb)
4. [Recompute the 57 ensembles](04_ensemble_combinations.ipynb)
5. [Explore the ensemble findings](05_ensemble_findings.ipynb)

The implementation notebooks contain the source code for the models, scripts and tests. The [runtime notebook](00_runtime.ipynb) prepares importable copies locally for Jupyter and GPU training. Sections follow the source definitions, so individual layers, split rules, metrics and training loops can be inspected on GitHub. Training commands and input requirements are in the [reproduction guide](../REPRODUCE.md).

## Preprocessing

- [download_mimii](code/preprocessing/download_mimii.ipynb)
- [verify_full_preprocessing](code/preprocessing/verify_full_preprocessing.ipynb)
- [audio](code/preprocessing/audio.ipynb)
- [config](code/preprocessing/config.ipynb)
- [dataset](code/preprocessing/dataset.ipynb)
- [features](code/preprocessing/features.ipynb)
- [prepare_full](code/preprocessing/prepare_full.ipynb)
- [prepare_multisnr](code/preprocessing/prepare_multisnr.ipynb)
- [preprocess](code/preprocessing/preprocess.ipynb)
- [training_data](code/preprocessing/training_data.ipynb)
- [visualization](code/preprocessing/visualization.ipynb)

## Architectures

- [acoustic_cnn](code/architectures/acoustic_cnn.ipynb)
- [large_acoustic_cnn](code/architectures/large_acoustic_cnn.ipynb)
- [mobilenet](code/architectures/mobilenet.ipynb)
- [models](code/architectures/models.ipynb)
- [transfer_models](code/architectures/transfer_models.ipynb)

## Training

- [extend_cnn_study](code/training/extend_cnn_study.ipynb)
- [train_acoustic_cnn](code/training/train_acoustic_cnn.ipynb)
- [train_cnn_optimization](code/training/train_cnn_optimization.ipynb)
- [train_large_acoustic_cnn](code/training/train_large_acoustic_cnn.ipynb)
- [train_mobilenet](code/training/train_mobilenet.ipynb)
- [cnn_optimization](code/training/cnn_optimization.ipynb)
- [custom_cnn_experiment](code/training/custom_cnn_experiment.ipynb)
- [finetune](code/training/finetune.ipynb)
- [large_cnn_experiment](code/training/large_cnn_experiment.ipynb)
- [train](code/training/train.ipynb)

## Evaluation

- [report_acoustic_fault_detection](code/evaluation/report_acoustic_fault_detection.ipynb)
- [verify_mobilenet_results](code/evaluation/verify_mobilenet_results.ipynb)
- [ensemble_xgb](code/evaluation/ensemble_xgb.ipynb)
- [evaluation](code/evaluation/evaluation.ipynb)
- [failure_analysis](code/evaluation/failure_analysis.ipynb)
- [report_cnn_optimization](code/evaluation/report_cnn_optimization.ipynb)
- [report_multisnr](code/evaluation/report_multisnr.ipynb)
- [report_transfer](code/evaluation/report_transfer.ipynb)
- [reproduce_results](code/evaluation/reproduce_results.ipynb)

## Support

- [__init__](code/support/__init__.ipynb)
- [ae_experiment](code/support/ae_experiment.ipynb)
- [ae_report](code/support/ae_report.ipynb)
- [ae_windows](code/support/ae_windows.ipynb)

## Tests

- [test_acoustic_cnn](code/tests/test_acoustic_cnn.ipynb)
- [test_ae_windows](code/tests/test_ae_windows.ipynb)
- [test_cnn_continuation](code/tests/test_cnn_continuation.ipynb)
- [test_cnn_optimization](code/tests/test_cnn_optimization.ipynb)
- [test_download_mimii](code/tests/test_download_mimii.ipynb)
- [test_full_preprocessing](code/tests/test_full_preprocessing.ipynb)
- [test_large_acoustic_cnn](code/tests/test_large_acoustic_cnn.ipynb)
- [test_mobilenet](code/tests/test_mobilenet.ipynb)
- [test_multisnr](code/tests/test_multisnr.ipynb)
- [test_preprocessing](code/tests/test_preprocessing.ipynb)
- [test_training](code/tests/test_training.ipynb)
