# Study notebooks

| Question | Notebook |
|---|---|
| How does a ten-second recording become a 128 × 313 input? | [Audio to log-mel](03_audio_to_logmel.ipynb) |
| How do the five CNNs differ in depth and width? | [Model architectures](02_model_architectures.ipynb) |
| Do high healthy-ID scores translate into reliable fault alarms? | [ID accuracy and fault detection](01_reproduce_results.ipynb) |
| Which combinations catch more faults, and on which machines? | [Ensemble comparison and findings](04_ensemble_combinations.ipynb) |
| Will the published ensemble flag my recording? | [Run the trained models on a WAV](05_score_recording.ipynb) |

The two results notebooks run on a CPU with the saved evidence. The architecture and preprocessing examples use the same implementations as training. For a full run, the [runtime notebook](00_runtime.ipynb) builds local Python modules from the implementation cells under `code/`. [Installation and training](../REPRODUCE.md)
