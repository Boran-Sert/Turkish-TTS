# Open Source Licenses and Notices

This project incorporates the following open-source software components. This document serves as the required notice for compliance with their respective licenses, particularly for corporate usage.

## 1. VoxCPM
- **License**: Apache License 2.0
- **Source**: [openbmb/VoxCPM]
- **Modifications**: Pursuant to Section 4(b) of the Apache License 2.0, please note that the original source code has been modified for this project to support advanced streaming and backend integration. The following files were modified:
  - `src/turkish_tts/_vendor/voxcpm/__init__.py`
  - `src/turkish_tts/_vendor/voxcpm/core.py`
  - `src/turkish_tts/_vendor/voxcpm/model/voxcpm.py`
  - `src/turkish_tts/_vendor/voxcpm/model/voxcpm2.py`
  - `src/turkish_tts/_vendor/voxcpm/training/runner.py`

  The vendored tree was also reduced to inference and fine-tuning only. The
  Gradio demos (`app.py`, `app_old.py`, `lora_ft_webui.py`), the `voxcpm` CLI,
  the timestamp tooling and the upstream release workflow were removed, and
  `scripts/train_voxcpm_finetune.py` was moved to
  `src/turkish_tts/_vendor/voxcpm/training/runner.py` and converted into a
  package module.

  Additionally, the following new files have been introduced to the `voxcpm` package:
  - `src/turkish_tts/_vendor/voxcpm/streaming.py`
  - `src/turkish_tts/_vendor/voxcpm/streaming_async.py`
  - `src/turkish_tts/_vendor/voxcpm/text_buffer.py`
  - `src/turkish_tts/_vendor/voxcpm/config_loader.py`
  
All modifications are subject to the terms of the Apache License 2.0. A complete copy of the Apache License 2.0 can be found in the `src/turkish_tts/_vendor/voxcpm/LICENSE` file.

## 2. Trendyol-TTS
- **License**: MIT License
- **Source**: [Trendyol/Trendyol-TTS]
- **Description**: A Turkish text-to-speech research model built on top of VoxCPM2. The model weights and inference scripts are used under the MIT License.


---
*This notice ensures that corporate usage of the aforementioned tools complies with their respective open-source licensing requirements.*
