# Third-Party Notices

## Transformer components and feature conventions

The attention, feed-forward, and GELU components in `ecra/layers.py` were adapted from the SDT/TEN source distribution accompanying:

Hui Ma, Jian Wang, Hongfei Lin, Bo Zhang, Yijia Zhang, and Bo Xu. *A Transformer-Based Model With Self-Distillation for Multimodal Emotion Recognition in Conversations*. IEEE Transactions on Multimedia, 26, 776–788, 2024. DOI: [10.1109/TMM.2023.3271019](https://doi.org/10.1109/TMM.2023.3271019).

Source distribution: <https://github.com/pinuocao163/TEN>.

The multimodal pickle interface follows that distribution's feature conventions. Rights to the adapted components remain with their holders, and applicable upstream terms govern their use.

The SDT/TEN distribution also acknowledges [COSMIC](https://github.com/declare-lab/conv-emotion) and [MMGCN](https://github.com/hujingwen6666/MMGCN) for shared code and datasets. Those projects are not vendored in this archive.

## External data and model assets

IEMOCAP and MELD are distributed by their respective maintainers and remain subject to their access and usage terms. This archive includes supplied transcript/label metadata derived from those datasets under `resources/`. It does not include complete extracted multimodal feature tensors. Inclusion does not alter the underlying dataset terms or establish a new redistribution permission.

Qwen2.5-7B-Instruct is an external pretrained model distributed by the Qwen team. Its model card identifies the Apache-2.0 license: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>. Model weights, tokenizer files, and their accompanying license are obtained separately and are not redistributed here.

## Dependencies

PyTorch, NumPy, SciPy, scikit-learn, Transformers, Accelerate, and Safetensors are installed separately through the dependency manifest. Their licenses are supplied by their respective distributions. Black is an optional formatting tool configured in `pyproject.toml`; it is not required for model execution.

No blanket license for third-party resources is implied by the inclusion of this code submission.
