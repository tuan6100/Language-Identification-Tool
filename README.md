# Language Identification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A language identification model using a Naive Bayes classifier without dependency on the <a href="https://scikit-learn.org" target="_blank">scikit-learn</a> library.

## ⚡ CUDA Acceleration (Optional)

If you'd like to leverage **CUDA** to accelerate computations:

1. Download and install the <a href="https://developer.nvidia.com/cuda-downloads" target="_blank"> NVIDIA CUDA Toolkit </a> .
2. Ensure that the `$CUDA_PATH` environment variable is properly set.
3. Then, run the following commands to complete the setup:

```bash
   conda install cuda -c nvidia
   pip install nvidia-cuda-runtime-cu12
   pip install -U pip setuptools wheel
```

You may need to restart your computer to enable all CUDA functionality.

For more information, please visit <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html" target="_blank">this page</a>

## References
- <a href="https://www.kaggle.com/code/mehmetlaudatekman/naive-bayes-based-language-identification-system">kaggle.com/code/mehmetlaudatekman/naive-bayes-based-language-identification-system</a>
- <a href="https://huggingface.co/papluca/xlm-roberta-base-language-detection">huggingface.co/papluca/xlm-roberta-base-language-detection</a>
