# Integrating Biological Knowledge for Robust Microscopy Image Profiling on *De Novo* Cell Lines (ICCV2025)



This repo contains code for our ICCV 2025 paper Integrating Biological Knowledge for Robust Microscopy Image Profiling on *De Novo* Cell Lines, a novel framework for phenotypic screening involving unseen cell lines.

[**📖 arXiv**](https://arxiv.org/pdf/2507.10737) | [**💻 GitHub**](https://github.com/The-Real-JerryChen/BioMicroscopyProfiler) | [**🤗 Model**](https://huggingface.co/Jerrychen229/rxrx1_vit)

## Install
Please install the required packages before running the code:
```python
timm==0.9.16
scanpy==1.10.3
torch==2.1.2 (cuda 12.1)
transformers==4.26.1
wandb
anndata
```

## Data 

- For the RxRx datasets, please download the raw image data and the metadata file from: https://www.rxrx.ai/datasets . The processed metadata for pretraing and evaluation can be found under the data folder.   

- The raw RNA-seq data are from *GSE288929* and *GSM7745109*. We also provide the preprocessed h5ad file in [Hugging Face](https://huggingface.co/Jerrychen229/rxrx1_vit) .

The `src/data/` folder contains the following key files:

- **`rxrxmeta.csv`**: Pre-training data metadata file 
- **`u2os_data.csv`**: Evaluation dataset metadata for U2OS cell line experiments
- **`scvi.pkl`**: Cell line representations obtained using scVI 
- **`u2os.pkl`**: Contains perturbation graph edge weights specifically for U2OS cell line

## Pretraining

1. **Download Data**: First, download the meta_data and image datasets from the RxRx1 dataset.

2. **Configure Paths**: Update the data paths in your configuration:
   - Modify `csv_path` to point to your metadata file location
   - Update `root_dir` to point to your image dataset directory

3. **Run Pre-training**:
   ```bash
   bash src/pretrain/run_pretrain.sh
   ```

4. **Hyperparameter Tuning**: 
   - You can sweep various hyperparameters according to your needs
   - Alternatively, customize using the hyperparameters provided in src/eval/run_eval.sh 


## Evaluation

1. **Configure Evaluation Paths**: Update the following paths in `src/eval/main.py`:
   - `pkl_path`: Path to gene expression feature file
   - Dataset address: Path to your evaluation dataset
   - Pre-trained model weights path: Path to your saved model weights
   - **Alternative**: You can download our pre-trained model weights from [Hugging Face](https://huggingface.co/Jerrychen229/rxrx1_vit)

2. **Run Evaluation**:
   ```bash
   bash src/eval/run_eval.sh
   ```

3. **Graph Regularization**: 
   - In our implementation, we found that adding graph regularization during evaluation is more efficient than during pre-training
   - You can directly adjust the graph loss weight in `run_eval.sh` for optimal performance



## Citation
If you find our paper useful, please cite us with
```
@misc{chen2025integratingbiologicalknowledgerobust,
      title={Integrating Biological Knowledge for Robust Microscopy Image Profiling on De Novo Cell Lines}, 
      author={Jiayuan Chen and Thai-Hoang Pham and Yuanlong Wang and Ping Zhang},
      year={2025},
      eprint={2507.10737},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.10737}, 
}
```
