# FlashPPI: Linear-time prediction of proteome-scale microbial protein interactions

<p align="center">
  <a href="https://www.pnas.org/doi/abs/10.1073/pnas.2610619123"><img src="https://img.shields.io/badge/Paper-PNAS-navy" style="max-width: 100%;"></a>
  <a href="https://huggingface.co/tattabio/flashppi"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Model" style="max-width: 100%;"></a>
</p>

<p align="center">
  <img src="docs/images/figure1.png" alt="FlashPPI model overview" width="600"/>
</p>

## Model Description
FlashPPI is a contrastively trained model for protein-protein interaction (PPI) prediction, grounded in residue-level interactions, that enables full-proteome interaction prediction in minutes.

By reframing PPI prediction as a dense retrieval task, FlashPPI circumvents the $\mathcal{O}(N^2)$ computational bottleneck of traditional all-vs-all structural screening.

- **Scalable:** Reduces proteome-wide screening from days/months to minutes.
- **Interpretable:** Predicts fine-grained, residue-level 2D contact maps for retrieved interaction candidates.
- **Genomic Priors:** Leverages [gLM2](https://huggingface.co/tattabio/gLM2_650M) initialization to capture cross-protein, multi-gene co-evolutionary signals.


## Web Server
FlashPPI is integrated into [seqhub.org](https://seqhub.org). You can upload a FASTA and interactively explore whole-proteome networks and contact maps. Explore an example network [here](https://seqhub.org/tattabio/mycobacterium_tb?ppi=true).

## Installation

```bash
pip install -r requirements.txt
```

Optionally, install [Flash Attention](https://github.com/Dao-AILab/flash-attention) for faster inference on GPU:

```bash
pip install flash-attn --no-build-isolation
```

## Usage

### Fast Proteome-wide PPI Screening (All-vs-All)
Run the prediction script by passing your proteome FASTA file. It will output a predictions file with predicted pairs of interacting proteins and confidence scores.
Note: Requires a machine with at least 1 GPU.

```bash
python predict_proteome.py --fasta my_proteome.fasta --output predictions.csv
```

Embeddings are automatically computed in batches, stored in an optimized HDF5 file (`my_proteome.flashppi.h5` in float16), and loaded on-demand. This guarantees constant, low VRAM usage regardless of proteome scale, and makes future runs instantaneous:

```bash
# Re-run with a different threshold without recomputing embeddings:
python predict_proteome.py --embeddings my_proteome.flashppi.h5 --threshold 0.7 --output high_conf.csv
```

### Cross-Proteome PPI Screening (Host–Viral)

Predict interactions between two proteomes, for example a viral genome and its host genome:

```bash
python predict_cross_proteome.py \
    --host_fasta host.fasta \
    --viral_fasta virus.fasta \
    --output predictions.csv
```

Host embeddings can be computed once and reused across dozens of viral screens:

```bash
python predict_cross_proteome.py \
    --host_embeddings host.flashppi.h5 \
    --viral_fasta virus.fasta \
    --output predictions.csv
```

### Standalone Embedding Computation

You can also precompute and cache embeddings separately:

```bash
python embed_proteome.py --fasta proteome.fasta --output proteome.flashppi.h5
```


### Visualizing contact predictions

```python
import torch
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

seq1 = "MKTAYIAKQRQISFVKSHFSRQL"
seq2 = "MSTAGKVIKCKAAVLW"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("tattabio/flashppi", trust_remote_code=True)
model = AutoModel.from_pretrained("tattabio/flashppi", trust_remote_code=True).to(device).eval()

inputs1 = tokenizer(seq1, return_tensors="pt").to(device)
inputs2 = tokenizer(seq2, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(
        input_ids1=inputs1["input_ids"],
        attention_mask1=inputs1["attention_mask"],
        input_ids2=inputs2["input_ids"],
        attention_mask2=inputs2["attention_mask"],
        return_dict=True
    )

# Extract map and trim padding
contact_map = outputs.contact_map[0].cpu().numpy()
len1, len2 = inputs1["attention_mask"].sum().item(), inputs2["attention_mask"].sum().item()
contact_map = contact_map[:len1, :len2]

plt.imshow(contact_map, cmap="Blues", vmin=0, vmax=1)
plt.savefig("contact_map.png")
```

## Training

```bash
# Multi-GPU
accelerate launch --config_file configs_accelerate/multi_gpu.yaml -m flashppi.train configs_train/flashppi.yaml

# Single CPU (testing)
accelerate launch --config_file configs_accelerate/cpu.yaml -m flashppi.train configs_train/flashppi_ESM_small.yaml
```

## License
This repository is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. Free for academic and research use.
## Citing 
If you use FlashPPI or our datasets in your research, please cite:

```
@article{
	  doi:10.1073/pnas.2610619123,
	  author = {Andre Cornman  and Matt Tranzillo  and Nicolo G. Zulaybar  and Imane Bouzit  and Yunha Hwang },
	  title = {Linear-time prediction of proteome-scale microbial protein interactions},
	  journal = {Proceedings of the National Academy of Sciences},
	  volume = {123},
	  number = {25},
	  pages = {e2610619123},
	  year = {2026},
	  doi = {10.1073/pnas.2610619123},
	  URL = {https://www.pnas.org/doi/abs/10.1073/pnas.2610619123},
	  eprint = {https://www.pnas.org/doi/pdf/10.1073/pnas.2610619123},
}
```
