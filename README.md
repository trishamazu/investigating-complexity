# Investigating the Emergence of Complexity from the Dimensional Structure of Mental Representations

### Objective
This project assesses the ability of CLIP-Human Behavior Assessment (CLIP-HBA), which was created by fine-tuning [CLIP-IQA](https://github.com/IceClear/CLIP-IQA.git) (Wang et al., 2022^[1]) on behavioral embeddings developed from human similarity judgements, to predict human perceptions of visual complexity.

### Prerequisites
Running this code requires:
- Python 3.8.18 (other versions may work, but this is the tested version)
```
git clone https://github.com/trishamazu/complexity-experiment-final.git
```
```
conda env create -f environment.yml
conda activate complexity_experiment
```

### Usage
To obtain CLIP-HBA embeddings for your own images, you will have to clone a separate repo. See the instructions [here](https://github.com/stephenczhao/CLIP-HBA-Official.git).
```
git clone https://github.com/stephenczhao/CLIP-HBA-Official.git
```
