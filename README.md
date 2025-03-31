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
### Directory Structure
```
├── Embeddings              # Embeddings for all images from both models
│   ├── CLIP-CBA
│   │   ├── IC9600
│   │   ├── Savoias
│   │   └── THINGS
│   └── CLIP-HBA
│       ├── IC9600
│       ├── Savoias
│       └── THINGS
├── Figures
│   └── create_figures.py   # Code to create all paper figures
├── Images
│   ├── Savoias-Dataset     # Savoias images and target complexity scores
│   └── THINGS              # THINGS images 
├── Models                  # General and category-specific models
│   ├── GeneralModels
│   └── IC9600Models
│   └── SavoiasModels
└── output                  # Model performance plots
```

### Image Datasets
* [THINGS](https://things-initiative.org/) (Hebart et al., 2020)^[2])
* [Savoias-Dataset](https://github.com/esaraee/Savoias-Dataset) (Saraee et al., 2018)^[3]) The GroundTruth subfolder contains human complexity ratings, separated by category. They are organized in order from the lowest-numbered image to the highest-numbered image.
* [IC9600](https://github.com/tinglyfeng/IC9600) (Feng et al., 2023)^[4]) Note: These images have not been included because they require explicit permission from the authors.

### References
[^1]: Wang, J., Chan, K. C. K., & Loy, C. C. (2022). Exploring CLIP for assessing the look and feel of images. Proceedings of the AAAI Conference on Artificial Intelligence. arXiv. https://doi.org/10.48550/arXiv.2207.12396

[^2]: Zheng, C. Y., Pereira, F., Baker, C. I., & Hebart, M. N. (2019). Revealing interpretable object representations from human behavior. International Conference on Learning Representations (ICLR) 2019. arXiv. https://doi.org/10.48550/arXiv.1901.02915

[^3]: Copyright [2018] [Saraee et al.] Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

[^4]: T. Feng, Y. Zhai, J. Yang, J. Liang, D. Fan, & J. Zhang. (2023) IC9600: A Benchmark Dataset for Automatic Image Complexity Assessment. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(7), 8577-8593. https://doi.org/10.1109/TPAMI.2022.3232328
