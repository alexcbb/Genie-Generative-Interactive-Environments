# Unofficial Genie : Generative Interactive Environments
[[Project Page]]()

[![License](https://img.shields.io/badge/License-MIT-yellow)]([https://github.com/alexcbb/Genie-Generative-Interactive-Environments/LICENSE](https://github.com/alexcbb/Genie-Generative-Interactive-Environments/blob/main/LICENSE)) 

## 💪 Goal
This repo aims to reproduce and open the results obtained from "Generative Interactive Environments" of Google DeepMind.

I'm currently trying to find people interested in reproducing the papers results at a smaller scale.
Don't hesitate to contact me to work together on it : alexandre.chapin@ec-lyon.fr

## 📰 News
**[2024.03.09]** Beginning of the project

## 🛠️ Installation
Create a Python venv or a Conda environment using Python 3.10. Inside of the environment install `torch` and `torchivision` using the instructions matching your system as listed on the [Pytorch website](https://pytorch.org/).

Then install the remaining dependencies:
```
python -m pip install -r requirements.txt
python -m pip install -e .
```
### 🔎 Unit tests 
Run the tests using :
```
pytest tests/
```

While the project will grow, more test will be added and you'll maybe need to just select a subset of tests related to the changes you made by using the `-k` option of `pytest`. Running tests in parallel (in the example 4 processes) with the `-n` option may help :

```
pytest -k "substring-to-match" -n 4 tests/
```

## ✅ TODO list :
### Setup the codebase
- [x] Create README
- [x] Prepare issues labels
### Latent Action Model (LAM)
- [x] Prepare the original Vision Transformer
- [ ] Combine the ViT with Temporal (For ST-Tranformer)
- [ ] Prepare the VQ-VAE objective
- [ ] Combine model parts to get the final LAM
### Video Tokenizer (VT)
- [ ] Prepare VQ-VAE of video
- [ ] Prepare the ST-ViViT version
- [ ] Combine different layers for the final VT
### Dynamic model (DM)
- [ ] Implement the Decoder-only Mask-GiT
### Training
- [ ] Prepare the training dataset
- [ ] Prepare the training pipeline
- [ ] Prepare the evaluation pipeline
- [ ] Prepare visualisation/probing scripts
### Documentation
- [ ] Document the Vision Transformer code
- [ ] Document the VQ-VAE code
- [ ] Document the ST-Transformer code

### Conduct experiments
- [ ] Train the model on other datasets

## 📂 Repo structure (WIP)
```
├── README.md
├── CONTRIBUTION.md
├── scripts                        -> All scripts.
├── genie
│   ├── dataset
│   ├── models
│   │   ├── ae      
│   │   │   ├── vae
│   │   │   └── vqvae
│   │   ├── stt              
│   │   │   ├── vision_transformer      
│   │   │   ├── mask_git          
│   │   │   └── st_transformer
│   │   ├── video_tokenizer
│   │   ├── dynamic_model
│   │   ├── latent_action_model
│   │   └── genie_model
│   ├── train                      -> Training code
│   └── utils
```

## 🤝Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## 👍 Acknowledgment
This project builds upon the work from Google DeepMind's research team :
```
@misc{bruce2024genie,
      title={Genie: Generative Interactive Environments}, 
      author={Jake Bruce and Michael Dennis and Ashley Edwards and Jack Parker-Holder and Yuge Shi and Edward Hughes and Matthew Lai and Aditi Mavalankar and Richie Steigerwald and Chris Apps and Yusuf Aytar and Sarah Bechtle and Feryal Behbahani and Stephanie Chan and Nicolas Heess and Lucy Gonzalez and Simon Osindero and Sherjil Ozair and Scott Reed and Jingwei Zhang and Konrad Zolna and Jeff Clune and Nando de Freitas and Satinder Singh and Tim Rocktäschel},
      year={2024},
      eprint={2402.15391},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
We also built upon the following repositories for the different parts of our models :
- Spatio-Temporal Transformer : https://github.com/xumingxingsjtu/STTN
- VQ-VAE : https://github.com/rosinality/vq-vae-2-pytorch/tree/master 

## 🔒License
This project is licensed under the MIT license - see the [LICENSE](LICENSE) file for details.

