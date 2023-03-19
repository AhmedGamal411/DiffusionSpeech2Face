# DiffusionSpeech2Facereproduce

## Steps to reproduce


Ubuntu 
```
https://ubuntu.com/download/desktop
```

Anaconda Environment
```
https://www.anaconda.com/products/distribution
```

Git (obviously)
```
sudo apt-get install git-all
```

Visual Studio Code (Can use any development enviroment or none if you want)
```
On Ubuntu Software --> Visual Studio Code --> Install
```

In a folder of your choosing run this in a terminal
```
git clone --recurse-submodules https://github.com/AhmedGamal411/DiffusionSpeech2Face
```
This clones this repo with all its submodules

Open that folder with Visual Studio Code

View the readme of every submodule to download and configure its depedencies.

## Submodules

### Voxceleb Trainer

```
https://github.com/AhmedGamal411/voxceleb_trainer
```

To download VoxCeleb2 Dataset

Progress: Started - Close to finishing

### AVSpeechDownloader (Not included in project)

```
https://github.com/naba89/AVSpeechDownloader
```

Downloads AVSpeech dataset.

### DiffusionSpeech2Face_Preprocessing

```
https://github.com/AhmedGamal411/DiffusionSpeech2Face_Preprocessing
```

# DiffusionSpeech2Face_Preprocessing

## Requirments
```
conda create -n ds2f_pre python=3.8
conda activate ds2f_pre
conda install --yes --file requirements.txt
```

## Steps to run

Open `configuration.txt`. change the path of the dataset (datasetPathVideo). it doesn't matter what structure it is, as long as it's videos only. Then change datasetPathAudio, datasetPathFrames and datasetPathFaces to where you want the data to be extracted.

Make sure you run `conda activate ds2f_pre`

Run:
[1] python dbCreateAndPopulate.py
[2] python extractAudio.py
[3] python extractFaces.py

Will make a script to run all those later


References
=======



## Mentions of Projects used

DeepFace
```
https://github.com/serengil/deepface
```

```
@inproceedings{serengil2020lightface,
  title        = {LightFace: A Hybrid Deep Face Recognition Framework},
  author       = {Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle    = {2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages        = {23-27},
  year         = {2020},
  doi          = {10.1109/ASYU50717.2020.9259802},
  url          = {https://doi.org/10.1109/ASYU50717.2020.9259802},
  organization = {IEEE}
}
```

```
@inproceedings{serengil2021lightface,
  title        = {HyperExtended LightFace: A Facial Attribute Analysis Framework},
  author       = {Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle    = {2021 International Conference on Engineering and Emerging Technologies (ICEET)},
  pages        = {1-4},
  year         = {2021},
  doi          = {10.1109/ICEET53442.2021.9659697},
  url          = {https://doi.org/10.1109/ICEET53442.2021.9659697},
  organization = {IEEE}
}
```

```
@misc{serengil2023db,
  author       = {Serengil, Sefik Ilkin and Ozpinar, Alper},
  title        = {An Evaluation of SQL and NoSQL Databases for Facial Recognition Pipelines},
  year         = {2023},
  publisher    = {Cambridge Open Engage},
  doi          = {10.33774/coe-2023-18rcn},
  url          = {https://doi.org/10.33774/coe-2023-18rcn},
  howpublished = {https://www.cambridge.org/engage/coe/article-details/63f3e5541d2d184063d4f569},
  note         = {Preprint}
}
```

To preprocess the dataset.

Progress: Started but unfinished

### DiffusionSpeech2Face_DiffusionModel

```
Unstarted
```

Builds and runs the diffusion model on the preprocessed dataset.

Progress: Unstarted

### DiffusionSpeech2Face_Evaluation

```
Unstarted
```

Evaluates the model.

Progress: Unstarted


## Mentions to Projects and Libraries used

### Voxceleb Trainer

```
https://github.com/clovaai/voxceleb_trainer
```

[1] In defence of metric learning for speaker recognition
```
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Proc. Interspeech},
  year={2020}
}
```

[2] The ins and outs of speaker recognition: lessons from VoxSRC 2020
```
@inproceedings{kwon2021ins,
  title={The ins and outs of speaker recognition: lessons from {VoxSRC} 2020},
  author={Kwon, Yoohwan and Heo, Hee Soo and Lee, Bong-Jin and Chung, Joon Son},
  booktitle={Proc. ICASSP},
  year={2021}
}
```

[3] Pushing the limits of raw waveform speaker recognition
```
@inproceedings{jung2022pushing,
  title={Pushing the limits of raw waveform speaker recognition},
  author={Jung, Jee-weon and Kim, You Jin and Heo, Hee-Soo and Lee, Bong-Jin and Kwon, Youngki and Chung, Joon Son},
  booktitle={Proc. Interspeech},
  year={2022}
}
```