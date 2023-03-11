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
git clone --recurse-submodules https://github.com/AhmedGamal411/DiffusionSpeech2Face_Preprocessing
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