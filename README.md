# State Farm Distracted Driver Detection
## Goals
  
## Requirements

## Experimental Setup
![Augmentation](./imgs/state-farm-detection1.jpg)

## Experiment Result
* The evaluation criterion for this Kaggle competition is multi-class logarithmic loss.
* As the validation set, 25% of the images were randomly assigned. However, in the case of the 5-fold CV ensemble, the dataset was divided into 5 equal parts.

|                                                                                                    Treatment                                                                                                    | Public Score | Private Score |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------:|:-------------:|
|                                                                                    [RAdam](https://arxiv.org/abs/1908.03265)                                                                                    |    1.0260    |    0.6792     |
|                                                                                    [AdamW](https://arxiv.org/abs/1711.05101)                                                                                    |    0.9117    |    0.7140     |
|                                                                                 RAdam + [SWA](https://arxiv.org/abs/1803.05407)                                                                                 |    1.1547    |    0.7527     |
|                                                                                RAdam + [Mixup](https://arxiv.org/abs/1710.09412)                                                                                |    0.8331    |    0.6423     |
|                                                                        RAdam + [Label Smoothing](https://arxiv.org/abs/1906.02629) (0.1)                                                                        |    0.9047    |    0.7891     |
|                                                                                               RAdam + Mixup + TTA                                                                                               |    1.0701    |    1.4429     |
|                                                                                    RAdam + Mixup + TTA + 5-fold CV Ensemble                                                                                     |    1.0701    |    1.4429     |
| [Pseudo Labeling](https://www.researchgate.net/publication/280581078_Pseudo-Label_The_Simple_and_Efficient_Semi-Supervised_Learning_Method_for_Deep_Neural_Networks) + RAdam + Mixup + TTA + 5-fold CV Ensemble |    1.0701    |    1.4429     |

### Model Explainability with [Grad-CAM](https://arxiv.org/abs/1610.02391)
* 20 samples were randomly selected from the test set and visualized using the Grad-CAM technique. Labels shown are predicted.
![Grad-CAM](./imgs/state-farm-detection2.jpg)

## Model Serving
### TF Serving
* You need to download and run the docker image via `scripts/run.sh` file. Then, you can test model inference through a locally hosted TF Serving.
### SageMaker 