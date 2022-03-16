# State Farm Distracted Driver Detection
## Goals
  
## Requirements

## Experimental Setup
![Augmentation](./imgs/state-farm-detection1.jpg)

## Experiment Result
|                                 Treatment                                  | Public Score | Private Score |
|:--------------------------------------------------------------------------:|:------------:|:-------------:|
|                                   RAdam                                    |    0.7513    |    1.4200     |
|                                   AdamW                                    |    1.0701    |    1.4429     |
|                                AdamW + SWA                                 |    1.0701    |    1.4429     |
|                            Mixup + AdamW + SWA                             |    1.0701    |    1.4429     |
|                       Label Smoothing + AdamW + SWA                        |    1.0701    |    1.4429     |
|                    Label Smoothing + AdamW + SWA + TTA                     |    1.0701    |    1.4429     |
|          Label Smoothing + AdamW + SWA + TTA + 5-fold CV Ensemble          |    1.0701    |    1.4429     |
| Label Smoothing + AdamW + SWA + TTA + 5-fold CV Ensemble + Pseudo Labeling |    1.0701    |    1.4429     |

### Model Explainability with Grad-CAM
![Grad-CAM](./imgs/state-farm-detection2.jpg)

## Model Serving