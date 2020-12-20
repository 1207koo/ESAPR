# ESAPR: Sequential recommendation system based on VAE

## Overview

Sequential recommender system based on VAE

This project is based on [Multi-VAE_PR by Netflix](https://arxiv.org/pdf/1802.05814.pdf), but has been developed to be used as a sequential recommendation system.

## Run

### On Google Colab

You can just run ESAPR.ipynb on Google Colab, which runs on **MovieLens 1M** and **MovieLens 20M** dataset.

### Local

Install the required packages into your python environment:

```
pip install -r requirements.txt
```


In order to train my model, run main.py as follows:

```
python main.py --templates train_vae
```

This will apply all the options specified in `templates/train_vae.yaml`, and train **VAE** on **MovieLens 1M** dataset as a result.

You can also apply other templates in the `templates` folder. For example,

```
python main.py --templates train_bert
```

will train **BERT4Rec** model instead.

It is also possible to override some options with command line arguments. For example,

```
python main.py --templates train_vae --dataset_code ml-20m --train_window 128 --saturation_wait_epochs 16 --weight_type constant --weight_constant 1.0 --train_transfer false
```

will use **MovieLens 20M** dataset with original **VAE** model by Netflix(little difference for better result) and train faster.

These extra features are based on [MEANTIME repository](https://github.com/SungMinCho/MEANTIME).


## References

The baseline codes were based on **MEANTIME** repository ([https://github.com/SungMinCho/MEANTIME](https://github.com/SungMinCho/MEANTIME))

And implementations are based on following codes and researches
* Variational Autoencoders for Collaborative Filtering [(code1)](https://github.com/younggyoseo/vae-cf-pytorch) [(code2)](https://github.com/dawenl/vae_cf) [(paper)](https://arxiv.org/pdf/1802.05814.pdf)

