# Meta Optimal Transport

This repository is by
[Brandon Amos](http://bamos.github.io),
[Samuel Cohen](https://www.samcohen16.com/),
[Giulia Luise](https://giulslu.github.io/), and
[Ievgen Redko](https://ievred.github.io/)
and contains the source code building on
[JAX](https://github.com/google/jax) and
[OTT](https://github.com/ott-jax/ott)
to reproduce the
experiments for our
[Meta Optimal Transport](https://arxiv.org/abs/2206.05262)
paper.

![](https://user-images.githubusercontent.com/707462/173271876-61b18081-ffbb-4603-ab43-06c79e80616b.gif)

-----

Yijiang Pang has posted an unofficial PyTorch re-implementation
in the discrete setting
[here](https://github.com/pangyijiang/Implementation-of-Meta-OT-between-discrete-measures).

# Setup
After cloning this repository and installing PyTorch
on your system, you can install dependencies with:
```bash
pip install -r requirements.txt
```

set up the code with:

```bash
python3 setup.py develop
```

# Basic structure of this repository

+ [./conf](./conf): Hydra configuration
+ [./meta_ot](./meta_ot): Meta OT models and other utility code used by the experiments
+ [./paper](./paper): Source for our paper
+ [./train_discrete.py](./train_discrete.py): Training code for Meta OT in discrete settings
+ [./train_color_meta.py](./train_color_meta.py): Training code for color transfer with Meta OT
+ [./train_color_single.py](./train_color_single.py): Our re-implementation of [W2GN](https://arxiv.org/abs/1909.13082) for color transfer
+ [./eval_discrete.py](./eval_discrete.py): Evaluation and analysis for discrete OT
+ [./eval_color.py](./eval_color.py): Evaluation and analysis for color transfer
+ [./plot_mnist.py](./plot_mnist.py): Further plotting for the MNIST experiment
+ [./plot_world_pair.py](./plot_world_pair.py): Further plotting for the spherical transport experiment
+ [./create_video_mnist.py](./create_video_mnist.py): Create the MNIST transport video
+ [./create_video_world.py](./create_video_world.py): Create the spherical transport video
+ [./create_video_color.py](./create_video_color.py): Create the color transport video

# Reproducing our experimental results
## MNIST
This code will automatically download the MNIST dataset
for training and evaluation.
You can run the training code with:
```bash
./train_discrete.py data=mnist
```
This will create a directory saving out the model and
log informations, which you can evaluate and plot with:
```bash
./eval_discrete.py <exp_dir>
./plot_mnist.py <exp_dir>
```

## Spherical transport
First download the
[2020 Tiff data at 15-minute resolution](https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-adjusted-to-2015-unwpp-country-totals-rev11/data-download#)
and save the file to `data/pop-15min.tif`.
Then you can run the training code with:
```bash
./train_discrete.py data=world
```
This will create a directory saving out the model and
log informations, which you can evaluate and plot with:
```bash
./eval_discrete.py <exp_dir>
./plot_world_pair.py <exp_dir>
```

## Color transfer
First download images from WikiArt into `data/paintings` by running:
```bash
./data/download-wikiart.py
```
Then you can run the training code with:
```bash
./train_color_meta.py
```
This will create a directory saving out the model and
log informations, which you can evaluate and plot with:
```bash
./eval_color.py <exp_dir>
```

# Recreating our videos
Our main video can be re-created by running the following scripts:

```bash
./create_video_mnist.py <mnist_exp_dir>
./create_video_world.py <world_exp_dir>
./create_video_color.py <color_exp_dir>
```

# Citations
If you find this repository helpful for your publications,
please consider citing
[our paper](https://arxiv.org/abs/2206.05262):

```
@misc{amos2022meta,
  title={Meta Optimal Transport},
  author={Brandon Amos and Samuel Cohen and Giulia Luise and Ievgen Redko},
  year={2022},
  eprint={2206.05262},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

# Licensing
The source code in this repository is licensed under the
[CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).
