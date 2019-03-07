# Direct Feedback Alignment with Sparse Connections for Local Learning

This is the github page for the results and code to reproduce the results for "Direct Feedback Alignment with Sparse Connections for Local Learning" (https://arxiv.org/abs/1903.02083). The main concept for this work is using Feedback Alignment (https://www.nature.com/articles/ncomms13276) and a extremely sparse matrix to reduce datamovement by orders of magnitude while enabling bio-plausible learning. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to install the software and how to install them

```
tensorflow-gpu or tensorflow
ImageNet (http://www.image-net.org/)
```

### Installing

```
git clone https://github.com/bcrafton/ssdfa
```

### Running

```
cd ssdfa
python mnist_fc.py --dfa 1 --sparse 1
```

To run imagenet, the training and test set can be acquired from: http://www.image-net.org/
The links must be changed inside of imagenet.py and imagenet_vgg.py

## Hardware

This code was run on 8 Nvidia Titan Xp 12GB GPUs. Only 1 was used per simulation (no multi-gpu simulations).

## Built With

* [tensorflow](https://github.com/tensorflow/tensorflow) - The GPU framework used

## Authors

* **Brian Crafton** 
* **Abhinav Parihar** 
* **Evan Gebhardt** 
* **Arijit Raychowdhury** 

## Affiliation 

```
Georgia Institute of Technology, ICSRL (http://icsrl.ece.gatech.edu/)
```

![Alt text](./icsrl/icsrl.png?raw=true "Title")

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details





