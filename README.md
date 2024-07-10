# Algoz

Algoz is a platform of tools to develop algorithms for gases sensor.

### Installation

Algoz requires Python 3.8 or higher.

To install Algoz, run in the Algoz directory:

    python setup.py install

### Getting Started

The tools allow to train a model, resume training and perform inferences.

All of these tools use configuration files in JSON format. 
The entry point for all tools is the inputs and outputs 
configuration file which must be declared in the code of the used tool.

To train a model:

    python train.py

To resume the training of a saved model:

    python resume.py

To perform inferences of a saved model on a specific dataset:

    python inference.py

To perform inferences of a saved model on a several datasets:

    python multi-inference.py


### License

[Apache License 2.0](LICENSE)
