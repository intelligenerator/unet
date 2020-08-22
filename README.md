# unet

Basic U-Net implementation in pytorch.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][1]

## Table of Contents

-   [Getting Started](#getting-started)
    -   [Prerequisites](#prerequisites)
    -   [Initial setup](#initial-setup)
    -   [Running the example notebook](#running-the-example-notebook)
-   [Usage](#usage)
-   [Docs](#docs)
    -   [Building the docs](#building-the-docs)
-   [Contributing](#contributing)
-   [Versioning](#versioning)
-   [Authors](#authors)
-   [License](#license)
-   [See also](#see-also)
-   [Acknowledgments](#acknowledgments)

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes.

If you would just like to play around with the model without downloading
anything to your machine, you can open this notebook in Google Colab
(Note that a Google account is required to run the notebook):
[Open in Google Colab][1]

### Prerequisites

You will need python3 and pip3 installed on your machine. You can install it
from the official website https://www.python.org/.

To install pytorch with CUDA support, conda is recommended. An installation
guide is available in the conda docs:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/

To be able to view und run the example notebooks on your machine, jupyter is
required. An installation guide can be found on their website:
https://jupyter.org/install

### Initial setup

A step by step series of examples that tell you how to get the project up and
running.

Clone the git repository

```bash
git clone https://github.com/intelligenerator/unet.git
cd unet
```

Then create your conda virtual environment

```bash
conda create --name torch
conda activate torch
```

Next, installed the required packages. This may vary based on your system
hardware and requirements. Read more about pytorch installation:
https://pytorch.org/get-started/locally/

```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

To exit the virtual environment run

```bash
conda deactivate
```

Happy coding!

### Running the example notebook

To run the provided example notebook on your machine, make sure you have jupyter
installed.

First, create a jupyter kernel for your conda environment:

```bash
pip install --user ipykernel
python -m ipykernel install --user --name=torch
```

Then, open jupyter lab:

```bash
jupyter lab
```

> **Important:**
> Make sure you use the kernel you created above. After opening the notebook,
> navigate to `Kernel` > `Change Kernel...` in the UI and select `torch` from
> the dropdown.
> See this blog post for more info:
> https://janakiev.com/blog/jupyter-virtual-envs/

## Usage

Assuming, you have cloned this repo into the `unet/` subfolder, you can import
it from your project root:

```python
import torch
from unet import UNet

net = UNet(in_channels=3, out_channels=1)
# your code ...
```

## Docs

Check out the [unet docs](intelligenerator.github.io/unet/) for usage
information.

For a more hands-on approach, feel free to experiment with the
[unet example on Google Colab][1]

### Building the docs

To build the docs yourself, create a python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Next, install sphinx, numpydoc and the sphinx-rtd-theme:

```bash
pip install -r requirements.txt
```

Then, build the docs:

```bash
cd docs/
make html
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details on our code of conduct, and
the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available,
see the [tags on this repository](https://github.com/intelligenerator/unet/tags).

## Authors

Ulysse McConnell - [umcconnell](https://github.com/umcconnell/)

See also the list of
[contributors](https://github.com/intelligenerator/unet/contributors)
who participated in this project.

## License

This project is licensed under the MIT License - see the
[LICENSE.md](LICENSE.md) file for details.

## See also

## Acknowledgments

-

-   [numpy gitignore](https://github.com/numpy/numpy/blob/master/.gitignore) -
    Gitignore inspiration
-   [github python gitignore template](https://github.com/github/gitignore/blob/master/Python.gitignore) - The gitignore template
-   [python3 tutorial](https://docs.python.org/3/tutorial/venv.html) - Guide and
    explanations
-   [Contributor Covenant](https://www.contributor-covenant.org/) - Code of Conduct

[1]: http://colab.research.google.com/github/intelligenerator/unet/blob/master
