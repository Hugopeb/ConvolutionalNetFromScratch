# ConvolutionalNetFromScratch

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This project is a learning-focused implementation of a convolutional neural network (CNN) built from scratch in Python, with the forward and backward passes developed step by step. It is purely educational and does not aim to outperform existing models. The main goal is to understand how convolutional networks work internally and to implement these processes manually in Python.

Aditionally, this project is in an
early stage and I plan to extend it in the future.


---

## Features

- Forward and backward propagation implemented manually
- Training and validation loops implemented manually
- Jupyter Notebook demonstrations included
- Visualization of convolutional filters

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Hugopeb/ConvolutionalNetFromScratch.git
cd ConvolutionalNetFromScratch
```

2. **Create a virtual environment**

```bash
python -m venv ConvolutionalNetFromScratch_VENV
source ConvolutionalNetFromScratch/bin/activate
```

3. **Install required packages**

```bash
pip install -r requirements.txt
```
---

## Usage / Examples

After installing dependencies and setting up the virtual
environment you have two different options.

### Using the Jupyter Notebook
Open the notebook to interactively run experiments, explore the 
code step by step, and read the accompanying comments, which 
explain the reasoning and implementation details behind each 
part of the model.

```bash
jupyter notebook Main_notebook.ipynb
```

Visualizations of the convolutional filters are also included in
the notebook.

---

### Using the Python script

If someone wants to **run training automatically**:

```bash
python model.py
```

This way you can just modify the code without having to go
through the whole notebook and its explanations.

---

## Project Structure

Here’s an overview of the files and folders in this project:

- **Main_notebook.ipynb**: Ideal for exploring the network interactively and understanding each step of the implementation.
- **model.py**: Contains the classes and functions defining your neural network layers and forward/backward passes.
- **images/**: Visual outputs to showcase results, mainly filter activations.
- **requirements.txt**: Install all Python packages needed with `pip install -r requirements.txt`
- **README.md**: This file, explaining the project, how to use it, and providing examples.
- **LICENSE**: File containing the MIT License.

---

## License

This project is licensed under the **MIT License**.  

You are free to use, copy, modify, merge, publish, distribute, s>

- The above copyright notice and this permission notice shall be>

**Disclaimer:** The software is provided "as is", without warran>

For the full license text, see the [LICENSE](LICENSE) file.
