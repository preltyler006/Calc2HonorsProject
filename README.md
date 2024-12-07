
# Graphing Surface Areas of Two-Dimensional Curves

This is an Honors Project for Calculus II, which focuses on the topic of arc length and surface area of an equation rotated around a specified axis.

## Background

This program can be thought of similar to the popular graphing calculator [desmos](https://www.desmos.com/calculator), but it has more specified usage in terms of arc length and surface area. When you have a continuously differentiable curve over [a, b], and you rotate it around the x or y axis, you can determine the surface area that this graph creates given a few integrals for calculation. The full paper that was written concurrently with this program can be found [here](https://docs.google.com/document/d/1KXMEo30T6SQVw0jBhcdIRZlru9hD35Inw7ce38pOdXM/edit?usp=sharing).

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following libraries
```bash
pip install numpy
pip install matplotlib
pip install sympy
python -m pip install scipy
```

## Usage
In the ```main.py``` file, there are a few attributes that you can adjust to fit your needs. The first and most important one is the equation that you want to be calculating for and rotating. Next, you can adjust the bounds that you want to be calculating for, and finally, the axis of rotation. On lines ```15-17```, these attributes are found respectfully, and detailed instructions on how to input different equations are shown in the multi-line comments below.

