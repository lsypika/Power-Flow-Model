# Power Flow Calculation in Python
This repository implements power flow calculation methods using Python. The code is based on theoretical approaches presented in [text slides](Materials\TextSlides-Load%20generation%20balance.pptx) and provides a generalized model for calculating the power flow in electrical grids.

## Overview
The repository contains a step-by-step implementation of power flow calculating method, beginning with a simple two-node example and extending to a more generalized model that can be applied to larger and more complex networks. The generalized model is designed to calculate the power balance and line flows in an electrical grid based on given parameters for buses (nodes) and lines in a specific excel format.

## Getting Started
### 1. Understand the Theory
Before diving into the code, it is recommended to first review the accompanying [text slides](Materials\TextSlides-Load%20generation%20balance.pptx). These slides provide a detailed explanation of the power flow calculation methods and the underlying theoretical concepts.
### 2. Two-Node Example
The [Two-Node Example](Two-Node%20Example.ipynb) notebook serves as a practical introduction to the power flow calculation method. This notebook walks you through the implementation of the method in a simple two-node network, helping you practice and verify the concepts learned in text slides.
### 3. Generalized Power Flow Model
For more complex networks, the Generalized Model implemented in this repository can be used to calculate the power flow in grids with multiple buses and lines. The model is flexible and can handle networks with various configurations, provided that the parameters for buses and lines are given in the specified format. 

The model is verified: downloaded the 9-node power network data from the [PyPSA example](Original_data\Data_lpf_result): https://github.com/PyPSA/PyPSA/tree/master/examples/ac-dc-meshed/ac-dc-data/results-lpf, selected the necessary parts as the [model input](Model_input), and then used this model to simulate and calculate to obtain the [model output data](Model_output). Moreover, after comparing with the original data, our results are very close to it, so it can be judged that the function of this model is correct.

Two versions are provided: 
- the [explanation version](Generalized%20Model%20with%20Explanation.ipynb), which aims to help understand the model; 
- the [pure-code version](Generalized%20Model%20Example.py), which is easy to use.

## Features
[Two-Node Example](Two-Node%20Example.ipynb): A simple and clear demonstration of power flow calculation in a basic network.

[Generalized Model with Explanation](Generalized%20Model%20with%20Explanation.ipynb): A scalable and flexible model for larger electrical grids. Explanation is provided.

[Generalized Model](Generalized%20Model%20Example.py): After modifying the input data, directly run the simulation to use.

[Customizable](Model_input): The model can be adapted to different network configurations by adjusting the input parameters here.

## Applications
This repository try to be a practical tool for students and researchersworking in the field of power systems. It produces examples with explanation for power flow studies and can be used to produce power flow data for further applications, such as analyzing the performance and stability of electrical grids under various operating conditions.
