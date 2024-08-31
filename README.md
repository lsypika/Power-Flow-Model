# Power Flow Calculation in Python
This repository implements power flow calculation methods using Python within Jupyter Notebooks. The code is based on theoretical approaches presented in [text slides](TextSlides-Load%20generation%20balance.pptx) and provides tools for calculating the power flow in electrical grids.

## Overview
The repository contains a step-by-step implementation of power flow calculating method, beginning with a simple two-node example and extending to a more generalized model that can be applied to larger and more complex networks. The generalized model is designed to calculate the power balance and line flows in an electrical grid based on given parameters for buses (nodes) and lines in a specific excel format.

## Getting Started
### 1. Understand the Theory
Before diving into the code, it is recommended to first review the accompanying [text slides](TextSlides-Load%20generation%20balance.pptx). These slides provide a detailed explanation of the power flow calculation methods and the underlying theoretical concepts.
### 2. Two-Node Example
The ["Two_node_example.ipynb"](Two-Node Example.ipynb) notebook serves as a practical introduction to the power flow calculation method. This notebook walks you through the implementation of the method in a simple two-node network, helping you practice and verify the concepts learned.
### 3. Generalized Power Flow Basic Model
For more complex networks, the generalized model - [Basic Model](Generalized Basic Model.ipynb) implemented in this repository can be used to calculate the power flow in grids with multiple buses and lines. The model is flexible and can handle networks with various configurations, provided that the parameters for buses and lines are given in the specified format.
## Features
[Two-Node Example](Two-Node Example.ipynb): A simple and clear demonstration of power flow calculation in a basic network.

[Generalized Basic Model](Generalized Basic Model.ipynb): A scalable and flexible model for larger electrical grids.

[Customizable](Data): The code can be adapted to different network configurations by adjusting the input parameters.
## Applications
This repository try to a valuable tool for researchers and engineers working in the field of power systems. It produces a database for power flow studies and can be used to analyze the performance and stability of electrical grids under various operating conditions.
