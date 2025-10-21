# Thesis Project 1: Real $$\phi^4$$

*Impurity Tensor Renormalization for the real $$\phi^4$$-Model*

## 1 Background, Motivation and Goal

This project explores the use of tensor network renormalization (TNR) methods to study the two-dimensional $$\phi^4$$ [model](https://en.wikipedia.org/wiki/Quartic_interaction).
The work focuses on implementing an impurity-based variant of the Tensor Renormalization Group (TRG) to evaluate expectation values such as $$<\phi>$$ and $$<\phi^2>$$.

The purpose is to get familiar with the workflow and play with this simple toy model. I tried to produce expected results such as a phase diagram, cft data, ... and I tried to reproduce results of a paper of [Kadoh et.al.](https://arxiv.org/abs/1811.12376)

## 2 Project Structure
```
├── Impurity algorithm/
│   └── ImpTRG.jl               # First clean implementation ImpurityTRG, which is now part of TNRKit
│ 
├── Results/  
│   └── PhaseDiagram.jl         # Calculate and plot phase diagram
│   └── CFT_data.jl             # Calculate and plot cft data
│   └── SV_fmatrix.jl           # Calculate and plot figure 1 of the paper mention above
│   └── Susceptibility.jl       # Calculate and plot figures 2, 3 and 4 of the paper mention above
│                              
└── Tensormaker/                 
    └── TensorFactory.jl        # Makes the tensors for this model
```
