# Evolutionary Simulation with Geometric Fisher Model (Streamlit App)

This repository contains an **interactive evolutionary simulator** based on the **Geometric Fisher Model**, implemented as a **Streamlit web app**. 

The simulator models a population of organisms evolving over time to **chase a shifting global optimum**. It visualizes **adaptive dynamics** using a **genetic algorithm** with mutation and selection.

## Features

✅ **Interactive UI** – Select parameters and run simulations with one click.  
✅ **Genetic Algorithm** – Mutation-based adaptation to a shifting environment.  
✅ **Global View Mode** – Ensures the entire evolutionary process stays visible.  
✅ **Diagnostic Plots** – Track fitness distributions, reproduction success, and offspring variance.  
✅ **Progress Visualization** – Real-time progress bar for simulation execution.  
✅ **GIF Outputs** – Stores evolution and diagnostic plots as GIFs for analysis.  
✅ **Simulation History** – View past runs with filtering options.  

---

## **How the Model Works**

The **Geometric Fisher Model** describes adaptation in a **continuous phenotype space** where fitness is determined by **distance from an optimum**.

- **Organisms** are represented as **vectors of real numbers** (2D phenotype).
- **Fitness** follows an exponential decay:  
  \[
  w(x) = \exp(-\alpha \cdot \|x - O(t)\|^2)
  \]
  where \( O(t) \) is the **shifting optimum** and \( \alpha \) is the **selection strength**.
- **Mutation** introduces random changes in the phenotype.
- **Selection** determines which organisms reproduce based on their fitness.
- **The optimum shifts** in a predefined direction, requiring continuous adaptation.

---

## Quickstart Guide

### ** Create & Activate Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate      # On Windows
