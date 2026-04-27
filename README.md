# Kuwahara Filters

This repository contains Python implementations of the Kuwahara filter family, written for a university report on the evolution of these algorithms from the original 1976 formulation to more recent approaches.

## Implemented Filters

The project includes the following filter implementations:

*   **Classic Kuwahara Filter (1976):** original algorithm using 4 square subwindows and an `argmin` selection
*   **Generalized Kuwahara Filter (Papari et al., 2007):** improved version using circular sectors, Gaussian weighting, and a "soft" weighted sum
*   **Anisotropic Kuwahara Filter (Kyprianidis et al., 2009):** algorithm that adapts the filter's shape based on the local structure of the image, guided by the structure tensor

Additionally, you will find a `flawed` version of the classic Kuwahara filter to demonstrate color bleeding artifacts.

## Project Scope

This is an educational implementation prioritizing readability and faithfulness to the original mathematical formulations over performance. The anisotropic filter uses CuPy to keep runtime practical, but remaining code is pure NumPy and opencv.

## Getting Started

### Prerequisites

You will need [Poetry](https://python-poetry.org/) installed on your system.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/SaharaSurfer/kuwahara-filters.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd kuwahara-filters
    ```
3.  Install the project dependencies using Poetry:
    ```bash
    poetry install
    ```

### Usage

Usage examples are located in the Jupyter notebooks (`*.ipynb`). For programmatic use:

```python
import cv2
from kuwahara_filters import anisotropic_kuwahara

image = cv2.imread('my_image.png', cv2.IMREAD_COLOR_RGB)
stylized_image = anisotropic_kuwahara(image_rgb, radius=7)
```
