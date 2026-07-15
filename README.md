# ddDT-GFM
## Generalized graph foundation models as versatile data-driven digital twins for complex technological systems

[![PyPI version](https://img.shields.io/pypi/v/ddDT-GFM.svg)](https://pypi.org/project/ddDT-GFM/)
[![Python versions](https://img.shields.io/pypi/pyversions/ddDT-GFM.svg)](https://pypi.org/project/ddDT-GFM/)
[![Documentation Status](https://readthedocs.org/projects/dddt-gfm/badge/?version=latest)](https://dddt-gfm.readthedocs.io/en/latest/)
[![License: BSD-2-Clause](https://img.shields.io/badge/License-BSD--2--Clause-blue.svg)](LICENSE)

## Installation

```bash
pip install ddDT-GFM
```

The models and dataloaders additionally depend on the PyTorch Geometric stack
(`torch_geometric`, `torch_sparse`, `torch-spatiotemporal`), which often needs
wheels matched to your `torch`/CUDA build. Install those via the optional extra:

```bash
pip install "ddDT-GFM[full]"
```

See the [documentation](https://dddt-gfm.readthedocs.io) for details.

## Abstract
Digital twins are comprised of computational models that mimic the 'as built' characteristics of devices, systems, and networks of systems whose performance in the real world warrants quantitative and critical assessment. 
The literature on constructing digital twins is historically focused around task-specific, physics-based models that seek to understand the device from first principles, thereby constructing an idealized understanding of the system. 
However, these ``empirical digital twins" (eDT) can be quite difficult to construct, as the level of detail required to accurately model the complex physics of many devices is often missing or expensive to obtain, especially for systems with widespread deployment. 
Additionally, eDTs generally assume the device is working as intended; in practice, many systems experience some form of derating or degradation that causes them to operate off-specification. 
As it is often the goal of a digital twin model to quantify these departures from the idealized system, it is quite difficult to separate derated assumptions from the expected model output. 
In contrast, data-driven digital twins (ddDT) seek to model the system as it actually is based on real observations about the device in question. 
Additionally, ddDTs generally utilize a flexible model architecture (typically an artificial neural network) to avoid injecting implicit bias into the system. 
This flexibility also lends itself to another advantage: modularity. 
Digital twins are often utilized to answer multiple different questions about the system. 
With a ddDT, it is possible to train the model in a self-supervised manner via a reconstruction objective in order to obtain an encoder module. 
This module can then be used as a Foundation Model (FM) for different tasks without either constructing a entirely new task-specific eDT or training another ddDT from scratch. 
This work presents a unified pipeline for constructing data-driven Foundation Models for three exemplifying cases: solar photovoltaic fleets, direct ink write additive manufacturing, and laser powder bed fusion. 
Although these three systems are conceptually very different, the presented Foundation Model utilizes the flexibility of spatiotemporal graph neural networks (st-GNNs) to apply the same methodology to each case, allowing scientists to focus on their scientific objectives rather than troubleshooting a overwhelmingly detailed modeling pipeline. 

## Downloading data
You can download the data used in this work from OSF with the `util` module of this package. 
For example, to download a PV site:
```py
from dddt_gfm.util.io import OSF_download
OSF_download(0, 'downloaded_test.csv', './')
```

## Citation

If you use this software, please cite the accompanying paper:

> Pierce, B. G., Aung, H. H., Ciardi, T. G., Hernandez, K. J., Wieser, R. J., Yue, W., Fan, Y., Harding Bradley, A. C., Rajamohan, B. P., Barcelos, E. I., Jimenez, J. C., Spears, B. K., Giera, B., Gao, R. X., Li, M., Davis, K. O., Bruckman, L. S., Wu, Y., Tripathi, P. K., & French, R. H. Generalized graph foundation models as versatile data-driven digital twins for complex technological systems. *Scientific Reports* (2026). DOI: _to be added upon publication_.

```bibtex
@article{pierce2026ddltgfm,
  title   = {Generalized graph foundation models as versatile data-driven digital twins for complex technological systems},
  author  = {Pierce, Benjamin G. and Aung, Hein Htet and Ciardi, Thomas G. and Hernandez, Kristen J. and Wieser, Raymond J. and Yue, Weiqi and Fan, Yangxin and Harding Bradley, Alexander C. and Rajamohan, Balashanmuga Priyan and Barcelos, Erika I. and Jimenez, Jayvic C. and Spears, Brian K. and Giera, Brian and Gao, Robert X. and Li, Mengjie and Davis, Kristopher O. and Bruckman, Laura S. and Wu, Yinghui and Tripathi, Pawan K. and French, Roger H.},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {DOI to be added upon publication}
}
```