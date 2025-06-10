# Centrifugal Compressor Model

A Python implementation of a centrifugal compressor model, including geometry definition, thermodynamic analysis, and performance evaluation. This project is suitable for engineering analysis, educational purposes, and parametric or optimization studies.

## Features

- Thermodynamic modeling using CoolProp
- Detailed geometric definition
- Parametric studies and optimization examples
- Based on engineering literature and empirical loss models

## Requirements

Install dependencies via:
```bash
pip install -r requirements.txt
```

## Structure

```
compressor_model.py        # Core model definition
examples/
├── single_run.py          # Basic simulation example
├── parametric_study.py    # Efficiency vs flow coefficient
└── optimization_study.py  # Geometry optimization for max efficiency
```

## License

[MIT License](LICENSE)

## Citation

If you use this model or its components in your work, please cite:

Frate, G.F., Benvenuti, M., Chini, F., and Ferrari, L. (2024).  
_Optimal design of centrifugal compressors in small-size high-temperature Brayton heat pumps_,  
Proceedings of 37th International Conference on Efficiency, Cost, Optimization, Simulation and Environmental Impact of Energy Systems (ECOS),
Rhodes, Greece, 30 June - 5 July 2024, doi: [10.52202/077185-0031](https://doi.org/10.52202/077185-0031)
