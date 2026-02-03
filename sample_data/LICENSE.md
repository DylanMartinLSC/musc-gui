# Sample Data License

## MVTec AD Dataset

The sample images in this directory are sourced from the **MVTec Anomaly Detection Dataset (MVTec AD)**.

### Citation

If you use these images or the full MVTec AD dataset, please cite:

```bibtex
@inproceedings{bergmann2019mvtec,
  title={MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={9592--9600},
  year={2019}
}

@article{bergmann2021mvtec,
  title={The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and Batzner, Kilian and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  journal={International Journal of Computer Vision},
  volume={129},
  number={4},
  pages={1038--1059},
  year={2021},
  publisher={Springer}
}
```

### License Terms

The MVTec AD dataset is provided by MVTec Software GmbH under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**.

This means you may:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license

### Full Dataset

These sample images are a small subset for demonstration purposes. For the full dataset (5GB+), visit:

**https://www.mvtec.com/company/research/datasets/mvtec-ad**

### Categories Included

This sample includes images from:
- `bottle` - Glass bottles (normal and contaminated)
- `screw` - Metal screws (normal and defective)
- `metal_nut` - Hexagonal metal nuts (normal and defective)

Each category contains:
- 5 normal (good) images
- 5 anomaly (defective) images

### Disclaimer

The sample images are provided "as is" for testing and demonstration of the MuSc anomaly detection algorithm. For production use or extensive research, please download the complete dataset from MVTec.
