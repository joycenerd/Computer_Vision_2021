# Image Resolution

Implementation of Hybrid image -> Blend two images, one preserving only low frequency and another preserving only high frequency

## Dependency

The code is tested in conda environment with Python3, the dependencies includes:

* pillow
* numpy
* matplotlib

```
conda env create -f environment.yml
conda activate hybrid
```

## Execute the code

Change the `DATA_PATH` in `hybrid_image.py` if you are using your data

```
python hybrid_image.py
```

## Results

<img src="./results/1.jpg">
<img src="./results/3.jpg">

## License

This project is released under the MIT license

