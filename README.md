<p align="center">
  <a href="https://github.com/csalt-research">
    <img src="https://avatars.githubusercontent.com/u/43694569?s=200&v=4" alt="CSALT @ IITB" width="150" height="150">
  </a>
  <h3 align="center">Accented Speech Recognition With Accent-specific Codebooks</h3>
  <p align="center"> Empirical Methods in Natural Language Processing(EMNLP) 2023
    <br/>
    <br/>
  </p>
</p>
  
<a href="https://arxiv.org/abs/2310.15970"> <img src="https://img.shields.io/badge/PDF-Arxiv-teal"></a> ![Downloads](https://img.shields.io/github/downloads/csalt-research/accented-codebooks-asr/total.svg) ![Contributors](https://img.shields.io/github/contributors/csalt-research/accented-codebooks-asr?color=dark-green) ![Forks](https://img.shields.io/github/forks/csalt-research/accented-codebooks-asr?style=social) ![Stargazers](https://img.shields.io/github/stars/csalt-research/accented-codebooks-asr?style=social) 

## Table Of Contents

* [About The Repository](#about-the-repository)
* [Getting Started](#getting-started)
  * [Prerequisites and Installation](#prerequisites-and-installation)
  * [Training](#training)
* [Roadmap](#roadmap)
* [Dataset Statistics](#dataset-statistics)
* [Contributing](#contributing)
* [Authors](#authors)
* [Citation](#citation)
* [License](#license)

## About The Repository

This repository hosts the artefacts pertaining to [our paper](https://arxiv.org/abs/2310.15970) **<samp>Accented Speech Recognition With Accent-specific Codebooks</samp>** accepted to the main conference of  ***EMNLP 2023***.

The main contributions of our paper are as follows:

:mag_right:  A new <samp>accent adaptation technique</samp> that uses a set of *`learnable codebooks`* and a new *`beam-search decoding`* algorithm to achieve significant performance improvement on both seen and unseen accents. 

:white_check_mark: <samp>Reproducible splits</samp> on Commonvoice dataset for *accented ASR* setup to facilitate fair comparisons across existing and new accent adaptation techniques.

## Getting Started

The repository contains two folders:
* [<kbd>data :file_folder: </kbd>](https://github.com/csalt-research/accented-codebooks-asr/tree/main/data) - Contains the <samp>train</samp>, <samp>dev</samp> and <samp>test</samp> splits used for all our experiments. Additionally, the folder also contians scripts used to generate those splits. More details can be found [here](https://github.com/csalt-research/accented-codebooks-asr/tree/main/data).
*  [<kbd>espnet_code :file_folder: </kbd>](https://github.com/csalt-research/accented-codebooks-asr/tree/main/espnet_code) - Contains code to run our experiments on [ESPnet](https://github.com/espnet/espnet) toolkit. Detailed instruction on how to run our experiments can be found [here](#prerequisites-and-installation).


### Prerequisites and Installation

* ESPnet installation: Follow the instructions [here](https://espnet.github.io/espnet/installation.html).
* Clone the repository containing our code and dataset.
```sh
git clone https://github.com/csalt-research/accented-codebooks-asr.git
```
* Additionally, to run the dataset creation script, run the following:
```sh
pip install -r accented-codebooks-asr/data/requirements.txt
```

### Training

1. Extract the csvs from the `tar` file in data folder
```sh
tar  -xvzf accented-codebooks-asr/data/dataset.tar.gz 
```
2. Copy the files from espnet_code into ESPnet egs
```sh
cp -r accented-codebooks-asr/espnet_code/* <espnet_root_folder>/egs/commonvoice/asr1
```
3. Enter the path to the the directory hosting our splits in `run.sh`
```python
csvdir=  # Path to the directory hosting all our csvs.
```
4. Run the script
```sh
./run.sh
```

## Dataset Statistics
The statistics of <samp>train</samp>, <samp>dev</samp> and <samp>test</samp> splits used in our experiments are as follows:

| Accent | Train 100h (in hours) | Train (in hours) | Dev (in hours) | Test (in hours) |
| - | - | - | - | - |
| Australia | 6.95 | 45.36 | 4.33 | 0.46 |
| Canada | 6.79 | 41.13 | 1.16 | 1.21 |
| England | 19.51 | 119.9 | 3.22 | 1.65 |
| Scotland | 2.69 | 16.21 | 0.23 | 0.16 |
| US | 64.12 | 400.1 | 8.32 | 4.87 |
| Africa | - | - | - | 1.71 |
| Hongkong | - | - | - | 0.52 |
| India | - | - | - | 0.58 |
| Ireland | - | - | - | 1.94 |
| Malaysia | - | - | - | 0.39 |
| Newzealand | - | - | - | 2.11 |
| Philippines | - | - | - | 0.90 |
| Singapore | - | - | - | 0.64 |
| Wales | - | - | - | 0.27 |




## Roadmap

See the [open issues](https://github.com/csalt-research/accented-codebooks-asr/issues) for a list of proposed features (and known issues) relevant to this work. For <samp>ESPnet</samp> related features/issues, checkout their [github repository](https://github.com/espnet/espnet/).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/csalt-research/accented-codebooks-asr/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please open an individual PR for each suggestion.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add appropriate commit message'`). The correct way to write your commit message can be found [here](https://www.conventionalcommits.org/en/v1.0.0/)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## Authors

* **Darshan Prabhu** - *M.Tech, CSE, IIT Bombay* - [Darshan Prabhu](https://www.linkedin.com/in/darshan-prabhu/)
* **Preethi Jyothi** - *Associate Professor, CSE, IIT Bombay* - [Preethi Jyothi](https://www.cse.iitb.ac.in/~pjyothi/)
* **Sriram Ganapathy** - *Associate Professor, EE, IISc Bangalore* - [Sriram Ganapathy](http://www.leap.ee.iisc.ac.in/sriram/)
* **Vinit Unni** - *Ph.D, CSE, IIT Bombay* - [Vinit Unni](https://www.linkedin.com/in/vinit-unni/)


## Citation

If you use this code for your research, please consider citing our work.

```bibtex
@misc{prabhu2023accented,
      title={Accented Speech Recognition With Accent-specific Codebooks}, 
      author={Darshan Prabhu and Preethi Jyothi and Sriram Ganapathy and Vinit Unni},
      year={2023},
      eprint={2310.15970},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

Distributed under the MIT License. See [LICENSE](https://github.com/csalt-research/accented-codebooks-asr/blob/main/LICENSE.md) for more information.
