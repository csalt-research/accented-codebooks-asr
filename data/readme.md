## Dataset Creation

:white_check_mark: This folder :file_folder: contains  the <samp>train</samp>, <samp>dev</samp> and <samp>test</samp> splits used for all our experiments. Additionally, the folder also contians scripts used to generate those splits. More details can be found [here](https://github.com/csalt-research/accented-codebooks-asr/tree/main/data).
The `dataset.tar.gz` tar file contains the following csvs: <kbd>train.tsv</kbd>,<kbd>train_random_100h.tsv</kbd>, <kbd>train_equi_100h.tsv</kbd>,<kbd>dev.tsv</kbd>,<kbd>dev_small.tsv</kbd>,<kbd>test.tsv</kbd> and <kbd>test_small.tsv</kbd>.

The exact details of how these files are used can be found in [our paper](https://arxiv.org/abs/2310.15970).

### Prerequisites

* Clone the repository
```sh
git clone https://github.com/csalt-research/accented-codebooks-asr.git
```
* Install the required python packages:
```sh
pip install -r data/requirements.txt
```

### Running the script
Go to the data folder and execute `create_data.sh` script.
```sh
cd data && ./create_data.sh
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