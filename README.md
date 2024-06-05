# DeepNano

[Paper] 

***
### System requirements

python=3.9, transformers=4.27.4, biopython=1.83

***
### Setup of DeepNano

The code was executed under `python=3.9` and  `torch=1.13.1+cu116`, we recommend you to use similar package versions.

Install DeepNano: 

```shell
git clone https://github.com/ddd9898/DeepNano.git
cd DeepNano
pip install -r requirements.txt
```




***

### Checkpoints

Our trained models can be downloaded at [link](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/djt20_mails_tsinghua_edu_cn/EksN2AXNcUpFskpeq0AF-PIBpgrGfBuUsiU8GtPkDgRmtQ?e=1OUpw2) .

|                      DeepNano-seq(PPI)                       |                      DeepNano-seq(NAI)                       |                        DeepNano-site                         |                        DeepNano(NAI)                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [8M](https://cloud.tsinghua.edu.cn/f/b33f0f9eb9c14ead966b/?dl=1) | [8M](https://cloud.tsinghua.edu.cn/f/909b5deff3ac475bb23b/?dl=1) | [8M](https://cloud.tsinghua.edu.cn/f/fd930c06f26b46789d38/?dl=1) | [8M](https://cloud.tsinghua.edu.cn/f/4495bb43362942e3b30f/?dl=1) |
| [35M](https://cloud.tsinghua.edu.cn/f/966039751fee49538252/?dl=1) | [35M](https://cloud.tsinghua.edu.cn/f/627604df02404533864b/?dl=1) | [35M](https://cloud.tsinghua.edu.cn/f/b7812ad6f9994f20a760/?dl=1) | [35M](https://cloud.tsinghua.edu.cn/f/e2fe8128d74149ae91a8/?dl=1) |
| [150M](https://cloud.tsinghua.edu.cn/f/ee62b5e41310414496d3/?dl=1) | [150M](https://cloud.tsinghua.edu.cn/f/9244db9d1c114f018f57/?dl=1) | [150M](https://cloud.tsinghua.edu.cn/f/5132868cda8546b6ac00/?dl=1) | [150M](https://cloud.tsinghua.edu.cn/f/0e06245f4737476cbc7d/?dl=1) |
| [650M](https://cloud.tsinghua.edu.cn/f/9e6362ff1b6242738607/?dl=1) | [650M](https://cloud.tsinghua.edu.cn/f/9bb8665f020b410ba1c2/?dl=1) | [650M](https://cloud.tsinghua.edu.cn/f/b09dd329a6d5403fbf9c/?dl=1) | [650M](https://cloud.tsinghua.edu.cn/f/8c1723de686343bd9777/?dl=1) |


***
### Demo

Coming soon..


***
### Reproduction

1. Get predictions on five PPI test datasets:

   ```python
   python test_ppi_ESM2.py
   ```

2. Get predictions on the NAI test dataset:

   ```python
   python test_nai.py
   ```

3. Virtual screening of anti-HSA and anti-GST:

   ```python
   python test_case.py
   python test_background.py  --size 100w
   ```


***
## Contact


Feel free to contact djt20@mails.tsinghua.edu.cn if you have issues for any questions.