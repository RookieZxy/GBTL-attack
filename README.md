# Learning to Poison Large Language Models During Instruction Tuning
This is the official repository for "Learning to Poison Large Language Models During Instruction Tuning" by [Xiangyu Zhou](www.linkedin.com/in/xiangyu-zhou-71086321a), [Yao Qiang](https://qiangyao1988.github.io/), [Saleh Zare Zade](https://www.linkedin.com/in/saleh-zare-zade-5a821421b/), [Mohammad Amin Roshani](https://www.linkedin.com/in/mohammad-amin-roshani), [Douglas Zytko](https://dougzytko.com/) and [Dongxiao Zhu](https://dongxiaozhu.github.io/)  

![Illustration of our poisoning attack](example.png)

## Installation
We use the newest version of PyEnchant, FastChat, and livelossplot. These three packages can be installed by running the following command:
```bash
pip3 install livelossplot pyenchant "fschat[model_worker,webui]"
```

When you install PyEnchant, it typically requires the Enchant library to be installed on your system. you can install it using the following command:
```bash
sudo apt-get install libenchant1c2a
```

## Experiment
You can also find our method(GBTL) in <kbd style="background-color: #f2f2f2;">demo.ipynb</kbd>.

## Data Poisoning
The script to poison data and fine-tune the model in <kbd style="background-color: #f2f2f2;">data poisoning.py</kbd>.
We also crafted a small sentiment dataset from SST-2 for data poisoning you can find it in <kbd style="background-color: #f2f2f2;">/dataset-sentiment</kbd>.

## Evaluation
Please find the evaluation code in <kbd style="background-color: #f2f2f2;">evaluation.ipynb</kbd>.

## Citation
```bash
@article{qiang2024learning,
  title={Learning to Poison Large Language Models During Instruction Tuning},
  author={Qiang, Yao and Zhou, Xiangyu and Zade, Saleh Zare and Roshani, Mohammad Amin and Zytko, Douglas and Zhu, Dongxiao},
  journal={arXiv preprint arXiv:2402.13459},
  year={2024}
}
```
