# F2M

# Overview
This repository contains the source code for F2M: Improving Skin Disease Recognition by Fusing Multi-source and Multi-scale Image Features
![image](https://github.com/Wang-XingYi/RejoinViG/blob/main/Images/network.jpg)

# Usage

## Installation 
- Python 3.7.13
```
conda install pytorch==1.10.0 torchvision==0.11.0
```
```
pip install -r requirements.txt
```
### Dataset
```

- Dataset Structure
```
│dataset/
├──Train/
│  ├── eczema/
│  ├── others/
│  ├── psoriasis/
├──Val/
│  ├── eczema/
│  ├── others/
│  ├── psoriasis/
├──Test/
│  ├── eczema/
│  ├── others/
│  ├── psoriasis/
├──Train_List.txt
├──Val_List.txt
├──Test_List.txt
```

### Train F2M:
```
python train.py
```

### Test F2M:
- Test
```
python test.py
```
- Calculate evaluation indicators
```
python results_evaluate.py
```

