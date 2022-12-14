# dvc-project-template
DVC project template

## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE
```commandline
export PYTHONPATH=/Users/rajesh/Desktop/zuma/kelsey_experiments

```

```bash
conda create --prefix ./env python=3.8 -y
```
activate the env

```bash
conda activate ./env
```
To remove this long prefix in your shell prompt, modify the env_prompt setting in your .condarc file:
Note :- Not Recommended to execute if we don't have the valid reason

```commandline
conda config --set env_prompt '({name})'
```
OR
Install Tensorflow
``
conda install -c apple tensorflow-deps
``

```commandline
https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706

```

```commandline
pip install tensorflow-macos
```
Install metal plugin:
```commandline
pip install tensorflow-metal
```
Install Jupyter Notebook & Pandas

```bash
source activate ./env
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 05- initialize the dvc project
```bash
dvc init --subdir
```

To Add the remote repositery to track all the data and Model 

```commandline
 dvc remote add -d remote_storage Your_S3_dir_URL -f
```
To see the commited file
```commandline
 git log --oneline
```

we can checkout the Data 

```commandline
git checkout HEAD^1 data/train.csv.dvc
```

while installing the transformers if we get the rust compiler issue
Solution
```commandline
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.carge/env
```

to get our Data back

```commandline

 dvc checkout data/mark4/intent_model/train.csv.dvc

```


### STEP 06- commit and push the changes to the remote repository