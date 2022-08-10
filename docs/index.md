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

### STEP 06- commit and push the changes to the remote repository