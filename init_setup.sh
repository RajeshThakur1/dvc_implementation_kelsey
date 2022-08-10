echo [$(date)]: "START"
echo [$(date)]: "creating environment"
conda create --prefix ./env2 python=3.8 -y
echo [$(date)]: "activate environment"
conda activate ./env2
echo [$(date)]: "install requirements"
pip install -r requirements.txt
#echo [$(date)]: "export conda environment"
#conda env export > conda.yaml
# echo "# ${PWD}" > README.md
echo [$(date)]: "first commit"
#git add .
#git commit -m "first commit"
echo [$(date)]: "installing the tensorflow in macos"
conda install -c apple tensorflow-deps -y
pip install tensorflow-macos

echo [$(date)]: "installing the jupyter notebook and pandas"
conda install -c conda-forge -y pandas jupyter

echo [$(date)]: "END"

# to remove everything -
# rm -rf env/ .gitignore conda.yaml README.md .git/