.PHONY : env
env :
    mamba env create -f environment.yml --name hw7_env
    
.PHONY : all
all :
    jupyter execute main.ipynb
    jupyter execute model.ipynb
