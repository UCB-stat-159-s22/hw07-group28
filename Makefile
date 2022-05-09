.PHONY : env
env :
	mamba env create -f environment.yml --name hw7_env
	conda activate hw7_env
	python -m ipykernel install --user --name hw7_env --display-name "HW7"
    
.PHONY : all
all :
	jupyter execute EDA.ipynb
	jupyter execute model.ipynb
	jupyter execute main.ipynb

