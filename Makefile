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

.PHONY : html-hub
html-hub :
	jupyter-book config sphinx .
	sphinx-build  . _build/html -D html_baseurl=${JUPYTERHUB_SERVICE_PREFIX}/proxy/absolute/8000

.PHONY : clean
clean :
	rm -f figures/*.png
	rm -r _build/*
