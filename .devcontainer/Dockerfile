FROM continuumio/miniconda3:23.3.1-0

# Install Jupyter Notebook

RUN conda create -n ml-zoomcamp python=3.9
# SHELL ["conda", "run", "-n", "ml-zoomcamp", "/bin/bash", "-c"]
RUN conda install -n ml-zoomcamp numpy pandas scikit-learn seaborn jupyter ipykernel

RUN apt-get update && apt-get install -y docker.io

RUN useradd -u 1000 -m user
USER user

# Install VS Code Server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# COPY requirements.txt requirements.txt 
# RUN pip install -r requirements.txt

# COPY ext_requirements.txt ext_requirements.txt 
# RUN pip install -r ext_requirements.txt

# Set the command to start the Jupyter Notebook server and VS Code Server
CMD ["sh", "-c", "code-server --auth none --disable-telemetry --allow-http --no-auth --port 8080 & conda run -n ml-zoomcamp jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"]
