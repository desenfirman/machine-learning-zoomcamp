FROM python:3.9.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker 
COPY ["predict.py", "./"]
COPY ["model/", "./model"]

# We need to expose the 8080 port because we're not able to communicate with Docker outside it
EXPOSE 8080

# If we run the Docker image, we want our app to be running
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]
