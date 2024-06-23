FROM python:3.9.19-slim  

# get the latest pip version
RUN pip install -U pip  

# to manage dependencies
RUN pip install pipenv

# set workdir 
WORKDIR /app 

# copy packages files into the workdir on the container
COPY ["Pipfile", "Pipfile.lock", "./"]   

# to avoid creating an env on the container, add --system. 
RUN pipenv install --system --deploy 

# copy the model and prediction script into the dir
COPY ["predict.py", "lin_reg2.bin", "./"] 

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]





