## Steps in deploying a model as a web service

* Creating a virtual env with Pipenv
* Creating a script for predicting
* Putting the script into a flask app
* Packaging the app to Docker

``` bash
docker build -t ride-duration-prediction-service:v1 .
```

```
docker run -it --rm -p 9696:9696 ride-prediction-service

```