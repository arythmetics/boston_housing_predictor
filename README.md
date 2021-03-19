# boston_housing_predictor
This is a simple API that can predict Boston Housing data from sklearn.
The purpose of this is to illustrate how a neural network can be packaged and put into production as a microservice.

## Usage
(a) Clone the Repo

(b) Execute uvicorn from the root folder: uvicorn src.main:app

(c) Execute with Curl locally:
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": [
    0.00632,18.00,2.310,0.0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98

  ]
}'

Testing can be engaged with pytest from the root folder.

## License
[MIT](https://choosealicense.com/licenses/mit/)