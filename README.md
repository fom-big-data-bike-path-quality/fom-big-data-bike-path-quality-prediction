[![Issues](https://img.shields.io/github/issues/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-prediction)](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-prediction/issues)

<br />
<p align="center">
  <a href="https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-prediction">
    <img src="./logo.png" alt="Logo" width="80" height="80">
  </a>

  <h1 align="center">Bike Path Quality (Observatorio)</h1>

  <p align="center">
    FastAPI based web service that predicts surface type based on bike activity time series data
  </p>
</p>

## About The Project

The aim of this app is to provide REST endpoints which accept bike activity accelerometer measurements and predict the surface type. 
Therefore it makes use of the [results](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-results) trained by an
 [analytics component](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-analytics).

### Built With

* [FastAPI](https://fastapi.tiangolo.com/)

## Installation

Initialize the submodules of this repository by running the following commands.

```shell script
git submodule init
git submodule update
```

Install the following dependencies to fulfill the requirements for this project to run.

```shell script
python -m pip install --upgrade pip
pip install flake8 pytest
pip install pandas
pip install matplotlib
pip install sklearn
pip install torch
pip install tqdm
pip install seaborn
pip install telegram-send
pip install fastapi
pip install uvicorn
pip install requests
```

## Usage

Run this command to start the dev server.

```shell script
python app.py
```

### Usage (local docker)

Run this command to run the docker container locally.

```shell
docker build -t bike-path-quality-prediction .
docker run -p 8000:8000 bike-path-quality-prediction
```

## Roadmap

See the [open issues](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-prediction/issues) for a list of proposed features
 (and known issues).

## Contributing

Since this project is part of an ongoing Master's thesis contributions are not possible as for now.

## License

Distributed under the GPLv3 License. See [LICENSE.md](./LICENSE.md) for more information.

## Contact

Florian Schwanz - florian.schwanz@gmail.com

## Acknowledgements

Icon made by Freepik from www.flaticon.com
