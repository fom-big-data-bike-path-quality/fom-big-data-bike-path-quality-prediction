[![Issues](https://img.shields.io/github/issues/florianschwanz/fom-big-data-bike-path-quality-prediction)](https://github.com/florianschwanz/fom-big-data-bike-path-quality-prediction/issues)

<br />
<p align="center">
  <a href="https://github.com/florianschwanz/fom-big-data-bike-path-quality-prediction">
    <img src="./logo.png" alt="Logo" width="80" height="80">
  </a>

  <h1 align="center">Bike Path Quality (Observatorio)</h1>

  <p align="center">
    Flask based web service that predicts surface type based on bike activity time series data
  </p>
</p>

## About The Project

The aim of this app is to provide REST endpoints which accept bike activity accelerometer measurements and predict the surface type. 
Therefore it makes use of [models](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-model) trained by an
 [analytics component](https://github.com/fom-big-data-bike-path-quality/fom-big-data-bike-path-quality-analytics).

### Built With

* [Flask](https://flask.palletsprojects.com/)

## Installation

Initialize the submodules of this repository by running the following commands.

```shell script
git submodule init
git submodule update
```

Install the following dependencies to fulfill the requirements for this project to run.

```shell script
python -m pip install --upgrade pip
pip install flask
pip install torch
```

## Usage

Run this command to start the dev server.

```shell script
python app.py
```

## Roadmap

See the [open issues](https://github.com/florianschwanz/fom-big-data-bike-path-quality-prediction/issues) for a list of proposed features
 (and known issues).

## Contributing

Since this project is part of an ongoing Master's thesis contributions are not possible as for now.

## License

Distributed under the GPLv3 License. See [LICENSE.md](./LICENSE.md) for more information.

## Contact

Florian Schwanz - florian.schwanz@gmail.com

## Acknowledgements

Icon made by Freepik from www.flaticon.com
