# cstool Docker image

This Docker image dramatically simplifies the installation procedure for cstool. The docker image comes with all dependencies installed. The user is expected to add scripts (for calculating cross sections) and parameters to the `/data` volume.

## Installing
Go to the present directory, and run
```sh
docker build -t cstool .
```

## Running
Prepare a folder, containing the following items:
* A script for computing your cross sections
* Input parameters

As an example, we will use the present git repository, which contains the desired script in `examples/cs.py` and some input data in the `data` folder.

Go to this folder, and run
```sh
docker run -it -v $(pwd):/data cstool
```
The `-v` flag tells docker to mount the current directory (`$(pwd)`) to the `/data` volume in the Docker image. The `-it` flags open an interactive terminal.

Then, you can run the script:
```sh
python3 examples/cs.py data/materials/silicon.yaml
```
