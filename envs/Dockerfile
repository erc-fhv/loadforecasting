# Use Miniconda base image
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the Conda environment definition file
COPY envs/env_linux.yml .

# Create the Conda environment from the environment file
RUN conda env create -f env_linux.yml

# Ensure all future commands use the created Conda environment
SHELL ["conda", "run", "-n", "load_forecasting", "/bin/bash", "-c"]
