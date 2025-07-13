# Dockerfile
FROM kubeflownotebookswg/jupyter-pytorch-full:v1.8.0

WORKDIR /home/jovyan/flight_pattern_of_life

COPY --chown=jovyan:users requirements.txt flight_pattern_of_life/requirements.txt
RUN pip install -r flight_pattern_of_life/requirements.txt

COPY --chown=jovyan:users . /home/jovyan/flight_pattern_of_life

ENTRYPOINT ["python3", "main.py", "experiment=experiment_fast_run"]
