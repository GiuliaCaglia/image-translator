# Base Image
FROM python:3.11

# User Setup
# RUN useradd -ms /bin/bash mlflow-user
RUN addgroup --gid 1001 mlflow-user
RUN adduser --uid 1001 --ingroup mlflow-user mlflow-user
USER mlflow-user
WORKDIR /home/mlflow-user/mlflow

# Dependencies
ENV PATH=/home/mlflow-user/.local/bin:$PATH
RUN pip3 install --upgrade --user pip setuptools
RUN pip3 install --user mlflow

# Expose Port
EXPOSE 5000

# Start service
# ENTRYPOINT mlflow ui --host 0.0.0.0
CMD ["mlflow", "ui", "--host", "0.0.0.0"]
