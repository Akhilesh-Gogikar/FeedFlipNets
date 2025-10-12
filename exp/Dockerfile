FROM python:3.11-slim

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir --upgrade pip \
 && if [ -f requirements-lock.txt ]; then pip install -r requirements-lock.txt; fi \
 && pip install --no-cache-dir -e . \
 && pip install --no-cache-dir typer pyyaml

CMD ["python", "-m", "exp.runner", "run", "--subset", "MWR"]

