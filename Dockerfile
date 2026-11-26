FROM python:3.15.2-slim-trixie

WORKDIR /app

COPY requirements-reconstructed.txt .

RUN pip install --no-cache-dir -r requirements-reconstructed.txt

COPY . .

CMD ["bash"]
