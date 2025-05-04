FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 这里假设你的入口是 gcp/api.py 里的 app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "gcp.api:app"]