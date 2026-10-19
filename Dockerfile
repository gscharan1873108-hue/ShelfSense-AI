# Edge simulation:  docker build -t shelfsense .  &&  docker run --cpus=2 --memory=1g -p 8501:8501 shelfsense
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV SHELFSENSE_THREADS=2
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
