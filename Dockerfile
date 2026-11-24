# Dashboard del proyecto (solo inferencia en CPU). Construir desde la raíz del repositorio:
#   docker build -t nagfuse . && docker run -p 8050:8050 nagfuse
# Para la inferencia en vivo, montar el modelo exportado:
#   docker run -p 8050:8050 -v /ruta/nagfuse_seed00.onnx:/app/modelos/nagfuse_seed00.onnx nagfuse
FROM python:3.11.9-slim
WORKDIR /app
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dashboard/ .
ENV PORT=8050
EXPOSE 8050
CMD ["sh", "-c", "gunicorn app:server --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:${PORT}"]
