FROM python:3.10-slim

# Paquetes de sistema mínimos
RUN apt-get update && apt-get install -y \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Copiar scripts y requirements
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Usar backend sin interfaz gráfica
ENV MPLBACKEND=Agg

# Ejecuta el pipeline completo (ejemplo)
CMD ["python", "analisis.py"]