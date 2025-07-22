# Gunakan base image Python
FROM python:3.9-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements dan instal library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek ke dalam container
COPY . .

# Beri tahu Gunicorn untuk menjalankan app.py di port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]