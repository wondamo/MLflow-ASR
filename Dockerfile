FROM python:3.10

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
    
COPY train.py .