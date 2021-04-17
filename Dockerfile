FROM python:3.8

WORKDIR /iatos-model-app

COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1 ffmpeg

COPY ./saved_model ./saved_model

COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]