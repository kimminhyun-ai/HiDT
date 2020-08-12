FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgtk2.0-dev

COPY requirements.txt .

#install python package
RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD python app.py