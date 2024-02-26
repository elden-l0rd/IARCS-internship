FROM python:3.11-bullseye

WORKDIR /usr/src/app

RUN        pip3 install virtualenv
RUN        virtualenv /usr/src/app
RUN        /usr/src/app/bin/pip install --upgrade pip

COPY requirements.in ./

# RUN pip install -r requirements.txt

COPY ./helloworld.py ./helloworld.py

CMD [ "python3", "./helloworld.py" ]