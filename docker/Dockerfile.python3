FROM python:3.6

RUN mkdir /code

COPY ./docker/requirements.txt /code/
RUN pip install -r /code/requirements.txt

RUN mkdir /code/athena
RUN mkdir /code/tests

COPY ./ /code/athena/
COPY ./docker/tests/* /code/tests/

WORKDIR /code

RUN pip install -e /code/athena

CMD python /code/tests/basic.py && python /code/tests/genetic.py

