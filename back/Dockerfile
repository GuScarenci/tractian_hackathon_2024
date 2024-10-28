FROM python:3.11 as build
RUN apk add libpq-dev build-base python3-dev

WORKDIR /srv/

COPY ./requirements.txt /srv/
RUN python -m venv /srv/venv
RUN . /srv/venv/bin/activate && pip install wheel && pip install -r requirements.txt


FROM python:3.11
RUN apk add libpq

WORKDIR /srv/

COPY --from=build /srv/ /srv/
COPY ./src/ /srv/
COPY ./imagem.jpg /srv/

ENTRYPOINT ["venv/bin/python", "main.py"]
