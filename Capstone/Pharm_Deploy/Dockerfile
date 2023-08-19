FROM python:3.10-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./pharm_model.pkl /deploy/
COPY ./norm_weights.pkl /deploy/
COPY static /deploy/static
COPY templates /deploy/templates
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT [ "python", "app.py" ]