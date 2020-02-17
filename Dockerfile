FROM python:3.7-slim

COPY ./requirements.txt .
RUN pip install -r requirements.txt

WORKDIR ./kaggle

EXPOSE 8080

CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8080", "--no-browser", "--allow-root"]

