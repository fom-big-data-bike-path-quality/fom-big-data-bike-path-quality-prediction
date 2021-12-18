FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

COPY app.py /app/app.py
COPY test_values.py /app/test_values.py
COPY requirements.txt requirements.txt
COPY analytics /app/analytics
COPY results /app/results

RUN python -m pip install --upgrade pip
RUN pip install flake8 pytest
RUN pip install pandas
RUN pip install matplotlib
RUN pip install sklearn
RUN pip install torch
RUN pip install tqdm
RUN pip install seaborn
RUN pip install telegram-send
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install requests

CMD python app.py
