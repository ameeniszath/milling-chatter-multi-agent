FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY chatter_agents ./chatter_agents
COPY data ./data
COPY results ./results
EXPOSE 8000
CMD ["uvicorn", "chatter_agents.api:app", "--host", "0.0.0.0", "--port", "8000"]
