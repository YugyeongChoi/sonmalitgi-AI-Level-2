FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 PATH="/opt/venv/bin:$PATH" PYTHONPATH=/app/model_source
RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN /opt/venv/bin/pip install --no-cache-dir "numpy<3" "fastapi>=0.115,<1" "uvicorn[standard]>=0.34,<1" "pillow>=11,<13"
WORKDIR /app
COPY deployment/ai_server ai_server
COPY word_dataset.py hands_dataset.py web_adapter.py model_source/
COPY ai_server/ml/word_model.py model_source/ai_server/ml/word_model.py
LABEL app.sonmalitgi.unit5.training-signers="REAL01,REAL02,REAL03,REAL04,REAL05,REAL06,REAL07,REAL08,REAL09,REAL10,REAL11,REAL12"
COPY unit5.pt models/unit5.pt
LABEL app.sonmalitgi.unit5.word-training-signers="REAL01-REAL12;custom-driving-v1"
COPY unit5_words.pt models/unit5_words.pt
ENV UNIT5_WORD_CHECKPOINT=/app/models/unit5_words.pt
ENV UNIT5_CHECKPOINT=/app/models/unit5.pt
ENV WORD_MODEL_ADAPTER=web_adapter:create_model HANDS_WORD_CHECKPOINT=/app/models/best.pt
COPY intermediate_v4 intermediate_v4
ENV WORD_MODEL_ADAPTER=intermediate_v4.adapter:create_model INTERMEDIATE_CHECKPOINT=/app/intermediate_v4/best.pt
RUN useradd --uid 10001 --create-home app
USER app
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn ai_server.main:app --host 0.0.0.0 --port ${PORT:-8080} --ws-max-size 200000"]
