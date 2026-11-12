FROM python:3.12-slim AS build
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "numpy<3" "fastapi>=0.115,<1" "uvicorn[standard]>=0.34,<1" "pillow>=11,<13"
WORKDIR /train
COPY word_dataset.py hands_dataset.py train_word_lstm.py ./
COPY data/word/manifest.csv data/word/manifest.csv
COPY data/word/clips/ data/word/clips/
RUN python train_word_lstm.py --words WORD1290 WORD1201 WORD1589 WORD1191 WORD1185 WORD1342 WORD1343 WORD1345 WORD1149 WORD1295 WORD1274 --epochs 60 --eval-every 10 --batch 16 --steps 2 --hidden 64 --layers 1 --out runs/hands_intermediate_experiment

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 PATH="/opt/venv/bin:$PATH" PYTHONPATH=/app/model_source
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY deployment/ai_server ai_server
COPY word_dataset.py hands_dataset.py web_adapter.py model_source/
COPY ai_server/ml/word_model.py model_source/ai_server/ml/word_model.py
COPY --from=build /train/runs/hands_intermediate_experiment/best.pt models/best.pt
COPY --from=build /train/runs/hands_intermediate_experiment/classes.json models/classes.json
LABEL app.sonmalitgi.unit5.training-signers="REAL01,REAL02,REAL03,REAL04,REAL05,REAL06,REAL07"
COPY unit5.pt models/unit5.pt
ENV UNIT5_CHECKPOINT=/app/models/unit5.pt
ENV WORD_MODEL_ADAPTER=web_adapter:create_model HANDS_WORD_CHECKPOINT=/app/models/best.pt
RUN useradd --uid 10001 --create-home app
USER app
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn ai_server.main:app --host 0.0.0.0 --port ${PORT:-8080} --ws-max-size 200000"]
