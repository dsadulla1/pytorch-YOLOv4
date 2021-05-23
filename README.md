# YOLOv4 Object Detection App

## Environment Details and Setup
Use `env-setup.md` to setup your enviroment correctly.

## Model Weights
Place the yolov4.pth model weights and update `weightfile`  in `app-inference-server.py`'s `load_model` function.

## Run the app
Edit the paths to various log files in `sh app-start-trigger.sh`.
Run the app using `sh app-start-trigger.sh >> ~/logs/app-start-trigger.log 2>&1 &` which will trigger an inference server, redis queue, fastapi backend, streamlit frontend.

Endpoints:
 - FastAPI: http://localhost:8000/predict
 - Streamlit: http://localhost:8501
 - Redis: http://localhost:6379

You can clear the redis queue using `FLUSHALL` within `redis-cli` on commandline.

## Stop the servers
sudo lsof -t -i tcp:8000 | xargs kill -9
sudo lsof -t -i tcp:8501 | xargs kill -9
sudo lsof -t -i tcp:6379 | xargs kill -9

To kill the `app-inference-script.py` script, use `ps aux | grep app-inference-script.py` and use the process-id in `kill -9 <PID>`
