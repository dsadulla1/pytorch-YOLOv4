# YOLOv4 Object Detection App

## Environment Details and Setup
Use `app-env-setup.md` to setup your enviroment correctly.

## Model Weights
Place the `yolov4.pth` file (model weights) at your preferred location and update `weightfile` parameter  in `app-inference-server.py`'s `load_model` function.

## Run the app
Edit the paths to various log files in `sh app-start-trigger.sh`.
Run the app using `sh app-start-trigger.sh >> ~/logs/app-start-trigger.log 2>&1 &` which will trigger an inference server, redis queue, fastapi backend, streamlit frontend.

Endpoints:
 - FastAPI: http://localhost:8000/predict
 - Streamlit: http://localhost:8501
 - Redis: http://localhost:6379

You can clear the redis queue using `FLUSHALL` within `redis-cli` on commandline.

## Monitoring processes
How to check the current length of the current redis-queue: `watch -n 1 redis-cli -h localhost -p 6379 -n 0 llen image_queue`
How to find the length of the log file: `watch -n 1 wc -l  ~/logs/app-server.log`
How to check the load on GPU: `watch -n 1 nvidia-smi`

`Ctrl + C` to quit from `watch`.

## Stop the servers
In the following order:
```bash
sudo lsof -t -i tcp:8000 | xargs kill -9
sudo lsof -t -i tcp:8501 | xargs kill -9
sudo lsof -t -i tcp:6379 | xargs kill -9
```

Before shutting down redis server, feel free to `FLUSHALL` within `redis-cli`

To kill the `app-inference-script.py` script, use `ps aux | grep app-inference-script.py` and use the process-id in `kill -9 <PID>`
