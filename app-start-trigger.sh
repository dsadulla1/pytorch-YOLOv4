# conda activate Dev
cd /home/deepak_sadulla/projects/pytorch-YOLOv4/
/usr/local/bin/redis-server >> ~/logs/redis.log 2>&1 &
/home/deepak_sadulla/miniconda3/envs/Dev/bin/python app-inference-server.py >> ~/logs/app-inference-server.log 2>&1 &
/home/deepak_sadulla/miniconda3/envs/Dev/bin/uvicorn app-server:app >> ~/logs/app-server.log 2>&1 &
/home/deepak_sadulla/miniconda3/envs/Dev/bin/streamlit run app-client.py >> ~/logs/app-client.log 2>&1 &