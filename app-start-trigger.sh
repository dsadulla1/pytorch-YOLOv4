# conda activate Dev

cd /home/deepak_sadulla/projects/pytorch-YOLOv4/

/usr/local/bin/redis-server >> ~/logs/redis.log 2>&1 &

/home/deepak_sadulla/miniconda3/envs/Dev/bin/python app-inference-server.py --tag_process "GPU-0" --enable_locks "F" --use_cuda "T" >> ~/logs/app-inference-server_GPU_0.log 2>&1 &
sleep 5 && echo "Started GPU-0"

/home/deepak_sadulla/miniconda3/envs/Dev/bin/python app-inference-server.py --tag_process "CPU-0" --enable_locks "F" --use_cuda "F" >> ~/logs/app-inference-server_CPU_0.log 2>&1 &
sleep 5 && echo "Started CPU-0"

/home/deepak_sadulla/miniconda3/envs/Dev/bin/python app-inference-server.py --tag_process "CPU-1" --enable_locks "F" --use_cuda "F" >> ~/logs/app-inference-server_CPU_1.log 2>&1 &
sleep 5 && echo "Started CPU-1"

/home/deepak_sadulla/miniconda3/envs/Dev/bin/python app-inference-server.py --tag_process "CPU-2" --enable_locks "F" --use_cuda "F" >> ~/logs/app-inference-server_CPU_2.log 2>&1 &
sleep 5 && echo "Started CPU-2"

# /home/deepak_sadulla/miniconda3/envs/Dev/bin/uvicorn app-server:app --reload >> ~/logs/app-server.log 2>&1 &
/home/deepak_sadulla/miniconda3/envs/Dev/bin/uvicorn app-server:app >> ~/logs/app-server.log 2>&1 &
echo "API Server Started"

/home/deepak_sadulla/miniconda3/envs/Dev/bin/streamlit run app-client.py >> ~/logs/app-client.log 2>&1 &
echo "API Client Started"