cd /home/deepak_sadulla/projects/pytorch-YOLOv4/
redis-server >> ~/logs/redis.log 2>&1 &
python app-inference-server.py >> ~/logs/app-inference-server.log 2>&1 &
uvicorn app-server:app --reload >> ~/logs/app-server.log 2>&1 &
streamlit run app-client.py >> ~/logs/app-client.log 2>&1 &