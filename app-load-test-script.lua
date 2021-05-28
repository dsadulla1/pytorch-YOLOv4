wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

file = io.open("/home/deepak_sadulla/projects/pytorch-YOLOv4/data/json-files/dog.json", "r")
wrk.body = file:read("*a")
file.close()

---------------------------------- Unnecessary ----------------------------------

-- /home/deepak_sadulla/tools/wrk/wrk "http://127.0.0.1:8000/predict" -s /home/deepak_sadulla/projects/pytorch-YOLOv4/app-load-test-script.lua --latency -t 1 -c 1 -d 30s --timeout 100s
-- tail  ~/logs/app-server.log
-- wc -l  ~/logs/app-server.log

-- -R 1
-- wrk.method = "POST"
-- wrk.body   = "foo=bar&baz=quux"
-- wrk.headers["Content-Type"] = "application/x-www-form-urlencoded"

-- wrk.method = "POST"
-- wrk.body = '{"firstKey": "somedata", "secondKey": "somedata"}'
-- wrk.headers["Content-Type"] = "application/json"
-- wrk.body = file:read("*a")