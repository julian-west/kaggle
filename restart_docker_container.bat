docker start kaggle-comp
docker exec -it kaggle-comp jupyter lab --ip 0.0.0.0 --port 8080 --no-browser --allow-root