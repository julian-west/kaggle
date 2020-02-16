docker run -v %cd%:/kaggle kaggle --name kaggle-container kaggle

:: open up bash
docker exec -it kaggle-container bash