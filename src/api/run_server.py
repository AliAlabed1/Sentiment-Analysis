import os 
import sys
from fastapi import FastAPI,Request,HTTPException
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
# Add the directory containing the `src` folder to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))  # Adjust to reach `src`
sys.path.append(PROJECT_ROOT)
current_path = os.getcwd()
worksapce = current_path.split('src')[0]

print(worksapce)
from utils.logging_utils.app_logger import app_logger
from predicting.predictor import Predict
import joblib

class SentenceInput(BaseModel):
    sentence: str

class server:
    def run(self):
        app_logger.info('Start Serevr...')
        app = FastAPI()
        # Set up templates
        templates = Jinja2Templates(directory=f"{worksapce}/src/templates")

        @app.get("/", response_class=HTMLResponse)
        async def upload_form(request: Request):
            return templates.TemplateResponse("upload.html", {"request": request})

        @app.post("/predict")
        async def predict(sentence_input: SentenceInput):
            print("starting predict")
            try:
                input_text = f"{sentence_input.sentence.strip()}" if sentence_input.sentence.strip().endswith(".") else f"{sentence_input.sentence.strip()}."
                print(f"input text:{input_text}")
                predictor = Predict()
                model = joblib.load(f'{worksapce}/Models/logistic_model.pkl')
                sentiment = predictor.make_prediction(input_text,model)

                return {"sentiment": "POSITIVE" if sentiment == 1 else "NEGATIVE"}
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        uvicorn.run(app,host="127.0.0.1",port=8000)