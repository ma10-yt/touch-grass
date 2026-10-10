from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

app = FastAPI()

class Game(BaseModel):
    id: str = Field(alias='_id')
    name: str

class GameResponse(BaseModel):
    id: str
    name: str

@app.get('/test', response_model=GameResponse)
def get_test():
    g = Game(_id='123', name='test')
    return g

client = TestClient(app)
print("Response:", client.get("/test").json())

