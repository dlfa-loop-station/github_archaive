# main.py
from fastapi import FastAPI
from serviceRouter import router  # router를 model.py에서 가져옴
from fastapi.middleware.cors import CORSMiddleware

origins = [
   '*'
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)



@app.get("/")
def read_root():
    return {"Hello": "World"}