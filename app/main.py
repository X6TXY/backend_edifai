from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from app.auth.router import router as auth_router
from app.config import client, env, fastapi_config
from app.helpbot.router import router as helpbot_router
from app.storybot.router import router as storybot_router
from app.wtask2.router import router as wtask2_router

app = FastAPI(**fastapi_config)


@app.on_event("shutdown")
def shutdown_db_client():
    client.close()


app.add_middleware(
    CORSMiddleware,
    allow_origins=env.CORS_ORIGINS,
    allow_methods=env.CORS_METHODS,
    allow_headers=env.CORS_HEADERS,
    allow_credentials=True,
)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth_router, prefix="/auth", tags=["Auth"])
app.include_router(wtask2_router, prefix="/wtask2", tags=["Wtask2"])
app.include_router(helpbot_router, prefix="/helpbot", tags=["Helpbot"])
app.include_router(storybot_router, prefix="/storybot", tags=["Storybot"])
