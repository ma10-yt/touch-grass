from motor.motor_asyncio import AsyncIOMotorClient
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import Optional
from app.config import settings

class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None

db_client = MongoDB()

async def connect_to_mongo():
    db_client.client = AsyncIOMotorClient(settings.MONGODB_URI)
    db_client.db = db_client.client[settings.MONGODB_DATABASE]

async def close_mongo_connection():
    if db_client.client:
        db_client.client.close()

def get_db():
    if db_client.db is None:
        raise RuntimeError("MongoDB connection is not initialized")
    return db_client.db
