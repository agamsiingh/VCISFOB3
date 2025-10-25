# app.py
"""
FastAPI backend for Value Chain Integration System for Oilseed By-Products

Features:
- SQLAlchemy models (User, Device, Listing, Transaction, PriceHistory)
- JWT auth (register/login) with bcrypt password hashing (passlib)
- Marketplace endpoints (create/list/purchase)
- Device registration & telemetry (simulate and update)
- Transaction ledger: SHA256 hashed payloads for immutability
- ML forecasting endpoints: generate synthetic price history, train RandomForest, predict
- Export matchmaking (mock global buyers)
- SQLite by default; configure DATABASE_URL env var for PostgreSQL
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from Crypto.Hash import SHA256
from passlib.context import CryptContext
import jwt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# ---------------------
# Configuration
# ---------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./valuechain.db")
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey_change_me")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------
# Database (SQLAlchemy)
# ---------------------
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(20), nullable=False)  # Farmer | Processor | Buyer


class Device(Base):
    __tablename__ = 'devices'
    id = Column(Integer, primary_key=True)
    device_id = Column(String(50), unique=True, nullable=False)
    device_type = Column(String(50))
    location = Column(String(100))
    temperature = Column(Float)
    humidity = Column(Float)
    production_rate = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Listing(Base):
    __tablename__ = 'listings'
    id = Column(Integer, primary_key=True)
    seller = Column(String(100))
    product = Column(String(100))
    quantity = Column(Float)
    price_per_kg = Column(Float)
    quality_grade = Column(String(20))
    location = Column(String(100))
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.utcnow)


class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    tx_type = Column(String(50))
    seller = Column(String(100))
    buyer = Column(String(100))
    product = Column(String(100))
    quantity = Column(Float)
    price = Column(Float)
    payload = Column(Text)
    tx_hash = Column(String(128), unique=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class PriceHistory(Base):
    __tablename__ = 'price_history'
    id = Column(Integer, primary_key=True)
    product = Column(String(100))
    price = Column(Float)
    date = Column(DateTime)


# create tables
Base.metadata.create_all(engine)


# ---------------------
# Utility: DB Session context manager
# ---------------------
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------
# Security: hashing + JWT
# ---------------------
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_access_token(token)
    username = payload.get("sub")
    role = payload.get("role")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    # Return minimal user info; DB lookup optional
    return {"username": username, "role": role}


# ---------------------
# Helpers: Transaction hashing
# ---------------------
def hash_transaction(payload: dict) -> str:
    payload_str = json.dumps(payload, sort_keys=True)
    h = SHA256.new(payload_str.encode('utf-8'))
    return h.hexdigest()


def log_transaction(db: Session, tx_type: str, seller: str, buyer: str, product: str, quantity: float, price: float) -> str:
    payload = {
        'type': tx_type,
        'seller': seller,
        'buyer': buyer,
        'product': product,
        'quantity': quantity,
        'price': price,
        'timestamp': datetime.utcnow().isoformat()
    }
    tx_hash = hash_transaction(payload)
    tx = Transaction(
        tx_type=tx_type,
        seller=seller,
        buyer=buyer,
        product=product,
        quantity=quantity,
        price=price,
        payload=json.dumps(payload),
        tx_hash=tx_hash,
        timestamp=datetime.utcnow()
    )
    db.add(tx)
    db.commit()
    db.refresh(tx)
    return tx_hash


# ---------------------
# ML: synthetic data generation, train & predict
# ---------------------
def generate_synthetic_price_data(db: Session, product: str, days: int = 365) -> pd.DataFrame:
    # Check if exists; if not, generate and persist
    existing = db.query(PriceHistory).filter_by(product=product).count()
    if existing == 0:
        base_prices = {
            'Soymeal': 35,
            'Groundnut Cake': 42,
            'Sunflower Cake': 38,
            'Mustard Cake': 40,
            'Husk': 12
        }
        base_price = base_prices.get(product, 30)
        dates = [datetime.utcnow() - pd.Timedelta(days=x) for x in range(days, 0, -1)]
        trend = np.linspace(0, 8, days)
        seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, days))
        noise = np.random.normal(0, 2, days)
        prices = base_price + trend + seasonal + noise
        for d, p in zip(dates, prices):
            ph = PriceHistory(product=product, price=float(p), date=d.to_pydatetime())
            db.add(ph)
        db.commit()
    # retrieve
    rows = db.query(PriceHistory).filter_by(product=product).order_by(PriceHistory.date).all()
    df = pd.DataFrame([{'date': r.date, 'price': r.price} for r in rows])
    return df


def train_price_forecast_model(df: pd.DataFrame) -> Tuple[RandomForestRegressor, int]:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['days'] = (df['date'] - df['date'].min()).dt.days
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    X = df[['days', 'day_of_week', 'month']].values
    y = df['price'].values
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    last_day = int(df['days'].max())
    return model, last_day


def forecast_prices(model: RandomForestRegressor, last_day: int, days_ahead: int = 30) -> Tuple[List[datetime], np.ndarray]:
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1)
    future_dates = [datetime.utcnow() + timedelta(days=x) for x in range(1, days_ahead + 1)]
    X_future = []
    for i, date in enumerate(future_dates):
        X_future.append([int(future_days[i]), date.weekday(), date.month])
    X_future = np.array(X_future)
    predictions = model.predict(X_future)
    return future_dates, predictions


# ---------------------
# Pydantic Schemas
# ---------------------
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: str = Field(..., regex="^(Farmer|Processor|Buyer)$")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ListingCreate(BaseModel):
    seller: str
    product: str
    quantity: float
    price_per_kg: float
    quality_grade: str
    location: str


class ListingOut(BaseModel):
    id: int
    seller: str
    product: str
    quantity: float
    price_per_kg: float
    quality_grade: str
    location: str
    status: str
    created_at: datetime

    class Config:
        orm_mode = True


class DeviceCreate(BaseModel):
    device_id: str
    device_type: str
    location: str


class TelemetryUpdate(BaseModel):
    device_id: str
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    production_rate: Optional[float] = None


class ForecastRequest(BaseModel):
    product: str
    days_ahead: int = Field(30, ge=1, le=180)


# ---------------------
# Mock Global Buyers
# ---------------------
GLOBAL_BUYERS = [
    {'name': 'FeedCorp International', 'country': 'UAE', 'products': ['Soymeal', 'Groundnut Cake'], 'volume': 5000, 'price_range': '35-45'},
    {'name': 'AgriGlobal Ltd', 'country': 'Bangladesh', 'products': ['Mustard Cake', 'Sunflower Cake'], 'volume': 3000, 'price_range': '38-48'},
    {'name': 'EuroFeed Solutions', 'country': 'Netherlands', 'products': ['Soymeal', 'Husk'], 'volume': 10000, 'price_range': '40-50'},
    {'name': 'Asia Pacific Traders', 'country': 'Vietnam', 'products': ['Groundnut Cake', 'Soymeal'], 'volume': 7000, 'price_range': '32-42'},
    {'name': 'Middle East Feed Co', 'country': 'Saudi Arabia', 'products': ['Sunflower Cake', 'Mustard Cake'], 'volume': 4000, 'price_range': '36-46'},
    {'name': 'African AgriHub', 'country': 'Kenya', 'products': ['Soymeal', 'Groundnut Cake', 'Husk'], 'volume': 6000, 'price_range': '30-40'}
]


# ---------------------
# FastAPI app & routes
# ---------------------
app = FastAPI(title="Value Chain Integration - API", version="1.0")


@app.post("/register", response_model=dict)
def register(u: UserCreate):
    with get_db() as db:
        existing = db.query(User).filter_by(username=u.username).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")
        user = User(username=u.username, password_hash=hash_password(u.password), role=u.role)
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"username": user.username, "role": user.role}


@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    with get_db() as db:
        user = db.query(User).filter_by(username=form_data.username).first()
        if not user or not verify_password(form_data.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Incorrect credentials")
        access_token = create_access_token({"sub": user.username, "role": user.role})
        return {"access_token": access_token, "token_type": "bearer"}


@app.get("/me", response_model=dict)
def read_current_user(current_user: dict = Depends(get_current_user)):
    return current_user


# ---------------------
# Listings
# ---------------------
@app.post("/listings", response_model=ListingOut)
def create_listing(payload: ListingCreate, current_user: dict = Depends(get_current_user)):
    with get_db() as db:
        listing = Listing(
            seller=payload.seller,
            product=payload.product,
            quantity=payload.quantity,
            price_per_kg=payload.price_per_kg,
            quality_grade=payload.quality_grade,
            location=payload.location,
            status='active',
            created_at=datetime.utcnow()
        )
        db.add(listing)
        db.commit()
        db.refresh(listing)
        # Log listing as transaction
        log_transaction(db, 'listing', listing.seller, 'N/A', listing.product, listing.quantity, listing.quantity * listing.price_per_kg)
        return listing


@app.get("/listings", response_model=List[ListingOut])
def list_active_listings(skip: int = 0, limit: int = 50):
    with get_db() as db:
        rows = db.query(Listing).filter_by(status='active').order_by(Listing.created_at.desc()).offset(skip).limit(limit).all()
        return rows


@app.post("/listings/{listing_id}/purchase", response_model=dict)
def purchase_listing(listing_id: int, buyer_name: Optional[str] = "Buyer", current_user: dict = Depends(get_current_user)):
    with get_db() as db:
        listing = db.query(Listing).filter_by(id=listing_id).first()
        if not listing or listing.status != 'active':
            raise HTTPException(status_code=404, detail="Listing not available")
        total_price = listing.quantity * listing.price_per_kg
        tx_hash = log_transaction(db, 'sale', listing.seller, buyer_name, listing.product, listing.quantity, total_price)
        listing.status = 'sold'
        db.commit()
        return {"tx_hash": tx_hash, "listing_id": listing.id, "total_price": total_price}


# ---------------------
# Devices & Telemetry
# ---------------------
@app.post("/devices", response_model=dict)
def register_device(d: DeviceCreate, current_user: dict = Depends(get_current_user)):
    with get_db() as db:
        existing = db.query(Device).filter_by(device_id=d.device_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Device already registered")
        device = Device(
            device_id=d.device_id,
            device_type=d.device_type,
            location=d.location,
            temperature=0.0,
            humidity=0.0,
            production_rate=0.0,
            timestamp=datetime.utcnow()
        )
        db.add(device)
        db.commit()
        db.refresh(device)
        return {"device_id": device.device_id, "status": "registered"}


@app.post("/devices/telemetry", response_model=dict)
def update_telemetry(t: TelemetryUpdate):
    with get_db() as db:
        device = db.query(Device).filter_by(device_id=t.device_id).first()
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        if t.temperature is not None:
            device.temperature = t.temperature
        if t.humidity is not None:
            device.humidity = t.humidity
        if t.production_rate is not None:
            device.production_rate = t.production_rate
        device.timestamp = datetime.utcnow()
        db.commit()
        return {"device_id": device.device_id, "last_update": device.timestamp.isoformat()}


@app.post("/devices/simulate", response_model=dict)
def simulate_telemetry():
    # Simulate update for all devices (demo)
    with get_db() as db:
        devices = db.query(Device).all()
        if not devices:
            return {"updated": 0}
        for d in devices:
            d.temperature = float(np.random.uniform(20, 35))
            d.humidity = float(np.random.uniform(40, 70))
            d.production_rate = float(np.random.uniform(100, 500))
            d.timestamp = datetime.utcnow()
        db.commit()
        return {"updated": len(devices)}


@app.get("/devices", response_model=List[dict])
def list_devices():
    with get_db() as db:
        devs = db.query(Device).all()
        return [{
            "device_id": d.device_id,
            "type": d.device_type,
            "location": d.location,
            "temperature": d.temperature,
            "humidity": d.humidity,
            "production_rate": d.production_rate,
            "timestamp": d.timestamp
        } for d in devs]


# ---------------------
# Transactions
# ---------------------
@app.get("/transactions", response_model=List[dict])
def get_transactions(limit: int = 50):
    with get_db() as db:
        txs = db.query(Transaction).order_by(Transaction.timestamp.desc()).limit(limit).all()
        return [{
            "tx_hash": t.tx_hash,
            "type": t.tx_type,
            "seller": t.seller,
            "buyer": t.buyer,
            "product": t.product,
            "quantity": t.quantity,
            "price": t.price,
            "timestamp": t.timestamp,
            "payload": json.loads(t.payload) if t.payload else None
        } for t in txs]


@app.get("/transactions/verify/{tx_hash}", response_model=dict)
def verify_transaction(tx_hash: str):
    with get_db() as db:
        tx = db.query(Transaction).filter_by(tx_hash=tx_hash).first()
        if not tx:
            raise HTTPException(status_code=404, detail="Transaction not found")
        payload = json.loads(tx.payload)
        recomputed = hash_transaction(payload)
        ok = recomputed == tx.tx_hash
        return {"tx_hash": tx.tx_hash, "verified": ok, "payload": payload}


# ---------------------
# Forecasting: train & predict
# ---------------------
@app.post("/forecast/train", response_model=dict)
def train_forecast(req: ForecastRequest):
    with get_db() as db:
        df = generate_synthetic_price_data(db, req.product, days=365)
        model, last_day = train_price_forecast_model(df)
        model_path = os.path.join(MODEL_DIR, f"{req.product.replace(' ', '_')}_rf.joblib")
        joblib.dump({"model": model, "last_day": last_day}, model_path)
        return {"model_path": model_path, "trained_on_days": len(df)}


@app.post("/forecast/predict", response_model=dict)
def predict_forecast(req: ForecastRequest):
    model_path = os.path.join(MODEL_DIR, f"{req.product.replace(' ', '_')}_rf.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not trained. Call /forecast/train first.")
    saved = joblib.load(model_path)
    model = saved["model"]
    last_day = saved["last_day"]
    future_dates, predictions = forecast_prices(model, last_day, days_ahead=req.days_ahead)
    return {
        "product": req.product,
        "dates": [d.isoformat() for d in future_dates],
        "predictions": [float(p) for p in predictions]
    }


# ---------------------
# Export matchmaking
# ---------------------
class ExportRequest(BaseModel):
    product: str
    quantity: int
    quality: str
    location: str


@app.post("/export/match", response_model=List[dict])
def export_match(req: ExportRequest):
    matches = [b for b in GLOBAL_BUYERS if req.product in b['products']]
    # Rank by volume descending and fit within buyer volume
    matches_sorted = sorted(matches, key=lambda x: -x['volume'])
    # Provide guidance & contact stub
    out = []
    for m in matches_sorted:
        out.append({
            "name": m['name'],
            "country": m['country'],
            "products": m['products'],
            "monthly_volume": m['volume'],
            "price_range": m['price_range'],
            "contact_stub": f"contact@{m['name'].lower().replace(' ', '')}.com"
        })
    return out


# ---------------------
# Health & Info
# ---------------------
@app.get("/", response_model=dict)
def root():
    return {"status": "ok", "message": "Value Chain Integration API", "version": "1.0"}


# ---------------------
# Utility: create demo accounts (optional)
# ---------------------
@app.post("/demo/create_accounts", response_model=dict)
def create_demo_accounts():
    with get_db() as db:
        created = []
        for uname, role in [("demo_farmer", "Farmer"), ("demo_processor", "Processor"), ("demo_buyer", "Buyer")]:
            if not db.query(User).filter_by(username=uname).first():
                u = User(username=uname, password_hash=hash_password("password"), role=role)
                db.add(u)
                created.append(uname)
        db.commit()
    return {"created": created}
