"""
Value Chain Integration System for Oilseed By-Products
A production-ready AgriTech platform for marketplace, IoT, blockchain-style transactions, and AI forecasting.
"""

def render_live_food_dashboard():
    st.header("Live Food Data Dashboard")
    
    # Add auto-refresh button
    if st.button("Refresh Data"):
        update_food_data()
    
    # Get latest data
    session = Session()
    food_items = session.query(FoodItem).all()
    
    # Create DataFrames for visualization
    df = pd.DataFrame([{
        'Name': item.name,
        'Category': item.category,
        'Quantity': item.quantity,
        'Unit': item.unit,
        'Price/Unit': item.price_per_unit,
        'Quality Score': item.quality_score,
        'Temperature': item.storage_temperature,
        'Humidity': item.humidity_level,
        'Location': item.location,
        'Status': item.status,
        'Last Updated': item.last_updated
    } for item in food_items])
    
    # Display current inventory
    st.subheader("Current Inventory")
    st.dataframe(df)
    
    # Quality Monitoring
    st.subheader("Quality Monitoring")
    fig = px.scatter(df, 
                    x='Temperature', 
                    y='Humidity',
                    size='Quality Score',
                    color='Category',
                    hover_name='Name',
                    title='Storage Conditions vs Quality Score')
    st.plotly_chart(fig)
    
    # Temperature and Humidity Gauges
    st.subheader("Environmental Conditions")
    col1, col2 = st.columns(2)
    
    with col1:
        avg_temp = df['Temperature'].mean()
        fig_temp = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_temp,
            title={'text': "Average Temperature (°C)"},
            gauge={'axis': {'range': [0, 40]},
                  'steps': [
                      {'range': [0, 15], 'color': "lightblue"},
                      {'range': [15, 25], 'color': "lightgreen"},
                      {'range': [25, 40], 'color': "red"}
                  ],
                  'threshold': {
                      'line': {'color': "red", 'width': 4},
                      'thickness': 0.75,
                      'value': 25}}))
        st.plotly_chart(fig_temp)
    
    with col2:
        avg_humidity = df['Humidity'].mean()
        fig_humidity = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_humidity,
            title={'text': "Average Humidity (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 40], 'color': "lightblue"},
                    {'range': [40, 60], 'color': "lightgreen"},
                    {'range': [60, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        st.plotly_chart(fig_humidity)
    
    session.close()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from Crypto.Hash import SHA256
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import json
import time
from random import uniform
import os
import requests
try:
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
except Exception:
    InstalledAppFlow = None
    id_token = None
    google_requests = None
    
# Commodities helper (yfinance-based). Implemented in services/commodities.py
try:
    from services.commodities import get_commodities_data, DEFAULT_TICKER_MAP
except Exception:
    # If import fails (module missing), set defaults so the app still runs.
    def get_commodities_data(names, period='90d', interval='1d'):
        return {n: {'ticker': None, 'current': None, 'history': None} for n in names}
    DEFAULT_TICKER_MAP = {}

# Database Setup
Base = declarative_base()
engine = create_engine('sqlite:///valuechain.db', echo=False)
Session = sessionmaker(bind=engine)

def init_database():
    """Initialize database and create all tables"""
    # Create all tables
    Base.metadata.create_all(engine)
    # Run lightweight schema migrations to add any new columns to existing tables
    run_schema_migrations()
    
    session = Session()
    
    try:
        # Check if we need to create sample data
        if session.query(FoodItem).count() == 0:
            create_sample_food_data()

        # Ensure demo users exist for quick role-based access (safe: create_user is idempotent)
        try:
            create_user('demo_farmer', 'farmer', 'Farmer')
            create_user('demo_processor', 'processor', 'Processor')
            create_user('demo_buyer', 'buyer', 'Buyer')
            create_user('demo_govt', 'govt', 'Govt')
        except Exception:
            # Non-fatal: continue if user creation fails (DB might be locked during tests)
            pass
        
        # Commit any pending transactions
        session.commit()
    except Exception as e:
        session.rollback()
        st.error(f"Database initialization error: {str(e)}")
    finally:
        session.close()

# ============================================================================
# DATABASE MODELS
# ============================================================================

class FoodItem(Base):
    """Food items with real-time tracking"""
    __tablename__ = 'food_items'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    category = Column(String(50))
    quantity = Column(Float)
    unit = Column(String(20))
    price_per_unit = Column(Float)
    quality_score = Column(Float)
    expiry_date = Column(DateTime)
    storage_temperature = Column(Float)
    humidity_level = Column(Float)
    last_updated = Column(DateTime, default=datetime.now)
    location = Column(String(100))
    status = Column(String(50))  # e.g., 'in_storage', 'in_transit', 'delivered'

# Create sample food data
def create_sample_food_data():
    session = Session()
    food_items = [
        FoodItem(
            name='Soybean Oil',
            category='Oils',
            quantity=1000.0,
            unit='liters',
            price_per_unit=2.5,
            quality_score=95.0,
            expiry_date=datetime.now() + timedelta(days=180),
            storage_temperature=20.0,
            humidity_level=45.0,
            location='Warehouse A',
            status='in_storage'
        ),
        FoodItem(
            name='Canola Oil',
            category='Oils',
            quantity=800.0,
            unit='liters',
            price_per_unit=3.0,
            quality_score=92.0,
            expiry_date=datetime.now() + timedelta(days=180),
            storage_temperature=20.0,
            humidity_level=45.0,
            location='Warehouse B',
            status='in_storage'
        ),
        FoodItem(
            name='Sunflower Seeds',
            category='Raw Materials',
            quantity=2000.0,
            unit='kg',
            price_per_unit=1.5,
            quality_score=98.0,
            expiry_date=datetime.now() + timedelta(days=365),
            storage_temperature=18.0,
            humidity_level=40.0,
            location='Silo C',
            status='in_storage'
        )
    ]
    
    for item in food_items:
        if not session.query(FoodItem).filter_by(name=item.name).first():
            session.add(item)
    
    session.commit()
    session.close()


def run_schema_migrations():
    """Run simple, idempotent schema migrations for SQLite by adding missing columns.

    This avoids requiring users to delete their database when we add new optional
    columns to models during development.
    """
    try:
        with engine.connect() as conn:
            # Inspect current columns for devices
            res = conn.execute(text("PRAGMA table_info('devices')")).fetchall()
            existing_cols = {row[1] for row in res}  # second field is name

            # Desired columns added in development
            migrations = {
                'quality_metrics': "TEXT",
                'maintenance_status': "VARCHAR(50)",
                'last_maintenance': "DATETIME",
                'firmware_version': "VARCHAR(20)",
                'battery_level': "FLOAT",
                'signal_strength': "FLOAT",
                'alert_status': "VARCHAR(50)",
                'api_key': "VARCHAR(64)"
            }

            for col, col_type in migrations.items():
                if col not in existing_cols:
                    sql = f"ALTER TABLE devices ADD COLUMN {col} {col_type}"
                    try:
                        conn.execute(text(sql))
                    except Exception:
                        # Best-effort: if migration fails, continue — app can still run
                        pass
    except Exception:
        # If migrations can't run (e.g., DB locked), ignore and proceed — user can recreate DB
        pass

# ============================================================================
# IOT INTEGRATION AND API FUNCTIONS
# ============================================================================

def generate_api_key():
    """Generate a secure API key for IoT device authentication"""
    return SHA256.new(f"{datetime.now()}:{np.random.rand()}".encode()).hexdigest()

def validate_api_key(api_key):
    """Validate device API key"""
    session = Session()
    device = session.query(Device).filter_by(api_key=api_key).first()
    session.close()
    return device is not None

def process_telemetry_data(device_id, data):
    """Process and validate incoming telemetry data"""
    try:
        # Basic data validation
        required_fields = ['temperature', 'humidity', 'production_rate']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Process quality metrics
        quality_metrics = data.get('quality_metrics', {})
        if quality_metrics:
            try:
                # Validate quality metrics format
                json.dumps(quality_metrics)
            except:
                return False, "Invalid quality metrics format"
        
        # Check for alert conditions
        alert_triggered = False
        alert_message = []
        
        if data['temperature'] > 35:
            alert_triggered = True
            alert_message.append("High temperature alert")
        if data['humidity'] > 70:
            alert_triggered = True
            alert_message.append("High humidity alert")
        
        return True, {
            'device_id': device_id,
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'production_rate': float(data['production_rate']),
            'quality_metrics': json.dumps(quality_metrics),
            'alert_triggered': alert_triggered,
            'alert_message': '; '.join(alert_message) if alert_message else None,
            'raw_data': json.dumps(data)
        }
    except Exception as e:
        return False, f"Data processing error: {str(e)}"

def ingest_telemetry(api_key, data):
    """Ingest telemetry data from IoT devices"""
    session = Session()
    try:
        # Validate API key
        device = session.query(Device).filter_by(api_key=api_key).first()
        if not device:
            return False, "Invalid API key"
        
        # Process telemetry data
        success, result = process_telemetry_data(device.device_id, data)
        if not success:
            return False, result
        
        # Update device current state
        device.temperature = result['temperature']
        device.humidity = result['humidity']
        device.production_rate = result['production_rate']
        device.quality_metrics = result['quality_metrics']
        device.alert_status = 'alert' if result['alert_triggered'] else 'normal'
        device.timestamp = datetime.now()
        
        # Store historical telemetry
        telemetry = DeviceTelemetry(**result)
        session.add(telemetry)
        
        # Update food items in storage if applicable
        if device.device_type == 'storage_monitor':
            update_storage_conditions(device.location, result)
        
        session.commit()
        return True, "Data ingested successfully"
    
    except Exception as e:
        session.rollback()
        return False, f"Error ingesting data: {str(e)}"
    finally:
        session.close()

def update_storage_conditions(location, telemetry_data):
    """Update food storage conditions based on telemetry"""
    session = Session()
    try:
        items = session.query(FoodItem).filter_by(location=location).all()
        for item in items:
            item.storage_temperature = telemetry_data['temperature']
            item.humidity_level = telemetry_data['humidity']
            
            # Update quality score based on conditions
            if item.storage_temperature > 25 or item.humidity_level > 60:
                item.quality_score = max(0, item.quality_score - uniform(0.1, 0.3))
            
            item.last_updated = datetime.now()
        session.commit()
    except:
        session.rollback()
    finally:
        session.close()

def get_device_stats(device_id, hours=24):
    """Get device statistics for the specified time period"""
    session = Session()
    try:
        # Get telemetry data for the specified period
        since = datetime.now() - timedelta(hours=hours)
        telemetry = session.query(DeviceTelemetry).filter(
            DeviceTelemetry.device_id == device_id,
            DeviceTelemetry.timestamp >= since
        ).order_by(DeviceTelemetry.timestamp.asc()).all()
        
        if not telemetry:
            return None
        
        # Calculate statistics
        temps = [t.temperature for t in telemetry]
        humid = [t.humidity for t in telemetry]
        prod = [t.production_rate for t in telemetry]
        
        stats = {
            'temperature': {
                'current': temps[-1],
                'min': min(temps),
                'max': max(temps),
                'avg': sum(temps) / len(temps)
            },
            'humidity': {
                'current': humid[-1],
                'min': min(humid),
                'max': max(humid),
                'avg': sum(humid) / len(humid)
            },
            'production_rate': {
                'current': prod[-1],
                'total': sum(prod),
                'avg': sum(prod) / len(prod)
            },
            'alerts': len([t for t in telemetry if t.alert_triggered]),
            'data_points': len(telemetry),
            'timestamps': [t.timestamp for t in telemetry]
        }
        
        return stats
    finally:
        session.close()

# Function to update food data with simulated changes
def update_food_data():
    session = Session()
    items = session.query(FoodItem).all()
    
    for item in items:
        # Simulate small changes in temperature and humidity
        item.storage_temperature += uniform(-0.5, 0.5)
        item.humidity_level += uniform(-1, 1)
        item.humidity_level = max(30, min(70, item.humidity_level))
        
        # Update quality score based on conditions
        if item.storage_temperature > 25 or item.humidity_level > 60:
            item.quality_score = max(0, item.quality_score - uniform(0.1, 0.3))
        
        item.last_updated = datetime.now()
    
    session.commit()
    session.close()

# ============================================================================
# DATABASE MODELS
# ============================================================================

class Device(Base):
    """IoT device registry and telemetry storage"""
    __tablename__ = 'devices'
    id = Column(Integer, primary_key=True)
    device_id = Column(String(50), unique=True, nullable=False)
    device_type = Column(String(50))
    location = Column(String(100))
    temperature = Column(Float)
    humidity = Column(Float)
    production_rate = Column(Float)
    quality_metrics = Column(Text)  # JSON field for quality parameters
    maintenance_status = Column(String(50))
    last_maintenance = Column(DateTime)
    firmware_version = Column(String(20))
    battery_level = Column(Float)
    signal_strength = Column(Float)
    alert_status = Column(String(50))
    timestamp = Column(DateTime, default=datetime.now)
    api_key = Column(String(64), unique=True)  # For device authentication

class DeviceTelemetry(Base):
    """Historical telemetry data for IoT devices"""
    __tablename__ = 'device_telemetry'
    id = Column(Integer, primary_key=True)
    device_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    temperature = Column(Float)
    humidity = Column(Float)
    production_rate = Column(Float)
    quality_metrics = Column(Text)  # JSON field for quality data
    alert_triggered = Column(Boolean, default=False)
    alert_message = Column(Text)
    raw_data = Column(Text)  # JSON field for additional sensor data

class Transaction(Base):
    """Blockchain-style transaction ledger with SHA256 hashing"""
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    tx_type = Column(String(50))
    seller = Column(String(100))
    buyer = Column(String(100))
    product = Column(String(100))
    quantity = Column(Float)
    price = Column(Float)
    payload = Column(Text)
    tx_hash = Column(String(64), unique=True)
    timestamp = Column(DateTime, default=datetime.now)


class Listing(Base):
    """Marketplace product listings"""
    __tablename__ = 'listings'
    id = Column(Integer, primary_key=True)
    seller = Column(String(100))
    product = Column(String(100))
    quantity = Column(Float)
    price_per_kg = Column(Float)
    quality_grade = Column(String(20))
    location = Column(String(100))
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.now)


class PriceHistory(Base):
    """Historical price data for ML forecasting"""
    __tablename__ = 'price_history'
    id = Column(Integer, primary_key=True)
    product = Column(String(100))
    price = Column(Float)
    date = Column(DateTime)


# ---------------------------------
# User model for demo auth & roles
# ---------------------------------
class User(Base):
    """Simple user model for demo login/registration"""
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    role = Column(String(20), nullable=False)  # Farmer | Processor | Buyer

# Tables are created by init_database()


def hash_password(password: str) -> str:
    """Return SHA256 hex digest of password (demo only)."""
    return SHA256.new(password.encode('utf-8')).hexdigest()


def create_user(username: str, password: str, role: str) -> bool:
    session = Session()
    existing = session.query(User).filter_by(username=username).first()
    if existing:
        session.close()
        return False
    user = User(username=username, password_hash=hash_password(password), role=role)
    session.add(user)
    session.commit()
    session.close()
    return True


def authenticate_user(username: str, password: str):
    """Authenticate a user by username and password.

    For developer convenience this will auto-create known demo accounts if they
    don't exist yet (idempotent). Returns a dict {'username', 'role'} on
    success, or None on failure.
    """
    demo_accounts = {
        'demo_farmer': ('farmer', 'Farmer'),
        'demo_processor': ('processor', 'Processor'),
        'demo_buyer': ('buyer', 'Buyer'),
        'demo_govt': ('govt', 'Govt'),
    }

    from sqlalchemy import or_
    session = Session()
    # Try exact match first, then case-insensitive match
    user = session.query(User).filter_by(username=username).first()
    if not user:
        try:
            user = session.query(User).filter(User.username.ilike(username)).first()
        except Exception:
            # ilike may not be available on some DBs; ignore and continue
            user = None

    # If user not found but it's a known demo account, create it and re-query
    if not user and username in demo_accounts:
        try:
            pwd, role = demo_accounts[username]
            created = create_user(username, pwd, role)
            if created:
                # re-open session/query to get the newly created user
                session.close()
                session = Session()
                user = session.query(User).filter_by(username=username).first()
        except Exception:
            # ignore and continue — authentication will fail below
            user = None

    if not user:
        session.close()
        return None

    # compare password hashes
    if user.password_hash == hash_password(password):
        role = user.role if user.role else 'Farmer'
        session.close()
        return {'username': user.username, 'role': role}

    # If password mismatch but this is a known demo account, reset password to demo default
    demo_accounts = {
        'demo_farmer': ('farmer', 'Farmer'),
        'demo_processor': ('processor', 'Processor'),
        'demo_buyer': ('buyer', 'Buyer'),
        'demo_govt': ('govt', 'Govt'),
    }
    uname_lower = (user.username or '').lower()
    if uname_lower in demo_accounts:
        demo_pwd, demo_role = demo_accounts[uname_lower]
        try:
            user.password_hash = hash_password(demo_pwd)
            user.role = demo_role
            session.add(user)
            session.commit()
            # after reset, check again
            if hash_password(password) == user.password_hash:
                session.close()
                return {'username': user.username, 'role': user.role}
        except Exception:
            session.rollback()

    session.close()
    return None


# ============================================================================
# BLOCKCHAIN AND SMART CONTRACTS
# ============================================================================

class SmartContract(Base):
    """Smart contract for automated trade execution"""
    __tablename__ = 'smart_contracts'
    id = Column(Integer, primary_key=True)
    contract_hash = Column(String(64), unique=True)
    contract_type = Column(String(50))  # 'forward', 'spot', 'option'
    seller = Column(String(100))
    buyer = Column(String(100))
    product = Column(String(100))
    quantity = Column(Float)
    price_per_unit = Column(Float)
    quality_requirements = Column(Text)
    delivery_date = Column(DateTime)
    payment_terms = Column(Text)
    status = Column(String(50))  # 'pending', 'active', 'completed', 'cancelled'
    conditions_hash = Column(String(64))
    signatures = Column(Text)  # JSON storing multiple signatures
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

def hash_transaction(payload):
    """Generate SHA256 hash for transaction payload"""
    payload_str = json.dumps(payload, sort_keys=True)
    hash_obj = SHA256.new(payload_str.encode('utf-8'))
    return hash_obj.hexdigest()

def create_smart_contract(contract_type, seller, buyer, product, quantity, price_per_unit, 
                         quality_requirements, delivery_date, payment_terms):
    """Create a new smart contract with specified terms"""
    session = Session()
    
    # Create contract conditions
    conditions = {
        'contract_type': contract_type,
        'seller': seller,
        'buyer': buyer,
        'product': product,
        'quantity': quantity,
        'price_per_unit': price_per_unit,
        'quality_requirements': quality_requirements,
        'delivery_date': delivery_date.isoformat() if delivery_date else None,
        'payment_terms': payment_terms,
        'timestamp': datetime.now().isoformat()
    }
    
    # Hash the conditions
    conditions_hash = hash_transaction(conditions)
    
    # Create contract record
    contract = SmartContract(
        contract_hash=hash_transaction(conditions),
        contract_type=contract_type,
        seller=seller,
        buyer=buyer,
        product=product,
        quantity=quantity,
        price_per_unit=price_per_unit,
        quality_requirements=json.dumps(quality_requirements),
        delivery_date=delivery_date,
        payment_terms=json.dumps(payment_terms),
        status='pending',
        conditions_hash=conditions_hash,
        signatures=json.dumps({})
    )
    
    session.add(contract)
    session.commit()
    session.refresh(contract)
    contract_id = contract.id
    session.close()
    
    return contract_id, conditions_hash

def sign_smart_contract(contract_id, signer, signature_type='approve'):
    """Sign a smart contract (simulated digital signature)"""
    session = Session()
    contract = session.query(SmartContract).filter_by(id=contract_id).first()
    
    if not contract:
        session.close()
        return False, "Contract not found"
    
    # Get current signatures
    signatures = json.loads(contract.signatures)
    
    # Add new signature
    signatures[signer] = {
        'type': signature_type,
        'timestamp': datetime.now().isoformat(),
        'hash': hash_transaction(f"{contract.conditions_hash}:{signer}:{datetime.now().isoformat()}")
    }
    
    # Update contract
    contract.signatures = json.dumps(signatures)
    
    # If both parties have signed, activate the contract
    if len(signatures) == 2:  # Both parties signed
        contract.status = 'active'
    
    session.commit()
    session.close()
    
    return True, "Signature added successfully"

def execute_smart_contract(contract_id):
    """Execute a smart contract and record the transaction"""
    session = Session()
    contract = session.query(SmartContract).filter_by(id=contract_id).first()
    
    if not contract or contract.status != 'active':
        session.close()
        return False, "Contract not found or not active"
    
    try:
        # Create transaction record
        payload = {
            'type': 'smart_contract_execution',
            'contract_hash': contract.contract_hash,
            'seller': contract.seller,
            'buyer': contract.buyer,
            'product': contract.product,
            'quantity': contract.quantity,
            'price': contract.quantity * contract.price_per_unit,
            'timestamp': datetime.now().isoformat()
        }
        
        tx_hash = hash_transaction(payload)
        
        # Record transaction
        tx = Transaction(
            tx_type='smart_contract_execution',
            seller=contract.seller,
            buyer=contract.buyer,
            product=contract.product,
            quantity=contract.quantity,
            price=contract.quantity * contract.price_per_unit,
            payload=json.dumps(payload),
            tx_hash=tx_hash
        )
        
        # Update contract status
        contract.status = 'completed'
        
        session.add(tx)
        session.commit()
        session.close()
        
        return True, tx_hash
    
    except Exception as e:
        session.rollback()
        session.close()
        return False, str(e)

def verify_contract(contract_hash):
    """Verify contract authenticity and signatures"""
    session = Session()
    contract = session.query(SmartContract).filter_by(contract_hash=contract_hash).first()
    
    if not contract:
        session.close()
        return None
    
    # Verify conditions hash
    conditions = {
        'contract_type': contract.contract_type,
        'seller': contract.seller,
        'buyer': contract.buyer,
        'product': contract.product,
        'quantity': contract.quantity,
        'price_per_unit': contract.price_per_unit,
        'quality_requirements': json.loads(contract.quality_requirements),
        'delivery_date': contract.delivery_date.isoformat() if contract.delivery_date else None,
        'payment_terms': json.loads(contract.payment_terms),
        'timestamp': contract.created_at.isoformat()
    }
    
    computed_hash = hash_transaction(conditions)
    hash_valid = computed_hash == contract.conditions_hash
    
    # Verify signatures
    signatures = json.loads(contract.signatures)
    
    verification = {
        'contract_valid': hash_valid,
        'conditions_hash': contract.conditions_hash,
        'computed_hash': computed_hash,
        'signatures': signatures,
        'status': contract.status,
        'created_at': contract.created_at,
        'updated_at': contract.updated_at
    }
    
    session.close()
    return verification

def log_transaction(tx_type, seller, buyer, product, quantity, price):
    """Record a transaction in the blockchain-style ledger"""
    session = Session()
    payload = {
        'type': tx_type,
        'seller': seller,
        'buyer': buyer,
        'product': product,
        'quantity': quantity,
        'price': price,
        'timestamp': datetime.now().isoformat()
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
        tx_hash=tx_hash
    )
    session.add(tx)
    session.commit()
    session.close()
    return tx_hash


# ============================================================================
# HELPER FUNCTIONS - DATA GENERATION
# ============================================================================

def generate_synthetic_price_data(product, days=365):
    """
    Generate synthetic historical price data with trend and noise for ML training.
    
    Args:
        product (str): Product name
        days (int): Number of historical days
    
    Returns:
        pd.DataFrame: Historical price data
    """
    session = Session()
    # Check if data already exists
    existing = session.query(PriceHistory).filter_by(product=product).count()
    
    if existing == 0:
        base_prices = {
            'Soymeal': 35,
            'Groundnut Cake': 42,
            'Sunflower Cake': 38,
            'Mustard Cake': 40,
            'Husk': 12
        }
        base_price = base_prices.get(product, 30)
        
        dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
        trend = np.linspace(0, 8, days)
        seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, days))
        noise = np.random.normal(0, 2, days)
        prices = base_price + trend + seasonal + noise
        
        for date, price in zip(dates, prices):
            ph = PriceHistory(product=product, price=price, date=date)
            session.add(ph)
        session.commit()
    
    # Retrieve data
    records = session.query(PriceHistory).filter_by(product=product).order_by(PriceHistory.date).all()
    session.close()
    
    df = pd.DataFrame([{
        'date': r.date,
        'price': r.price
    } for r in records])
    
    return df


def generate_iot_telemetry():
    """
    Simulate IoT device telemetry data for dashboard visualization.
    
    Returns:
        dict: Simulated sensor readings
    """
    return {
        'temperature': round(np.random.uniform(20, 35), 2),
        'humidity': round(np.random.uniform(40, 70), 2),
        'production_rate': round(np.random.uniform(100, 500), 2)
    }


# ============================================================================
# MACHINE LEARNING - ADVANCED ANALYTICS
# ============================================================================

def calculate_volatility(prices, window=30):
    """Calculate rolling volatility of prices"""
    returns = pd.Series(prices).pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    return volatility

def detect_seasonality(dates, prices):
    """Detect seasonal patterns in price data"""
    df = pd.DataFrame({'date': dates, 'price': prices})
    df['month'] = pd.to_datetime(df['date']).dt.month
    monthly_avg = df.groupby('month')['price'].mean()
    return monthly_avg

def train_price_forecast_model(df):
    """
    Train advanced ML models for price and demand forecasting
    
    Args:
        df (pd.DataFrame): Historical price data with 'date' and 'price' columns
    
    Returns:
        tuple: (price_model, demand_model, volatility_model, feature_data)
    """
    df = df.copy()
    # Feature engineering
    df['days'] = (df['date'] - df['date'].min()).dt.days
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = pd.to_datetime(df['date']).dt.quarter
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['volatility'] = calculate_volatility(df['price'])
    
    # Add lagged features
    df['price_lag1'] = df['price'].shift(1)
    df['price_lag7'] = df['price'].shift(7)
    df['price_lag30'] = df['price'].shift(30)
    df['rolling_mean_7'] = df['price'].rolling(window=7).mean()
    df['rolling_mean_30'] = df['price'].rolling(window=30).mean()
    
    # Drop NaN values from lagged features
    df = df.dropna()
    
    # Prepare features
    feature_cols = ['days', 'day_of_week', 'month', 'quarter', 'price_lag1', 
                   'price_lag7', 'price_lag30', 'rolling_mean_7', 'rolling_mean_30']
    X = df[feature_cols].values
    
    # Train price model
    price_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    price_model.fit(X, df['price'].values)
    
    # Train demand model (simulated demand as function of price and seasonality)
    df['demand'] = 1000 + (-0.5 * df['price']) + (100 * np.sin(2 * np.pi * df['month'] / 12))
    demand_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    demand_model.fit(X, df['demand'].values)
    
    # Train volatility model
    volatility_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    volatility_model.fit(X, df['volatility'].fillna(0).values)
    
    return price_model, demand_model, volatility_model, {
        'last_day': df['days'].max(),
        'feature_cols': feature_cols,
        'last_values': df.iloc[-1][['price', 'price_lag1', 'price_lag7', 'price_lag30', 
                                   'rolling_mean_7', 'rolling_mean_30']].to_dict()
    }

def forecast_market_metrics(price_model, demand_model, volatility_model, feature_data, days_ahead=30):
    """
    Generate comprehensive market forecasts including price, demand, and volatility
    
    Returns:
        dict: Forecasted metrics
    """
    future_days = np.arange(feature_data['last_day'] + 1, 
                           feature_data['last_day'] + days_ahead + 1)
    future_dates = [datetime.now() + timedelta(days=x) for x in range(1, days_ahead + 1)]
    
    # Initialize predictions
    price_predictions = []
    demand_predictions = []
    volatility_predictions = []
    
    # Current values for lagged features
    current_price = feature_data['last_values']['price']
    current_lag1 = feature_data['last_values']['price_lag1']
    current_lag7 = feature_data['last_values']['price_lag7']
    current_lag30 = feature_data['last_values']['price_lag30']
    current_roll7 = feature_data['last_values']['rolling_mean_7']
    current_roll30 = feature_data['last_values']['rolling_mean_30']
    
    # Generate predictions day by day
    for i, date in enumerate(future_dates):
        # Prepare features
        X = np.array([[
            future_days[i],
            date.weekday(),
            date.month,
            (date.month - 1) // 3 + 1,  # quarter
            current_price,
            current_lag7,
            current_lag30,
            current_roll7,
            current_roll30
        ]])
        
        # Predict
        price = price_model.predict(X)[0]
        demand = demand_model.predict(X)[0]
        volatility = volatility_model.predict(X)[0]
        
        # Store predictions
        price_predictions.append(price)
        demand_predictions.append(demand)
        volatility_predictions.append(volatility)
        
        # Update lagged values for next iteration
        current_lag30 = current_lag7
        current_lag7 = current_price
        current_price = price
        current_roll7 = np.mean(price_predictions[-7:] if len(price_predictions) >= 7 else price_predictions)
        current_roll30 = np.mean(price_predictions[-30:] if len(price_predictions) >= 30 else price_predictions)
    
    return {
        'dates': future_dates,
        'prices': price_predictions,
        'demand': demand_predictions,
        'volatility': volatility_predictions
    }


# ============================================================================
# STREAMLIT UI CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AgriTech Value Chain Integration",
    page_icon="🌾",
    layout="wide"
)

# Initialize database
init_database()


# Compatibility helper: safe rerun for Streamlit versions without experimental_rerun
def safe_rerun():
    """Call Streamlit's rerun API when available; otherwise no-op.

    This helper is defined early so UI elements that are rendered before the
    bottom-of-file helpers can still call it (prevents NameError when the
    login overlay calls a rerun before the later definition).
    """
    if hasattr(st, 'experimental_rerun') and callable(getattr(st, 'experimental_rerun')):
        try:
            st.experimental_rerun()
        except Exception:
            # Best-effort: if calling fails, ignore — Streamlit will rerun on next interaction
            pass

# -----------------------
# Full-screen Login UI
# -----------------------
SCOPES = [
        'openid',
        'https://www.googleapis.com/auth/userinfo.email',
        'https://www.googleapis.com/auth/userinfo.profile'
]


def google_oauth_flow():
        """Run OAuth2 flow (local server) and return userinfo dict.

        Requires a `client_secrets.json` file in the project root containing
        Google OAuth 2.0 credentials (type: Desktop or Web). For local dev,
        Desktop credentials with `run_local_server` work well.
        """
        if InstalledAppFlow is None:
                st.error("Google OAuth libraries are not installed. Add `google-auth-oauthlib` to your environment.")
                return None

        creds_file = os.path.join(os.getcwd(), 'client_secrets.json')
        if not os.path.exists(creds_file):
                st.error("`client_secrets.json` not found. Create OAuth credentials in Google Cloud Console and save the JSON as `client_secrets.json` in the project root.")
                return None

        flow = InstalledAppFlow.from_client_secrets_file(creds_file, scopes=SCOPES)
        creds = flow.run_local_server(port=0)

        # Fetch userinfo
        try:
                resp = requests.get('https://www.googleapis.com/oauth2/v1/userinfo', params={'alt': 'json'}, headers={'Authorization': f'Bearer {creds.token}'})
                if resp.status_code == 200:
                        return resp.json()
        except Exception:
                pass
        return None


def render_fullscreen_login():
    """Render a full-screen, centered login page styled like the provided design.

    Features:
    - Email + Password fields
    - Role selector
    - Forgot password link
    - Large green submit button
    - Optional Google sign-in (keeps existing behavior)
    """
    # Inject CSS to make the login card full-screen, centered and dark themed
    css = """
    <style>
    /* App background and base text color for the login screen */
    .stApp { background: #000000 !important; color: #ffffff !important; }
    /* Reduce Streamlit's default top padding so content sits higher on the page */
    .block-container, .reportview-container .main .block-container, .main, .stApp>div {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    /* Card uses dark translucent surface so it reads on black background (keeps widgets in Streamlit flow so they remain interactive) */
    .login-card { background: rgba(18,18,18,0.96); border-radius:12px; padding:28px; width:560px; margin:48px auto; box-shadow:0 20px 60px rgba(0,0,0,0.8); }
    .login-card h1 { text-align:center; margin-bottom:18px; font-weight:700; color:#ffffff; }
    /* Inputs: dark field with light text (best-effort across Streamlit versions) */
    .login-input input, .login-select select, .stTextInput>div>input, .stTextInput>div>textarea {
        height:44px; border-radius:6px; border:1px solid rgba(255,255,255,0.06); padding:10px 14px; width:100%; background:#0b0b0b; color:#e6e6e6;
    }
    .login-select select { padding:8px 12px; }
    .login-forgot { text-align:left; margin-top:8px; }
    .login-forgot a { color:#93c5fd; }
    .submit-btn button { background:#2f9e44; color:white; border-radius:8px; padding:12px 18px; width:100%; border: none; font-size:16px; }
    .submit-btn button:hover { opacity:0.95; }
    .google-btn { background:#111827; color:#fff; padding:10px 14px; border-radius:6px; border:1px solid rgba(255,255,255,0.04); width:100%; }
    /* Ensure Streamlit buttons inside card match */
    .login-card .stButton>button { background: #2f9e44 !important; color: white !important; border-radius: 8px !important; padding: 12px 18px !important; width: 100% !important; font-size: 16px !important; border: none !important; }
    /* Links and small text inside card should be readable on dark bg */
    .login-card a, .login-card label, .login-card .stMarkdown, .login-card .stText {
        color: #e6e6e6 !important;
    }
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

    # If there is a `video/` folder with videos, inject a full-screen background
    # video player that cycles through the files. Videos are muted and set to
    # autoplay; JS switches the src when each video ends so we can rotate files.
    try:
        video_dir = os.path.join(os.getcwd(), 'video')
        video_files = []
        if os.path.exists(video_dir) and os.path.isdir(video_dir):
            for fname in sorted(os.listdir(video_dir)):
                if fname.lower().endswith(('.mp4', '.webm', '.ogg')):
                    # Use a web-relative path so browser can load from local folder
                    rel = os.path.join('video', fname).replace('\\', '/')
                    video_files.append(rel)

        if video_files:
            # Natural sort so files named 1,2,3,4 order correctly even if names vary
            import re
            def _natural_key(s):
                parts = re.split(r'(\d+)', s)
                return [int(p) if p.isdigit() else p.lower() for p in parts]
            video_files = sorted(video_files, key=_natural_key)

            # Build JSON list and video HTML safely (avoid f-string brace interpolation)
            video_list_json = json.dumps(video_files)
            # Use <source> elements dynamically for cross-browser fallback and a small SVG poster as fallback image
            video_html = (
                """
            <style>
            /* Background video sits behind everything; keep it clickable-disabled */
            .bg-video { position: fixed; top:0; left:0; width:100%; height:100%; object-fit:cover; z-index: 0; pointer-events:none; filter: brightness(0.42) saturate(0.95); }
            .bg-video-overlay { position: fixed; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.36); z-index: 1; pointer-events:none; }
            /* Ensure login card sits above video */
            .login-card { position: relative; z-index: 9999; pointer-events: auto; }
            </style>
            <!-- Video element: muted + autoplay required by modern browsers to allow autoplay. We set src directly for reliability. -->
            <video id="bgVideo" class="bg-video" autoplay muted playsinline loop preload="metadata" poster="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='1600' height='900'><rect width='100%' height='100%' fill='%23000000'/></svg>"></video>
            <div class="bg-video-overlay"></div>
            <script>
                const _videos = """ + video_list_json + """;
                let _i = 0;
                const _v = document.getElementById('bgVideo');
                function _setSrc(url){
                    if(!_v) return;
                    try{
                        _v.pause();
                        _v.muted = true;
                        _v.playsInline = true;
                        _v.autoplay = true;
                        // Assign src directly for better cross-browser reliability
                        _v.src = url;
                        _v.load();
                        _v.play().catch(()=>{});
                    }catch(e){}
                }
                try{
                    const list = Array.isArray(_videos) ? _videos : JSON.parse(_videos);
                    if(list && list.length>0){
                        // Use metadata-only preload to avoid large initial downloads
                        _v.preload = 'metadata';
                        _setSrc(list[0]);
                        _v.addEventListener('ended', ()=>{ _i = (_i + 1) % list.length; _setSrc(list[_i]); });
                        _v.addEventListener('error', ()=>{ _i = (_i + 1) % list.length; _setSrc(list[_i]); });
                    }
                }catch(e){
                    try{ _setSrc(Array.isArray(_videos) ? _videos[0] : JSON.parse(_videos)[0]); }catch(_){ }
                }
            </script>
            """
            )
            st.markdown(video_html, unsafe_allow_html=True)
        else:
            # No local videos found. Offer uploader AND fall back to public demo videos
            try:
                st.markdown("<div style='text-align:center;color:#bfc7d6;margin-top:6px;'>No background videos found in <code>./video</code>. You can upload up to 6 videos to use as the login background. Falling back to demo videos so you can preview the effect.</div>", unsafe_allow_html=True)

                # Public demo videos (safe, widely-available samples). Used only as a visual preview
                # Prefer relative, web-style paths so browsers can fetch from the app
                demo_videos = [
                    "video/1.mp4",
                    "video/2.mp4",
                    "video/3.mp4"
                ]

                # Use demo videos as the background so users see the effect immediately
                video_list_json = json.dumps(demo_videos)
                demo_html = (
                    """
                <style>
                .bg-video { position: fixed; top:0; left:0; width:100%; height:100%; object-fit:cover; z-index: 0; pointer-events:none; filter: brightness(0.42) saturate(0.95); }
                .bg-video-overlay { position: fixed; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.36); z-index: 1; pointer-events:none; }
                .login-card { position: relative; z-index: 9999; pointer-events: auto; }
                </style>
                <video id="bgVideo" class="bg-video" autoplay muted playsinline loop preload="metadata" poster="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='1600' height='900'><rect width='100%' height='100%' fill='%23000000'/></svg>"></video>
                <div class="bg-video-overlay"></div>
                <script>
                    const _videos_demo = """ + video_list_json + """;
                    const _v = document.getElementById('bgVideo');
                    function _setSrc(url){ if(!_v) return; try{ _v.pause(); _v.muted = true; _v.playsInline = true; _v.autoplay = true; _v.src = url; _v.load(); _v.play().catch(()=>{}); }catch(e){} }
                    try{
                        const list = Array.isArray(_videos_demo) ? _videos_demo : JSON.parse(_videos_demo);
                        if(list && list.length>0){
                            _v.preload = 'metadata';
                            let idx = 0;
                            _setSrc(list[0]);
                            _v.addEventListener('ended', ()=>{ idx = (idx+1) % list.length; _setSrc(list[idx]); });
                            _v.addEventListener('error', ()=>{ idx = (idx+1) % list.length; _setSrc(list[idx]); });
                        }
                    }catch(e){ try{ _setSrc(Array.isArray(_videos_demo) ? _videos_demo[0] : JSON.parse(_videos_demo)[0]); }catch(_){} }
                </script>
                """
                )
                st.markdown(demo_html, unsafe_allow_html=True)

                uploaded = st.file_uploader("Upload background videos (mp4/webm/ogg)", accept_multiple_files=True, type=['mp4', 'webm', 'ogg'])
                if uploaded:
                    os.makedirs(video_dir, exist_ok=True)
                    saved = 0
                    for uf in uploaded:
                        try:
                            dest = os.path.join(video_dir, uf.name)
                            with open(dest, 'wb') as f:
                                f.write(uf.getbuffer())
                            saved += 1
                        except Exception:
                            pass
                    if saved:
                        st.success(f"Saved {saved} file(s) to ./video — reloading to apply background...")
                        safe_rerun()
            except Exception:
                pass
    except Exception:
        # If anything goes wrong (e.g., in environments that strip scripts), ignore and continue
        pass
    with st.container():
        # Demo credentials banner (helpful for quick local testing) — larger and high-contrast
        st.markdown("""
    <div class='demo-credentials' style='position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9996;background:linear-gradient(90deg, rgba(30,144,255,0.08), rgba(46,125,50,0.06));color:#ffffff;padding:10px 18px;border-radius:12px;font-size:14px;font-weight:600;border:1px solid rgba(255,255,255,0.06);backdrop-filter: blur(4px); pointer-events: none;'>
            Demo logins — Farmer: <code style="color:#fff; background:rgba(0,0,0,0.12); padding:2px 6px; border-radius:4px">demo_farmer</code> / <code style="color:#fff; background:rgba(0,0,0,0.12); padding:2px 6px; border-radius:4px">farmer</code> &nbsp; • &nbsp; Processor: <code style="color:#fff; background:rgba(0,0,0,0.12); padding:2px 6px; border-radius:4px">demo_processor</code> / <code style="color:#fff; background:rgba(0,0,0,0.12); padding:2px 6px; border-radius:4px">processor</code>
            &nbsp; • &nbsp; Buyer: <code style="color:#fff; background:rgba(0,0,0,0.12); padding:2px 6px; border-radius:4px">demo_buyer</code> / <code style="color:#fff; background:rgba(0,0,0,0.12); padding:2px 6px; border-radius:4px">buyer</code>
            &nbsp; • &nbsp; Govt: <code style="color:#fff; background:rgba(0,0,0,0.12); padding:2px 6px; border-radius:4px">demo_govt</code> / <code style="color:#fff; background:rgba(0,0,0,0.12); padding:2px 6px; border-radius:4px">govt</code>
        </div>
        """, unsafe_allow_html=True)

    # (Removed fixed backdrop so Streamlit widgets remain clickable.)
        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.markdown("<div class='login-card'>", unsafe_allow_html=True)
            # App title shown above sign-in to remind users what this portal is for
            st.markdown("<h2 style='text-align:center; margin-bottom:8px; color:#ffffff;'>🌾 Value Chain Integration System for Oilseed By-Products</h2>", unsafe_allow_html=True)
            st.markdown("<h1 style='font-size:28px; margin-bottom:10px;'>Sign in</h1>", unsafe_allow_html=True)

            # Large role-based quick-login buttons (big, accessible)
            quick_css = """
            <style>
            .quick-login .stButton>button { padding:14px 18px !important; font-size:16px !important; border-radius:10px !important; }
            .quick-login .stButton>button:focus { outline: 3px solid rgba(30,144,255,0.24) !important; }
            </style>
            """
            st.markdown(quick_css, unsafe_allow_html=True)
            cols_quick = st.columns(4)
            with cols_quick[0]:
                if st.button('Farmer', key='quick_farmer'):
                    st.session_state['login_email'] = 'demo_farmer'
                    st.session_state['login_password'] = 'farmer'
                    st.session_state['login_role'] = 'Farmer'
                    st.session_state['auto_login'] = True
                    safe_rerun()
            with cols_quick[1]:
                if st.button('Processor', key='quick_processor'):
                    st.session_state['login_email'] = 'demo_processor'
                    st.session_state['login_password'] = 'processor'
                    st.session_state['login_role'] = 'Processor'
                    st.session_state['auto_login'] = True
                    safe_rerun()
            with cols_quick[2]:
                if st.button('Buyer', key='quick_buyer'):
                    st.session_state['login_email'] = 'demo_buyer'
                    st.session_state['login_password'] = 'buyer'
                    st.session_state['login_role'] = 'Buyer'
                    st.session_state['auto_login'] = True
                    safe_rerun()
            with cols_quick[3]:
                if st.button('Govt', key='quick_govt'):
                    st.session_state['login_email'] = 'demo_govt'
                    st.session_state['login_password'] = 'govt'
                    st.session_state['login_role'] = 'Govt'
                    st.session_state['auto_login'] = True
                    safe_rerun()

            # Login form
            with st.form('login_form'):
                email = st.text_input('Email', key='login_email')
                password = st.text_input('Password', type='password', key='login_password')
                role = st.selectbox('Please Select User Role', ['Farmer', 'Processor', 'Buyer', 'Govt'], key='login_role')
                st.markdown("<div class='login-forgot'><a href='#'>Forgot Your Password?</a></div>", unsafe_allow_html=True)
                st.markdown("<br/>", unsafe_allow_html=True)
                submit = st.form_submit_button('Submit')

                if submit:
                    # Attempt authentication
                    auth = authenticate_user(email, password)
                    if auth:
                        # Use the chosen role if the user is creating a new session
                        user_record = {'username': auth['username'], 'role': auth['role']}
                        # If the existing user has no role or we want to override on first login,
                        # keep the DB role.
                        st.session_state['user'] = user_record
                        safe_rerun()
                    else:
                        st.error('Invalid credentials. If you are a new user, register using the sidebar or contact admin.')

            # Optional Google sign-in button below
            st.markdown("<div style='margin-top:12px'>", unsafe_allow_html=True)
            if st.button('Sign in with Google'):
                userinfo = google_oauth_flow()
                if userinfo:
                    email = userinfo.get('email')
                    name = userinfo.get('name') or email.split('@')[0]
                    picture = userinfo.get('picture')
                    st.session_state['user'] = {'username': email, 'name': name, 'picture': picture, 'role': 'Farmer'}
                    session = Session()
                    if not session.query(User).filter_by(username=email).first():
                        create_user(email, '', 'Farmer')
                    session.close()
                    safe_rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Floating glowing arrow button to indicate / jump to login
        # The button will trigger a rerun and set a small focus flag in session state
        arrow_css = """
        <style>
        .glow-arrow .stButton>button { 
            position: fixed !important; 
            bottom: 18px !important; 
            left: 50% !important; 
            transform: translateX(-50%) !important; 
            width: 56px !important; 
            height: 56px !important; 
            border-radius: 999px !important; 
            background: linear-gradient(180deg, #00c853, #2e7d32) !important; 
            box-shadow: 0 8px 24px rgba(46,125,50,0.36), 0 0 18px rgba(30,144,255,0.18) !important; 
            color: white !important; 
            font-size: 28px !important; 
            display:flex !important; align-items:center !important; justify-content:center !important;
            border: none !important;
        }
        .glow-arrow .stButton>button:hover { transform: translateX(-50%) scale(1.03) !important; }
        </style>
        """
        st.markdown(arrow_css, unsafe_allow_html=True)

        # Render the arrow button inside a wrapper so CSS can target it; clicking it will set a session flag and rerun
        st.markdown("<div class='glow-arrow'>", unsafe_allow_html=True)
        if st.button("↓", key='login_arrow'):
            st.session_state['focus_login'] = True
            safe_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Auto-login handler: if a quick-login button set auto_login flag, perform authentication immediately
        if st.session_state.get('auto_login'):
            e = st.session_state.get('login_email', '')
            p = st.session_state.get('login_password', '')
            auth = authenticate_user(e, p)
            # clear the flag to avoid loops
            st.session_state['auto_login'] = False
            if auth:
                st.session_state['user'] = {'username': auth['username'], 'role': auth['role']}
                safe_rerun()

# If no user logged in, show login overlay and stop further rendering
if 'user' not in st.session_state or not st.session_state.get('user'):
        render_fullscreen_login()
        st.stop()

# Post-login rotating background images removed per user request.
# The previous implementation injected a full-viewport <div class="app-bg" id="appBg"> and
# ran client-side JS to cycle background images. That logic was removed to avoid
# rendering the empty placeholder and to simplify the page.


# Compatibility helper: safe rerun for Streamlit versions without experimental_rerun
def safe_rerun():
    """Call Streamlit's rerun API when available; otherwise no-op.

    Some Streamlit versions expose `experimental_rerun`, others rely on automatic
    reruns after widget interactions. This helper avoids AttributeError.
    """
    if hasattr(st, 'experimental_rerun') and callable(getattr(st, 'experimental_rerun')):
        try:
            st.experimental_rerun()
        except Exception:
            # Best-effort: if calling fails, ignore — Streamlit will rerun on next interaction
            pass


# Custom CSS: tab hover effect and active-tab highlight
# Adds a subtle lift on hover and an accent + glow for the active tab.
# Placed here so it is injected early in the page lifecycle.
    css = """
    <style>
    /* Make app background fully black */
    .stApp { background: #000000 !important; color: #ffffff !important; }
    .login-center { display:flex; align-items:center; justify-content:center; height:85vh; }
    /* Card uses dark translucent surface so it reads on black background */
    .login-card { background: rgba(20,20,20,0.9); border-radius:10px; padding:32px; width:520px; box-shadow:0 8px 30px rgba(0,0,0,0.6); }
    .login-card h1 { text-align:center; margin-bottom:18px; font-weight:700; color:#ffffff; }
    /* Inputs: dark field with light text */
    .login-input input { height:44px; border-radius:6px; border:1px solid rgba(255,255,255,0.06); padding:10px 14px; width:100%; background:#0b0b0b; color:#e6e6e6; }
    .login-select select { height:44px; border-radius:6px; border:1px solid rgba(255,255,255,0.06); padding:8px 12px; width:100%; background:#0b0b0b; color:#e6e6e6; }
    .login-forgot { text-align:left; margin-top:8px; }
    .login-forgot a { color:#93c5fd; }
    .submit-btn button { background:#2f9e44; color:white; border-radius:8px; padding:12px 18px; width:100%; border: none; font-size:16px; }
    .submit-btn button:hover { opacity:0.95; }
    .google-btn { background:#111827; color:#fff; padding:10px 14px; border-radius:6px; border:1px solid rgba(255,255,255,0.04); width:100%; }
    /* Ensure Streamlit buttons inside card match and remain clickable */
    .login-card .stButton>button { background: #2f9e44 !important; color: white !important; border-radius: 8px !important; padding: 16px 20px !important; width: 100% !important; font-size: 18px !important; border: none !important; pointer-events: auto !important; }
    /* Ensure general Streamlit buttons inside the login card are prominent */
    .login-card .stButton>button:focus { outline: 3px solid rgba(30,144,255,0.22) !important; }
    </style>
    """

tab_css = """
<style>
/* Active tab: colored accent, glow and subtle border */
div[role="tablist"] > button[role="tab"][aria-selected="true"] {
    transform: translateY(-6px);
    box-shadow: 0 12px 36px rgba(30, 144, 255, 0.16);
    border: 1px solid rgba(30, 144, 255, 0.18);
    background: linear-gradient(180deg, rgba(30, 144, 255, 0.04), rgba(30, 144, 255, 0.01));
}

/* Make icon and label inside tabs align nicely (best-effort across Streamlit versions) */
div[role="tablist"] > button[role="tab"] > div {
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

/* Reduce visual jump when switching tabs */
div[role="tablist"] > button[role="tab"]:not([aria-selected="true"]) {
    opacity: 0.95;
}
</style>
"""

st.markdown(tab_css, unsafe_allow_html=True)

# Hero header (large, centered) — keeps existing dark background intact
hero_html = """
<style>
.hero { padding: 48px 24px; text-align: center; color: #fff; }
.hero .badge { display:inline-block; background: rgba(255,255,255,0.06); color:#fff; padding:10px 22px; border-radius:999px; font-weight:700; margin-bottom:18px; }
.hero h1 { font-size: calc(28px + 3.6vw); line-height: 1.02; margin: 8px 0 12px; font-weight:800; letter-spacing:-1px; color: #ffffff; }
.hero p.lead { font-size:18px; opacity:0.85; margin-bottom:14px; }
.hero .features { font-size:14px; opacity:0.9; }
@media (min-width:1200px) { .hero h1 { font-size: 72px; } }
.hero .container { max-width: 1200px; margin: 0 auto; }
</style>
<div class="hero">
    <div class="container">
        <div class="badge">Learn more about the Value Chain</div>
        <h1>Value Chain Integration System for Oilseed By‑Products</h1>
        <p class="lead">A unified web app for marketplace, IoT ingestion, blockchain-style ledger, and AI forecasting.</p>
        <div class="features">AI-Powered Marketplace &nbsp; • &nbsp; IoT Integration &nbsp; • &nbsp; Blockchain Transactions &nbsp; • &nbsp; Predictive Analytics</div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)


def render_role_difference_viewer():
    """Show a role-based difference viewer comparing capabilities for each role.

    Displays a table of capability categories vs roles and allows a side-by-side
    comparison (and a computed difference) between two selected roles.
    """
    import pandas as _pd

    roles = ["Farmer", "Processor", "Buyer", "Govt"]

    categories = [
        "Listings",
        "Inventory",
        "Telemetry / IoT",
        "Smart Contracts",
        "Forecasts & Analytics",
        "Transactions & Ledger",
        "Auditing / Reports",
        "Certifications / Compliance",
    ]

    # Permissions / capabilities per role (short descriptions)
    perms = {
        "Farmer": {
            "Listings": "Create listings; update stock levels; set reserve prices",
            "Inventory": "Full access to own inventory and quality metrics",
            "Telemetry / IoT": "Submit device telemetry; view own device history",
            "Smart Contracts": "Accept / propose contracts tied to own listings",
            "Forecasts & Analytics": "View forecasts relevant to own crops/commodities",
            "Transactions & Ledger": "View transactions involving own listings; receive payments",
            "Auditing / Reports": "Generate simple farm-level reports",
            "Certifications / Compliance": "Upload certificates; request inspections",
        },
        "Processor": {
            "Listings": "Search and bid on listings; create processing offers",
            "Inventory": "Track incoming raw material flows; update processing outputs",
            "Telemetry / IoT": "Access telemetry for contracted suppliers (consent-based)",
            "Smart Contracts": "Create/execute contracts for processing and delivery",
            "Forecasts & Analytics": "View demand forecasts and price trends for processed goods",
            "Transactions & Ledger": "Settle purchases; view related ledger entries",
            "Auditing / Reports": "Run processing KPIs and traceability reports",
            "Certifications / Compliance": "Manage compliance for processed products",
        },
        "Buyer": {
            "Listings": "Search, filter and purchase listings; request samples/certificates",
            "Inventory": "View shipment & delivery status for purchases",
            "Telemetry / IoT": "Limited access to telemetry attached to shipments (read-only)",
            "Smart Contracts": "Propose/accept purchase contracts and escrow terms",
            "Forecasts & Analytics": "Access market price dashboards and trend alerts",
            "Transactions & Ledger": "View own purchase history and invoices",
            "Auditing / Reports": "Request supplier audit reports",
            "Certifications / Compliance": "View supplier certifications before purchase",
        },
        "Govt": {
            "Listings": "View anonymized/aggregate listing data; request access for audits",
            "Inventory": "Access aggregated inventory stats for regulation and food security",
            "Telemetry / IoT": "Access anonymized telemetry; request detailed on legal grounds",
            "Smart Contracts": "View executed contracts for oversight (read-only/auditable)",
            "Forecasts & Analytics": "Access national/regional forecasts and trend analytics",
            "Transactions & Ledger": "Run compliance and anti-fraud queries across ledger",
            "Auditing / Reports": "Generate regulatory reports; export audit trails",
            "Certifications / Compliance": "Issue/verify certifications and approvals",
        },
    }

    # Build a DataFrame for a quick tabular view
    table = {role: [perms[role].get(cat, "-") for cat in categories] for role in roles}
    df = _pd.DataFrame(table, index=categories)

    st.subheader("Role-Based Access — Difference Viewer")
    st.markdown("Select roles to compare and inspect capability differences across the system.")

    with st.expander("Full role capability matrix", expanded=False):
        st.dataframe(df)

    # Compare two roles side-by-side
    sel = st.multiselect("Pick up to two roles to compare (side-by-side)", roles, default=[st.session_state.get('user', {}).get('role') or roles[0]])
    if len(sel) == 0:
        st.info("Choose one or two roles to see details.")
        return

    if len(sel) == 1:
        role = sel[0]
        st.markdown(f"### {role} — Capabilities")
        for cat in categories:
            st.markdown(f"**{cat}**: {perms[role].get(cat, '-')}")
        return

    # Two roles selected: show side-by-side and compute diffs
    left, right = sel[0], sel[1]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {left}")
        for cat in categories:
            st.markdown(f"**{cat}**: {perms[left].get(cat, '-')}")
    with col2:
        st.markdown(f"#### {right}")
        for cat in categories:
            st.markdown(f"**{cat}**: {perms[right].get(cat, '-')}")

    # Compute simple textual differences (capability strings that differ per category)
    diffs = []
    for cat in categories:
        a = perms[left].get(cat, '')
        b = perms[right].get(cat, '')
        if a != b:
            diffs.append((cat, a, b))

    if diffs:
        st.markdown("---")
        st.markdown("### Differences (per category)")
        for cat, a, b in diffs:
            st.markdown(f"**{cat}**")
            st.markdown(f"- {left}: {a}")
            st.markdown(f"- {right}: {b}")
    else:
        st.success("No differences detected between the selected roles for the tracked categories.")


# Sidebar login/register removed — login happens via the fullscreen overlay.
# Keep a session placeholder for `user`.
if 'user' not in st.session_state:
    st.session_state['user'] = None

# Optional preview flag: set environment variable PREVIEW_HERO=1 to bypass login and preview the hero/header.
try:
    if os.environ.get('PREVIEW_HERO', '0') == '1':
        # Create demo user for preview if not exists
        create_user('demo_farmer', 'farmer', 'Farmer')
        st.session_state['user'] = {'username': 'demo_farmer', 'role': 'Farmer'}
except Exception:
    pass


# Show the role-difference viewer on the main page for logged-in users
if st.session_state.get('user'):
    user_role = st.session_state['user'].get('role') if isinstance(st.session_state['user'], dict) else None
    # Default to collapsed; user can expand to inspect differences
    with st.expander(f"Role-Based Access (signed in as: {user_role})", expanded=False):
        render_role_difference_viewer()

# ============================================================================
# TAB STRUCTURE
# ============================================================================

tab_dashboard, tab_food, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👤 Dashboard",
    "🍲 Live Food Data",
    "🏪 Marketplace",
    "🔌 IoT Devices",
    "📈 AI Forecasting",
    "🔐 Transaction Ledger",
    "🌍 Export Matchmaking",
    "🗺️ Roadmap"
])

with tab_dashboard:
    st.header("👤 User Dashboard")
    user = st.session_state.get('user')
    if not user:
        st.info("You are not signed in. Use the sidebar to register or login.\n\nPreview: Farmer -> IoT & Prices; Processor -> Marketplace summary; Buyer -> Available stock.")
        # Demo quick links
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            if st.button("Quick demo: Farmer"):
                # create demo user if needed and sign-in
                create_user('demo_farmer', 'farmer', 'Farmer')
                st.session_state['user'] = {'username': 'demo_farmer', 'role': 'Farmer'}
                safe_rerun()
        with col_b:
            if st.button("Quick demo: Processor"):
                create_user('demo_processor', 'processor', 'Processor')
                st.session_state['user'] = {'username': 'demo_processor', 'role': 'Processor'}
                safe_rerun()
        with col_c:
            if st.button("Quick demo: Buyer"):
                create_user('demo_buyer', 'buyer', 'Buyer')
                st.session_state['user'] = {'username': 'demo_buyer', 'role': 'Buyer'}
                safe_rerun()
        with col_d:
            if st.button("Quick demo: Govt"):
                create_user('demo_govt', 'govt', 'Govt')
                st.session_state['user'] = {'username': 'demo_govt', 'role': 'Govt'}
                safe_rerun()
    else:
        role = user['role']
        st.markdown(f"**Signed in as:** `{user['username']}` — **{role}**")
        session = Session()
        if role == 'Farmer':
            st.subheader("IoT Metrics & Latest Prices")
            devices = session.query(Device).all()
            if devices:
                avg_temp = np.mean([d.temperature for d in devices if d.temperature is not None])
                avg_humidity = np.mean([d.humidity for d in devices if d.humidity is not None])
                total_prod = sum([d.production_rate for d in devices if d.production_rate is not None])
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
                with c2:
                    st.metric("Avg Humidity", f"{avg_humidity:.1f}%")
                with c3:
                    st.metric("Total Production", f"{total_prod:.0f} kg/hr")
            else:
                st.info("No devices registered yet. Register devices in the IoT tab.")

            st.markdown("---")
            st.subheader("Latest Market Prices")
            products = ["Soymeal", "Groundnut Cake", "Sunflower Cake", "Mustard Cake", "Husk"]
            price_rows = []
            for p in products:
                latest = session.query(PriceHistory).filter_by(product=p).order_by(PriceHistory.date.desc()).first()
                if latest:
                    price_rows.append({'Product': p, 'Price (₹/kg)': f"{latest.price:.2f}", 'Date': latest.date.strftime('%Y-%m-%d')})
            if price_rows:
                st.table(pd.DataFrame(price_rows))
            else:
                st.info("No historical price data available. Generate forecasts in the AI Forecasting tab.")

        elif role == 'Processor':
            st.subheader("Marketplace Summary & Device Health")
            listings = session.query(Listing).filter_by(status='active').all()
            st.write(f"Active Listings: **{len(listings)}**")
            if listings:
                df_list = pd.DataFrame([{'Seller': l.seller, 'Product': l.product, 'Qty (kg)': l.quantity, 'Price/kg': l.price_per_kg, 'Quality': l.quality_grade} for l in listings])
                st.dataframe(df_list, use_container_width=True)
            else:
                st.info("No active listings currently.")

            st.markdown("---")
            st.subheader("Device Health Snapshot")
            devices = session.query(Device).all()
            if devices:
                st.dataframe(pd.DataFrame([{'Device ID': d.device_id, 'Temp': d.temperature, 'Humidity': d.humidity, 'Production': d.production_rate} for d in devices]), use_container_width=True)
            else:
                st.info("No devices registered yet.")

        elif role == 'Buyer':
            st.subheader("Available Stock")
            listings = session.query(Listing).filter_by(status='active').order_by(Listing.created_at.desc()).all()
            if listings:
                for l in listings:
                    with st.expander(f"{l.product} — {l.quantity} kg @ ₹{l.price_per_kg}/kg"):
                        st.write(f"**Seller:** {l.seller} — **Quality:** {l.quality_grade} — **Location:** {l.location}")
                        st.write(f"**Total Value:** ₹{l.quantity * l.price_per_kg:,.2f}")
                        if st.button("Purchase (Buyer Dashboard)", key=f"buy_dash_{l.id}"):
                            buyer_name = user['username']
                            tx_hash = log_transaction('sale', l.seller, buyer_name, l.product, l.quantity, l.quantity * l.price_per_kg)
                            l.status = 'sold'
                            session.commit()
                            st.success(f"Purchase complete. TX: {tx_hash[:16]}...")
            else:
                st.info("No active listings. Check back later or post a request in the marketplace.")

        session.close()
# ============================================================================
# TAB 1: MARKETPLACE
# ============================================================================

# ============================================================================
# TAB: LIVE FOOD DATA
# ============================================================================

with tab_food:
    st.header("🍲 Live Food Data Monitoring")
    render_live_food_dashboard()

# ============================================================================
# TAB 1: MARKETPLACE
# ============================================================================

with tab1:
    st.header("Digital Marketplace for Oilseed By-Products")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Create New Listing")
        with st.form("listing_form"):
            seller_name = st.text_input("Seller Name", "Farmer Cooperative XYZ")
            product = st.selectbox("Product", [
                "Soymeal", "Groundnut Cake", "Sunflower Cake", 
                "Mustard Cake", "Husk"
            ])
            quantity = st.number_input("Quantity (kg)", min_value=1.0, value=1000.0, step=100.0)
            price_per_kg = st.number_input("Price per kg (₹)", min_value=1.0, value=35.0, step=1.0)
            quality = st.selectbox("Quality Grade", ["Premium", "Grade A", "Grade B", "Standard"])
            location = st.text_input("Location", "Punjab")
            
            submitted = st.form_submit_button("Post Listing")
            
            if submitted:
                session = Session()
                listing = Listing(
                    seller=seller_name,
                    product=product,
                    quantity=quantity,
                    price_per_kg=price_per_kg,
                    quality_grade=quality,
                    location=location
                )
                session.add(listing)
                session.commit()
                
                # Log to blockchain ledger
                tx_hash = log_transaction(
                    'listing',
                    seller_name,
                    'N/A',
                    product,
                    quantity,
                    quantity * price_per_kg
                )
                session.close()
                st.success(f"✅ Listing posted! Transaction hash: `{tx_hash[:16]}...`")
    
    with col2:
        st.subheader("Active Listings")
        session = Session()
        listings = session.query(Listing).filter_by(status='active').order_by(Listing.created_at.desc()).all()
        
        if listings:
            for listing in listings[:10]:
                with st.expander(f"{listing.product} - {listing.quantity} kg @ ₹{listing.price_per_kg}/kg"):
                    st.write(f"**Seller:** {listing.seller}")
                    st.write(f"**Quality:** {listing.quality_grade}")
                    st.write(f"**Location:** {listing.location}")
                    st.write(f"**Total Value:** ₹{listing.quantity * listing.price_per_kg:,.2f}")
                    st.write(f"**Posted:** {listing.created_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    buyer_name = st.text_input(f"Your Name", key=f"buyer_{listing.id}", value="Buyer Corp")
                    if st.button(f"Purchase", key=f"buy_{listing.id}"):
                        # Execute smart contract simulation
                        tx_hash = log_transaction(
                            'sale',
                            listing.seller,
                            buyer_name,
                            listing.product,
                            listing.quantity,
                            listing.quantity * listing.price_per_kg
                        )
                        listing.status = 'sold'
                        session.commit()
                        st.success(f"✅ Purchase complete! TX: `{tx_hash[:16]}...`")
                        st.rerun()
        else:
            st.info("No active listings. Create one to get started!")
        
        session.close()

# ============================================================================
# TAB 2: IOT DEVICES
# ============================================================================

with tab2:
    st.header("IoT Device Management & Telemetry")
    
    # Create tabs for different IoT features
    device_tab, telemetry_tab, api_tab = st.tabs([
        "📱 Devices", "📊 Telemetry", "🔌 API Integration"
    ])
    
    with device_tab:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Register IoT Device")
            with st.form("device_form"):
                device_id = st.text_input("Device ID", f"IOT-{np.random.randint(1000, 9999)}")
                device_type = st.selectbox("Device Type", [
                    "Storage Monitor",
                    "Production Monitor",
                    "Quality Analyzer",
                    "Environmental Sensor",
                    "Process Controller"
                ])
                device_location = st.text_input("Location", "Processing Unit A")
                firmware_version = st.text_input("Firmware Version", "1.0.0")
                
                register = st.form_submit_button("Register Device")
                
                if register:
                    session = Session()
                    existing = session.query(Device).filter_by(device_id=device_id).first()
                    if not existing:
                        # Generate API key for the device
                        api_key = generate_api_key()
                        
                        device = Device(
                            device_id=device_id,
                            device_type=device_type,
                            location=device_location,
                            temperature=0,
                            humidity=0,
                            production_rate=0,
                            firmware_version=firmware_version,
                            maintenance_status='active',
                            battery_level=100.0,
                            signal_strength=100.0,
                            alert_status='normal',
                            api_key=api_key
                        )
                        session.add(device)
                        session.commit()
                        st.success(f"✅ Device {device_id} registered!")
                        st.info(f"API Key: `{api_key}`\n\nStore this key securely - it's required for device authentication.")
                    else:
                        st.warning("Device ID already exists!")
                    session.close()
        
        with col2:
            st.subheader("Device Management")
            session = Session()
            devices = session.query(Device).all()
            
            if devices:
                for device in devices:
                    with st.expander(f"📱 {device.device_id} - {device.device_type}"):
                        cols = st.columns(3)
                        with cols[0]:
                            st.markdown(f"**Location:** {device.location}")
                            st.markdown(f"**Status:** {device.maintenance_status}")
                            st.markdown(f"**Firmware:** {device.firmware_version}")
                        
                        with cols[1]:
                            st.markdown(f"**Battery:** {device.battery_level}%")
                            st.markdown(f"**Signal:** {device.signal_strength}%")
                            st.markdown(f"**Alert Status:** {device.alert_status}")
                        
                        with cols[2]:
                            if st.button("Maintenance Check", key=f"maint_{device.id}"):
                                device.maintenance_status = 'under_maintenance'
                                device.last_maintenance = datetime.now()
                                session.commit()
                                st.success("Maintenance mode activated")
                                st.rerun()
                            
                            if st.button("Update Firmware", key=f"firm_{device.id}"):
                                device.firmware_version = f"1.{float(device.firmware_version.split('.')[1]) + 0.1:.1f}"
                                session.commit()
                                st.success("Firmware updated")
                                st.rerun()
            else:
                st.info("No devices registered. Register your first IoT device!")
            
            session.close()
    
    with telemetry_tab:
        st.subheader("Real-Time Telemetry Dashboard")
        
        # Time range selector
        hours = st.slider("Time Range (hours)", 1, 72, 24)
        
        session = Session()
        devices = session.query(Device).all()
        
        if devices:
            # Device selector
            selected_device = st.selectbox(
                "Select Device",
                options=[d.device_id for d in devices],
                format_func=lambda x: f"{x} ({next(d.device_type for d in devices if d.device_id == x)})"
            )
            
            # Get device statistics
            stats = get_device_stats(selected_device, hours)
            
            if stats:
                # Display real-time metrics
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric(
                        "Temperature",
                        f"{stats['temperature']['current']:.1f}°C",
                        f"{stats['temperature']['current'] - stats['temperature']['avg']:.1f}°C"
                    )
                with metric_cols[1]:
                    st.metric(
                        "Humidity",
                        f"{stats['humidity']['current']:.1f}%",
                        f"{stats['humidity']['current'] - stats['humidity']['avg']:.1f}%"
                    )
                with metric_cols[2]:
                    st.metric(
                        "Production Rate",
                        f"{stats['production_rate']['current']:.0f} kg/hr",
                        f"{stats['production_rate']['current'] - stats['production_rate']['avg']:.0f} kg/hr"
                    )
                with metric_cols[3]:
                    st.metric("Alerts", stats['alerts'])
                
                # Time series visualization
                fig = go.Figure()
                
                # Temperature trend
                fig.add_trace(go.Scatter(
                    x=stats['timestamps'],
                    y=[stats['temperature']['current'] for _ in stats['timestamps']],
                    name='Temperature (°C)',
                    line=dict(color='red', width=2)
                ))
                
                # Humidity trend
                fig.add_trace(go.Scatter(
                    x=stats['timestamps'],
                    y=[stats['humidity']['current'] for _ in stats['timestamps']],
                    name='Humidity (%)',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title='Environmental Trends',
                    xaxis_title='Time',
                    yaxis_title='Value',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Production metrics
                prod_fig = go.Figure()
                prod_fig.add_trace(go.Scatter(
                    x=stats['timestamps'],
                    y=[stats['production_rate']['current'] for _ in stats['timestamps']],
                    name='Production Rate',
                    fill='tozeroy',
                    line=dict(color='green', width=2)
                ))
                
                prod_fig.update_layout(
                    title='Production Rate Trend',
                    xaxis_title='Time',
                    yaxis_title='kg/hr',
                    height=300
                )
                
                st.plotly_chart(prod_fig, use_container_width=True)
            else:
                st.warning("No telemetry data available for selected time range")
        else:
            st.info("No devices registered. Register your first IoT device!")
        
        session.close()
    
    with api_tab:
        st.subheader("IoT API Integration")
        
        st.markdown("""
        ### API Documentation
        
        The IoT Integration API allows devices to send telemetry data securely. Each device needs to authenticate using its API key.
        
        #### Endpoint: `/api/v1/telemetry`
        **Method:** POST
        
        **Headers:**
        ```
        X-API-Key: <device_api_key>
        Content-Type: application/json
        ```
        
        **Request Body:**
        ```json
        {
            "temperature": float,
            "humidity": float,
            "production_rate": float,
            "quality_metrics": {
                "moisture": float,
                "protein_content": float,
                "oil_content": float,
                // ... other metrics
            },
            "device_status": {
                "battery_level": float,
                "signal_strength": float
            }
        }
        ```
        
        **Example Python Code:**
        ```python
        import requests
        import json
        
        def send_telemetry(api_key, data):
            url = "http://your-server/api/v1/telemetry"
            headers = {
                "X-API-Key": api_key,
                "Content-Type": "application/json"
            }
            response = requests.post(url, headers=headers, json=data)
            return response.json()
        ```
        
        ### Testing Tools
        """)
        
        # API Tester
        with st.expander("API Tester"):
            test_api_key = st.text_input("Device API Key")
            test_data = st.text_area(
                "Test Data (JSON)",
                value='''{
    "temperature": 25.5,
    "humidity": 60.0,
    "production_rate": 450.0,
    "quality_metrics": {
        "moisture": 12.5,
        "protein_content": 46.2,
        "oil_content": 1.8
    },
    "device_status": {
        "battery_level": 85.0,
        "signal_strength": 92.0
    }
}'''
            )
            
            if st.button("Test API"):
                try:
                    data = json.loads(test_data)
                    if validate_api_key(test_api_key):
                        success, result = ingest_telemetry(test_api_key, data)
                        if success:
                            st.success("Data ingested successfully!")
                        else:
                            st.error(f"Error: {result}")
                    else:
                        st.error("Invalid API key")
                except json.JSONDecodeError:
                    st.error("Invalid JSON data")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # API Metrics
        st.subheader("API Usage Metrics")
        api_cols = st.columns(4)
        
        with api_cols[0]:
            session = Session()
            total_devices = session.query(Device).count()
            st.metric("Registered Devices", total_devices)
        
        with api_cols[1]:
            total_telemetry = session.query(DeviceTelemetry).count()
            st.metric("Total Data Points", total_telemetry)
        
        with api_cols[2]:
            recent_telemetry = session.query(DeviceTelemetry).filter(
                DeviceTelemetry.timestamp >= datetime.now() - timedelta(hours=24)
            ).count()
            st.metric("24h Data Points", recent_telemetry)
        
        with api_cols[3]:
            alerts = session.query(DeviceTelemetry).filter_by(alert_triggered=True).count()
            st.metric("Total Alerts", alerts)

# ============================================================================
# TAB 3: AI FORECASTING
# ============================================================================

with tab3:
    st.header("Advanced AI Market Analytics")
    st.markdown("**AI-Powered Market Intelligence:** Price Trends, Demand Forecasting, and Volatility Analysis")
    
    # -----------------------------
    # Live Commodity Prices
    # -----------------------------
    st.subheader("Live Commodity Prices")
    st.markdown("Fetch near real-time commodity futures prices (via Yahoo Finance). Select commodities to display current price and a recent trend chart.")

    # Available commodities default to those in DEFAULT_TICKER_MAP
    available = list(DEFAULT_TICKER_MAP.keys()) if DEFAULT_TICKER_MAP else ['Soybean', 'Corn', 'Wheat', 'Soybean Oil']
    selected_commodities = st.multiselect("Select commodities to show", options=available, default=available[:3], key='selected_commodities')

    refresh_min = st.selectbox("Auto-refresh interval (minutes)", [1, 2, 5, 10, 30], index=2)
    auto_refresh = st.checkbox("Enable auto-refresh (page reload)", value=False)
    if auto_refresh:
        # Inject a simple page reload every refresh_min minutes
        ms = int(refresh_min) * 60 * 1000
        st.markdown(f"<script>setInterval(()=>{{location.reload();}}, {ms});</script>", unsafe_allow_html=True)

    period = st.selectbox("Historical period", ['30d', '90d', '180d', '365d'], index=1)
    interval = st.selectbox("Historical interval", ['1d', '1wk'], index=0)

    # Data loader with caching; TTL uses refresh minutes
    @st.cache_data(ttl=60 * int(refresh_min))
    def _load(selected, period, interval):
        if not selected:
            return {}
        return get_commodities_data(selected, period=period, interval=interval)

    with st.spinner('Fetching commodity prices...'):
        com_data = _load(selected_commodities, period, interval)

    # Display current prices as metrics
    if selected_commodities:
        metric_cols = st.columns(len(selected_commodities))
        for i, name in enumerate(selected_commodities):
            info = com_data.get(name, {}) or {}
            ticker = info.get('ticker')
            cur = info.get('current')
            label = f"{name} ({ticker})" if ticker else name
            if cur is None:
                metric_cols[i].metric(label, "N/A")
            else:
                # Format with two decimals
                metric_cols[i].metric(label, f"{cur:,.2f}")
    else:
        st.info("Choose one or more commodities to display current prices and chart.")

    # Plotly time series of selected commodities
    import plotly.graph_objects as pgo
    fig = pgo.Figure()
    plotted = False
    for name in selected_commodities:
        info = com_data.get(name, {}) or {}
        hist = info.get('history')
        if hist is None or hasattr(hist, 'empty') and getattr(hist, 'empty'):
            continue
        # Expect DataFrame with 'Datetime' and 'close'
        try:
            x = hist['Datetime'] if 'Datetime' in hist.columns else hist['Date']
            y = hist['close'] if 'close' in hist.columns else hist.iloc[:, -1]
            fig.add_trace(pgo.Scatter(x=x, y=y, mode='lines', name=name))
            plotted = True
        except Exception:
            continue

    if plotted:
        fig.update_layout(title='Commodity Price Trends', xaxis_title='Date', yaxis_title='Price', height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No historical data available for the selected commodities.')
    
    # Layout
    config_col, main_col = st.columns([1, 2])
    
    with config_col:
        st.subheader("Analysis Configuration")
        
        # Product selection
        forecast_product = st.selectbox("Select Product", [
            "Soymeal", "Groundnut Cake", "Sunflower Cake", 
            "Mustard Cake", "Husk"
        ], key='forecast_product')
        
        # Time horizon
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
        
        # Analysis options
        analysis_options = st.multiselect(
            "Select Analysis Metrics",
            ["Price Trends", "Demand Forecast", "Volatility Analysis", "Seasonality"],
            default=["Price Trends", "Demand Forecast"]
        )
        
        if st.button("Generate Market Analysis", type="primary"):
            with st.spinner("Training AI models and generating forecasts..."):
                # Generate synthetic data
                df_hist = generate_synthetic_price_data(forecast_product, days=365)
                
                # Train models
                price_model, demand_model, volatility_model, feature_data = train_price_forecast_model(df_hist)
                
                # Generate forecasts
                forecasts = forecast_market_metrics(
                    price_model, demand_model, volatility_model, 
                    feature_data, forecast_days
                )
                
                # Calculate seasonality
                seasonality = detect_seasonality(df_hist['date'], df_hist['price'])
                
                # Store in session state
                st.session_state['market_analysis'] = {
                    'historical': df_hist,
                    'forecasts': forecasts,
                    'product': forecast_product,
                    'seasonality': seasonality
                }
                
                st.success("✅ Analysis completed successfully!")
                st.rerun()
    
    with main_col:
        if 'market_analysis' in st.session_state:
            data = st.session_state['market_analysis']
            
            # Market Overview Metrics
            st.subheader("Market Overview")
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                current_price = data['historical']['price'].iloc[-1]
                forecast_price = data['forecasts']['prices'][-1]
                price_change = ((forecast_price - current_price) / current_price) * 100
                st.metric("Price Trend", 
                         f"₹{current_price:.2f}/kg",
                         f"{price_change:+.1f}%",
                         delta_color="normal")
            
            with metric_cols[1]:
                avg_demand = np.mean(data['forecasts']['demand'])
                st.metric("Projected Demand",
                         f"{avg_demand:,.0f} kg",
                         "Forecast")
            
            with metric_cols[2]:
                avg_volatility = np.mean(data['forecasts']['volatility']) * 100
                st.metric("Price Volatility",
                         f"{avg_volatility:.1f}%",
                         "Annual")
            
            with metric_cols[3]:
                peak_month = data['seasonality'].idxmax()
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                st.metric("Peak Season",
                         months[peak_month-1],
                         "Best Price")
            
            # Tabs for different visualizations
            viz_tabs = st.tabs(["📈 Price Analysis", "📊 Demand Forecast", "📉 Market Patterns"])
            
            with viz_tabs[0]:
                # Price Forecast Plot
                fig_price = go.Figure()
                
                # Historical prices
                fig_price.add_trace(go.Scatter(
                    x=data['historical']['date'],
                    y=data['historical']['price'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Forecasted prices
                fig_price.add_trace(go.Scatter(
                    x=data['forecasts']['dates'],
                    y=data['forecasts']['prices'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=6)
                ))
                
                fig_price.update_layout(
                    title=f'Price Forecast: {data["product"]}',
                    xaxis_title='Date',
                    yaxis_title='Price (₹/kg)',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
            
            with viz_tabs[1]:
                # Demand Forecast Plot
                fig_demand = go.Figure()
                
                fig_demand.add_trace(go.Scatter(
                    x=data['forecasts']['dates'],
                    y=data['forecasts']['demand'],
                    mode='lines+markers',
                    name='Projected Demand',
                    line=dict(color='green', width=2),
                    marker=dict(size=6)
                ))
                
                fig_demand.update_layout(
                    title=f'Demand Forecast: {data["product"]}',
                    xaxis_title='Date',
                    yaxis_title='Demand (kg)',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_demand, use_container_width=True)
            
            with viz_tabs[2]:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Seasonality Pattern
                    fig_season = go.Figure()
                    fig_season.add_trace(go.Scatter(
                        x=months,
                        y=data['seasonality'].values,
                        mode='lines+markers',
                        name='Seasonal Pattern',
                        line=dict(color='purple', width=2),
                        marker=dict(size=8)
                    ))
                    fig_season.update_layout(
                        title='Seasonal Price Pattern',
                        xaxis_title='Month',
                        yaxis_title='Average Price (₹/kg)',
                        height=300
                    )
                    st.plotly_chart(fig_season, use_container_width=True)
                
                with col2:
                    # Volatility Forecast
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Scatter(
                        x=data['forecasts']['dates'],
                        y=[v * 100 for v in data['forecasts']['volatility']],
                        mode='lines',
                        name='Volatility',
                        line=dict(color='orange', width=2)
                    ))
                    fig_vol.update_layout(
                        title='Price Volatility Forecast',
                        xaxis_title='Date',
                        yaxis_title='Annualized Volatility (%)',
                        height=300
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)
            
            # Detailed Forecast Table
            with st.expander("View Detailed Forecast Data"):
                forecast_df = pd.DataFrame({
                    'Date': [d.strftime('%Y-%m-%d') for d in data['forecasts']['dates']],
                    'Price (₹/kg)': [f"{p:.2f}" for p in data['forecasts']['prices']],
                    'Demand (kg)': [f"{d:,.0f}" for d in data['forecasts']['demand']],
                    'Volatility (%)': [f"{v*100:.1f}%" for v in data['forecasts']['volatility']]
                })
                st.dataframe(forecast_df, use_container_width=True)
            
            # Market Insights
            st.subheader("AI-Generated Market Insights")
            
            # Calculate insights
            price_trend = "upward" if price_change > 0 else "downward"
            volatility_level = "high" if avg_volatility > 20 else "moderate" if avg_volatility > 10 else "low"
            demand_trend = "strong" if avg_demand > 1000 else "moderate"
            
            insights = [
                f"🔹 Price Trend: {price_change:+.1f}% {price_trend} trend expected over the next {forecast_days} days",
                f"� Market Volatility: {volatility_level.title()} volatility environment ({avg_volatility:.1f}% annualized)",
                f"🔹 Demand Outlook: {demand_trend.title()} demand with projected volume of {avg_demand:,.0f} kg",
                f"🔹 Seasonal Pattern: Best pricing opportunities in {months[peak_month-1]}"
            ]
            
            for insight in insights:
                st.markdown(insight)
            
        else:
            st.info("👈 Configure and run the analysis to see market insights")

# ============================================================================
# TAB 4: BLOCKCHAIN LEDGER & SMART CONTRACTS
# ============================================================================

with tab4:
    st.header("🔐 Blockchain Ledger & Smart Contracts")
    
    # Create tabs for different blockchain features
    ledger_tab, contracts_tab, verify_tab = st.tabs([
        "📒 Transaction Ledger",
        "📝 Smart Contracts",
        "🔍 Verify & Audit"
    ])
    
    with ledger_tab:
        st.subheader("Blockchain Transaction Ledger")
        st.markdown("**Security:** SHA256 hashing with smart contract integration")
        
        session = Session()
        transactions = session.query(Transaction).order_by(Transaction.timestamp.desc()).all()
        
        if transactions:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", len(transactions))
            with col2:
                total_volume = sum([t.quantity for t in transactions])
                st.metric("Total Volume", f"{total_volume:,.0f} kg")
            with col3:
                total_value = sum([t.price for t in transactions])
                st.metric("Total Value", f"₹{total_value:,.0f}")
            with col4:
                contract_txs = len([t for t in transactions if t.tx_type == 'smart_contract_execution'])
                st.metric("Smart Contract Executions", contract_txs)
            
            # Transaction table
            st.markdown("### Recent Transactions")
            tx_data = []
            for tx in transactions[:50]:
                tx_data.append({
                    'TX Hash': tx.tx_hash[:16] + '...',
                    'Type': tx.tx_type.upper(),
                    'Product': tx.product,
                    'Seller': tx.seller,
                    'Buyer': tx.buyer,
                    'Quantity (kg)': f"{tx.quantity:,.0f}",
                    'Value (₹)': f"{tx.price:,.2f}",
                    'Timestamp': tx.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            df_tx = pd.DataFrame(tx_data)
            st.dataframe(df_tx, use_container_width=True)
        else:
            st.info("No transactions recorded yet.")
    
    with contracts_tab:
        st.subheader("Smart Contracts Platform")
        
        contract_col1, contract_col2 = st.columns([1, 1])
        
        with contract_col1:
            st.markdown("### Create Smart Contract")
            
            with st.form("smart_contract_form"):
                contract_type = st.selectbox(
                    "Contract Type",
                    ["forward", "spot", "option"],
                    help="Forward: Future delivery, Spot: Immediate, Option: Right to buy"
                )
                
                seller = st.text_input("Seller Name")
                buyer = st.text_input("Buyer Name")
                product = st.selectbox(
                    "Product",
                    ["Soymeal", "Groundnut Cake", "Sunflower Cake", "Mustard Cake", "Husk"]
                )
                quantity = st.number_input("Quantity (kg)", min_value=100.0, value=1000.0)
                price_per_unit = st.number_input("Price per kg (₹)", min_value=1.0, value=35.0)
                
                quality_requirements = st.multiselect(
                    "Quality Requirements",
                    ["Premium Grade", "Moisture < 12%", "Protein > 45%", "Oil content > 1%", 
                     "No contamination", "ISO 22000 certified"],
                    default=["Premium Grade", "No contamination"]
                )
                
                delivery_date = st.date_input(
                    "Delivery Date",
                    min_value=datetime.now().date(),
                    value=(datetime.now() + timedelta(days=30)).date()
                )
                
                payment_terms = {
                    "advance_payment": st.slider("Advance Payment (%)", 0, 100, 20),
                    "payment_days": st.number_input("Payment Days after Delivery", 0, 90, 30),
                    "penalty_rate": st.slider("Late Payment Penalty (% per month)", 0.0, 5.0, 1.0)
                }
                
                submit_contract = st.form_submit_button("Create Smart Contract")
                
                if submit_contract:
                    if seller and buyer:
                        contract_id, contract_hash = create_smart_contract(
                            contract_type, seller, buyer, product, quantity,
                            price_per_unit, quality_requirements,
                            datetime.combine(delivery_date, datetime.min.time()),
                            payment_terms
                        )
                        st.success(f"Smart Contract created successfully! Contract ID: {contract_id}")
                        st.code(f"Contract Hash:\n{contract_hash}")
                    else:
                        st.error("Please fill in all required fields")
        
        with contract_col2:
            st.markdown("### Active Smart Contracts")
            
            session = Session()
            active_contracts = session.query(SmartContract).filter(
                SmartContract.status.in_(['pending', 'active'])
            ).order_by(SmartContract.created_at.desc()).all()
            
            if active_contracts:
                for contract in active_contracts:
                    with st.expander(f"📄 {contract.product} - {contract.contract_type.upper()}"):
                        st.markdown(f"**Status:** {contract.status.upper()}")
                        st.markdown(f"**Seller:** {contract.seller}")
                        st.markdown(f"**Buyer:** {contract.buyer}")
                        st.markdown(f"**Quantity:** {contract.quantity:,} kg")
                        st.markdown(f"**Value:** ₹{contract.quantity * contract.price_per_unit:,.2f}")
                        
                        # Show signatures
                        signatures = json.loads(contract.signatures)
                        signed_by = list(signatures.keys())
                        
                        if contract.status == 'pending':
                            if 'user' in st.session_state and st.session_state['user']:
                                user = st.session_state['user']['username']
                                if user in [contract.seller, contract.buyer] and user not in signed_by:
                                    if st.button(f"Sign Contract", key=f"sign_{contract.id}"):
                                        success, msg = sign_smart_contract(contract.id, user)
                                        if success:
                                            st.success("Contract signed successfully!")
                                            st.rerun()
                                        else:
                                            st.error(msg)
                        
                        # Execute contract if fully signed
                        if contract.status == 'active':
                            if st.button(f"Execute Contract", key=f"exec_{contract.id}"):
                                success, result = execute_smart_contract(contract.id)
                                if success:
                                    st.success(f"Contract executed successfully! TX: {result}")
                                    st.rerun()
                                else:
                                    st.error(f"Execution failed: {result}")
            else:
                st.info("No active smart contracts")
            
            session.close()
    
    with verify_tab:
        st.subheader("Blockchain Verification & Audit")
        
        verify_col1, verify_col2 = st.columns([1, 1])
        
        with verify_col1:
            st.markdown("### Verify Transaction")
            tx_hash = st.text_input("Enter Transaction Hash")
            
            if tx_hash:
                session = Session()
                tx = session.query(Transaction).filter_by(tx_hash=tx_hash).first()
                
                if tx:
                    st.success("Transaction found! Verifying...")
                    
                    # Verify hash
                    payload = json.loads(tx.payload)
                    recomputed_hash = hash_transaction(payload)
                    
                    if recomputed_hash == tx.tx_hash:
                        st.markdown("✅ **Hash Verification:** PASSED")
                    else:
                        st.markdown("❌ **Hash Verification:** FAILED")
                    
                    # Show details
                    st.json(payload)
                else:
                    st.error("Transaction not found")
                
                session.close()
        
        with verify_col2:
            st.markdown("### Verify Smart Contract")
            contract_hash = st.text_input("Enter Contract Hash")
            
            if contract_hash:
                verification = verify_contract(contract_hash)
                
                if verification:
                    st.success("Contract found! Verifying...")
                    
                    # Show verification results
                    st.markdown(f"**Contract Valid:** {'✅' if verification['contract_valid'] else '❌'}")
                    st.markdown(f"**Status:** {verification['status'].upper()}")
                    st.markdown(f"**Created:** {verification['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    with st.expander("View Signatures"):
                        st.json(verification['signatures'])
                    
                    with st.expander("View Hash Details"):
                        st.code(f"Stored Hash:\n{verification['conditions_hash']}")
                        st.code(f"Computed Hash:\n{verification['computed_hash']}")
                else:
                    st.error("Contract not found")
        
        # Blockchain Statistics
        st.markdown("### Blockchain Statistics")
        session = Session()
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            total_blocks = session.query(Transaction).count()
            st.metric("Total Blocks", total_blocks)
        
        with stats_col2:
            total_contracts = session.query(SmartContract).count()
            st.metric("Total Contracts", total_contracts)
        
        with stats_col3:
            active_contracts = session.query(SmartContract).filter_by(status='active').count()
            st.metric("Active Contracts", active_contracts)
        
        with stats_col4:
            completed_contracts = session.query(SmartContract).filter_by(status='completed').count()
            st.metric("Completed Contracts", completed_contracts)
        
        session.close()

# ============================================================================
# TAB 5: EXPORT MATCHMAKING
# ============================================================================

with tab5:
    st.header("🌍 Export Opportunity Matchmaking")
    st.markdown("**AI-Powered Global Buyer Discovery**")
    
    # Mock global buyer database
    global_buyers = [
        {'name': 'FeedCorp International', 'country': 'UAE', 'products': ['Soymeal', 'Groundnut Cake'], 'volume': 5000, 'price_range': '35-45'},
        {'name': 'AgriGlobal Ltd', 'country': 'Bangladesh', 'products': ['Mustard Cake', 'Sunflower Cake'], 'volume': 3000, 'price_range': '38-48'},
        {'name': 'EuroFeed Solutions', 'country': 'Netherlands', 'products': ['Soymeal', 'Husk'], 'volume': 10000, 'price_range': '40-50'},
        {'name': 'Asia Pacific Traders', 'country': 'Vietnam', 'products': ['Groundnut Cake', 'Soymeal'], 'volume': 7000, 'price_range': '32-42'},
        {'name': 'Middle East Feed Co', 'country': 'Saudi Arabia', 'products': ['Sunflower Cake', 'Mustard Cake'], 'volume': 4000, 'price_range': '36-46'},
        {'name': 'African AgriHub', 'country': 'Kenya', 'products': ['Soymeal', 'Groundnut Cake', 'Husk'], 'volume': 6000, 'price_range': '30-40'}
    ]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Your Export Profile")
        export_product = st.selectbox("Product to Export", [
            "Soymeal", "Groundnut Cake", "Sunflower Cake", 
            "Mustard Cake", "Husk"
        ], key='export_product')
        export_quantity = st.number_input("Available Quantity (kg)", min_value=1000, value=5000, step=1000)
        export_quality = st.selectbox("Quality Grade", ["Premium", "Grade A", "Grade B", "Standard"])
        export_location = st.text_input("Export Location", "Mumbai Port")
        
        if st.button("Find Export Matches"):
            # Filter buyers
            matches = [b for b in global_buyers if export_product in b['products']]
            st.session_state['export_matches'] = matches
            st.success(f"✅ Found {len(matches)} potential buyers!")
    
    with col2:
        st.subheader("Matched International Buyers")
        
        if 'export_matches' in st.session_state and st.session_state['export_matches']:
            for buyer in st.session_state['export_matches']:
                with st.expander(f"🌐 {buyer['name']} - {buyer['country']}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Products:** {', '.join(buyer['products'])}")
                        st.write(f"**Monthly Volume:** {buyer['volume']:,} kg")
                    with col_b:
                        st.write(f"**Price Range:** ₹{buyer['price_range']}/kg")
                        st.write(f"**Location:** {buyer['country']}")
                    
                    st.info("📋 **Next Steps:** Contact buyer via platform, negotiate terms, arrange logistics (freight, customs, phytosanitary certification)")
                    
                    if st.button(f"Initiate Contact", key=f"contact_{buyer['name']}"):
                        st.success(f"✅ Contact request sent to {buyer['name']}!")
        else:
            st.info("👈 Enter your export details to find international buyers")
        
        st.subheader("Export Compliance Checklist")
        st.markdown("""
        - ✅ Phytosanitary Certificate
        - ✅ Export License from DGFT
        - ✅ Quality Certification (FSSAI/ISO)
        - ✅ Fumigation Certificate
        - ✅ Commercial Invoice & Packing List
        - ✅ Bill of Lading / Airway Bill
        - ✅ Certificate of Origin
        """)

# ============================================================================
# TAB 6: ROADMAP
# ============================================================================

with tab6:
    st.header("🗺️ Development Roadmap")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Phase 1: MVP (Current)")
        st.markdown("""
        **Status:** Completed
        
        **Features:**
        - Digital marketplace for oilseed by-products
        - IoT device registration and telemetry simulation
        - Blockchain-style transaction ledger with SHA256 hashing
        - AI-powered price forecasting using RandomForest
        - Export matchmaking with global buyer database
        - SQLite database for local data storage
        - Interactive Streamlit UI with real-time updates
        
        **Technology Stack:**
        - Python 3.11
        - Streamlit for UI
        - SQLAlchemy + SQLite for database
        - Scikit-learn for ML models
        - Plotly for data visualization
        - PyCryptodome for hashing
        """)
        
        st.subheader("🚧 Phase 2: IoT Integration")
        st.markdown("""
        **Timeline:** 2-3 months
        
        **Objectives:**
        - Integrate real IoT devices via MQTT protocol
        - Implement HTTP REST APIs for device communication
        - Add real-time streaming data pipeline
        - Deploy edge computing for local processing
        - Integrate with popular IoT platforms (AWS IoT, Azure IoT Hub)
        - Add device authentication and security
        
        **Hardware Partners:**
        - ESP32/ESP8266 for temperature/humidity sensors
        - Raspberry Pi for gateway devices
        - Industrial PLCs for production monitoring
        """)
    
    with col2:
        st.subheader("🔗 Phase 3: Blockchain Integration")
        st.markdown("""
        **Timeline:** 3-4 months
        
        **Objectives:**
        - Migrate to Hyperledger Fabric for permissioned blockchain
        - Implement smart contracts for automated trade execution
        - Add multi-signature wallets for secure payments
        - Integrate with Ethereum testnet for tokenization
        - Develop consensus mechanism for transaction validation
        - Add blockchain explorer for transparency
        
        **Benefits:**
        - True immutability and decentralization
        - Automated escrow and payment settlement
        - Reduced transaction costs
        - Enhanced trust and transparency
        """)
        
        st.subheader("☁️ Phase 4: Production Deployment")
        st.markdown("""
        **Timeline:** 4-6 months
        
        **Objectives:**
        - Deploy on national cloud infrastructure (NIC Cloud / AWS)
        - Implement user authentication (OAuth 2.0, JWT)
        - Add role-based access control (farmers, buyers, processors)
        - Integrate payment gateways (UPI, NEFT, international)
        - Deploy advanced ML models (LSTM, Prophet, ARIMA)
        - Add mobile app (React Native / Flutter)
        - Implement real-time notifications (SMS, Email, Push)
        - Scale to handle 100,000+ concurrent users
        
        **Compliance:**
        - GDPR and data privacy compliance
        - Agricultural marketing regulations
        - Export/import compliance automation
        - Financial transaction security (PCI-DSS)
        """)
    
    st.divider()
    
    st.subheader("📊 Technical Architecture Evolution")
    
    arch_col1, arch_col2, arch_col3 = st.columns(3)
    
    with arch_col1:
        st.markdown("""
        **Current (MVP)**
        ```
        ┌─────────────┐
        │  Streamlit  │
        │     UI      │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │   SQLite    │
        │  Database   │
        └─────────────┘
        ```
        """)
    
    with arch_col2:
        st.markdown("""
        **Phase 3**
        ```
        ┌─────────────┐
        │   Web UI    │
        │  Mobile App │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │  API Layer  │
        │  (FastAPI)  │
        └──┬───────┬──┘
           │       │
    ┌──────▼──┐ ┌─▼──────┐
    │PostgreSQL│ │Blockchain│
    │ Database │ │ Ledger  │
    └──────────┘ └─────────┘
        ```
        """)
    
    with arch_col3:
        st.markdown("""
        **Phase 4**
        ```
        ┌──────────────┐
        │ Load Balancer│
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │  Microservices│
        │ (K8s Cluster)│
        └─┬─────┬─────┬┘
          │     │     │
    ┌─────▼┐ ┌─▼───┐ ┌▼────┐
    │ DB   │ │Cache│ │MQTT │
    │Cluster│ │Redis│ │Broker│
    └──────┘ └─────┘ └─────┘
        ```
        """)
    
    st.divider()
    
    st.subheader("🎯 Success Metrics (Phase 4)")
    
    metrics = {
        'Active Users': '100,000+',
        'Daily Transactions': '10,000+',
        'IoT Devices Connected': '50,000+',
        'Trade Volume': '₹500 Cr/month',
        'Farmer Income Increase': '+25%',
        'Market Efficiency': '+40%'
    }
    
    metric_cols = st.columns(len(metrics))
    for col, (metric, value) in zip(metric_cols, metrics.items()):
        with col:
            st.metric(metric, value)
    
    st.success("🚀 **Vision:** Transform India's oilseed value chain into a digitally-enabled, transparent, and efficient ecosystem that maximizes value for all stakeholders while reducing waste and information asymmetry.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Value Chain Integration System for Oilseed By-Products</strong></p>
    <p>Powered by AI/ML | Secured by Blockchain | Connected by IoT</p>
    <p><em>Built with: Python • Streamlit • SQLAlchemy • Scikit-learn • Plotly • PyCryptodome</em></p>
</div>
""", unsafe_allow_html=True)
