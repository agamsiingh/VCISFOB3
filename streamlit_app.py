"""
Value Chain Integration System for Oilseed By-Products
A production-ready AgriTech platform for marketplace, IoT, blockchain-style transactions, and AI forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from Crypto.Hash import SHA256
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import json

# Database Setup
Base = declarative_base()
engine = create_engine('sqlite:///valuechain.db', echo=False)
Session = sessionmaker(bind=engine)


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
    timestamp = Column(DateTime, default=datetime.now)


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


# Initialize database
Base.metadata.create_all(engine)


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

# Create any new tables (idempotent)
Base.metadata.create_all(engine)


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
    session = Session()
    user = session.query(User).filter_by(username=username).first()
    session.close()
    if not user:
        return None
    if user.password_hash == hash_password(password):
        return {'username': user.username, 'role': user.role}
    return None


# ============================================================================
# HELPER FUNCTIONS - BLOCKCHAIN SIMULATION
# ============================================================================

def hash_transaction(payload):
    """
    Generate SHA256 hash for transaction payload to simulate blockchain immutability.
    
    Args:
        payload (dict): Transaction data to hash
    
    Returns:
        str: 64-character hexadecimal hash
    """
    payload_str = json.dumps(payload, sort_keys=True)
    hash_obj = SHA256.new(payload_str.encode('utf-8'))
    return hash_obj.hexdigest()


def log_transaction(tx_type, seller, buyer, product, quantity, price):
    """
    Record a transaction with blockchain-style hashing for proof of immutability.
    
    Args:
        tx_type (str): Type of transaction (e.g., 'sale', 'listing')
        seller (str): Seller name
        buyer (str): Buyer name
        product (str): Product name
        quantity (float): Quantity in kg
        price (float): Total price
    
    Returns:
        str: Transaction hash
    """
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
# MACHINE LEARNING - PRICE FORECASTING
# ============================================================================

def train_price_forecast_model(df):
    """
    Train RandomForest model for price forecasting using historical data.
    
    Args:
        df (pd.DataFrame): Historical price data with 'date' and 'price' columns
    
    Returns:
        tuple: (trained model, feature scaler info)
    """
    df = df.copy()
    df['days'] = (df['date'] - df['date'].min()).dt.days
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    X = df[['days', 'day_of_week', 'month']].values
    y = df['price'].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    return model, df['days'].max()


def forecast_prices(model, last_day, days_ahead=30):
    """
    Generate price forecasts for next N days using trained model.
    
    Args:
        model: Trained sklearn model
        last_day (int): Last day number in training data
        days_ahead (int): Number of days to forecast
    
    Returns:
        tuple: (forecast_dates, forecast_prices)
    """
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1)
    future_dates = [datetime.now() + timedelta(days=x) for x in range(1, days_ahead + 1)]
    
    X_future = []
    for i, date in enumerate(future_dates):
        X_future.append([future_days[i], date.weekday(), date.month])
    
    X_future = np.array(X_future)
    predictions = model.predict(X_future)
    
    return future_dates, predictions


# ============================================================================
# STREAMLIT UI CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AgriTech Value Chain Integration",
    page_icon="🌾",
    layout="wide"
)


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
tab_css = """
<style>
/* Basic transition + spacing for tab buttons */
div[role="tablist"] > button[role="tab"] {
    transition: transform 180ms ease, box-shadow 180ms ease, background-color 180ms ease;
    border-radius: 8px;
    padding: 6px 12px;
}

/* Hover: slight lift and soft shadow */
div[role="tablist"] > button[role="tab"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
}

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

st.title("🌾 Value Chain Integration System for Oilseed By-Products")
st.markdown("**AI-Powered Marketplace | IoT Integration | Blockchain Transactions | Predictive Analytics**")

# ------------------------
# Sidebar: Demo Login/Register
# ------------------------
st.sidebar.header("Login / Register (Demo)")

if 'user' not in st.session_state:
    st.session_state['user'] = None

with st.sidebar.expander("Register (demo)"):
    reg_username = st.text_input("Username", key='reg_user')
    reg_password = st.text_input("Password", type='password', key='reg_pass')
    reg_role = st.selectbox("Role", ["Farmer", "Processor", "Buyer"], key='reg_role')
    if st.button("Create Account", key='reg_btn'):
        if reg_username and reg_password:
            ok = create_user(reg_username, reg_password, reg_role)
            if ok:
                st.success(f"Account created for {reg_username} ({reg_role}). Please login below.")
            else:
                st.warning("Username already exists. Choose another.")
        else:
            st.warning("Enter username and password to register.")

with st.sidebar.expander("Login"):
    login_username = st.text_input("Username", key='login_user')
    login_password = st.text_input("Password", type='password', key='login_pass')
    if st.button("Login", key='login_btn'):
        auth = authenticate_user(login_username, login_password)
        if auth:
            st.session_state['user'] = auth
            safe_rerun()
        else:
            st.error("Invalid credentials.")

if st.session_state.get('user'):
    u = st.session_state['user']
    st.sidebar.markdown(f"**Signed in:** `{u['username']}`  \n**Role:** {u['role']}")
    if st.sidebar.button("Logout"):
        st.session_state['user'] = None
        safe_rerun()
else:
    st.sidebar.info("Register or login to enable role-based dashboards. Or use demo accounts by registering quickly.")

# ============================================================================
# TAB STRUCTURE
# ============================================================================

tab_dashboard, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👤 Dashboard",
    "� Marketplace",
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
        col_a, col_b, col_c = st.columns(3)
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
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Register IoT Device")
        with st.form("device_form"):
            device_id = st.text_input("Device ID", f"IOT-{np.random.randint(1000, 9999)}")
            device_type = st.selectbox("Device Type", [
                "Temperature Sensor",
                "Humidity Sensor",
                "Production Monitor",
                "Quality Analyzer"
            ])
            device_location = st.text_input("Location", "Processing Unit A")
            
            register = st.form_submit_button("Register Device")
            
            if register:
                session = Session()
                existing = session.query(Device).filter_by(device_id=device_id).first()
                if not existing:
                    device = Device(
                        device_id=device_id,
                        device_type=device_type,
                        location=device_location,
                        temperature=0,
                        humidity=0,
                        production_rate=0
                    )
                    session.add(device)
                    session.commit()
                    st.success(f"✅ Device {device_id} registered!")
                else:
                    st.warning("Device ID already exists!")
                session.close()
        
        st.subheader("Simulate Telemetry")
        if st.button("Generate Live Data"):
            session = Session()
            devices = session.query(Device).all()
            
            for device in devices:
                telemetry = generate_iot_telemetry()
                device.temperature = telemetry['temperature']
                device.humidity = telemetry['humidity']
                device.production_rate = telemetry['production_rate']
                device.timestamp = datetime.now()
            
            session.commit()
            session.close()
            st.success("✅ Telemetry data updated!")
            st.rerun()
    
    with col2:
        st.subheader("Live Device Dashboard")
        session = Session()
        devices = session.query(Device).all()
        
        if devices:
            # Create real-time metrics
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                avg_temp = np.mean([d.temperature for d in devices if d.temperature])
                st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
            with metrics_cols[1]:
                avg_humidity = np.mean([d.humidity for d in devices if d.humidity])
                st.metric("Avg Humidity", f"{avg_humidity:.1f}%")
            with metrics_cols[2]:
                total_production = sum([d.production_rate for d in devices if d.production_rate])
                st.metric("Total Production", f"{total_production:.0f} kg/hr")
            
            # Device table
            device_data = [{
                'Device ID': d.device_id,
                'Type': d.device_type,
                'Location': d.location,
                'Temp (°C)': d.temperature,
                'Humidity (%)': d.humidity,
                'Production (kg/hr)': d.production_rate,
                'Last Update': d.timestamp.strftime('%H:%M:%S') if d.timestamp else 'N/A'
            } for d in devices]
            
            df_devices = pd.DataFrame(device_data)
            st.dataframe(df_devices, use_container_width=True)
            
            # Visualization
            if len(devices) > 0:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[d.device_id for d in devices],
                    y=[d.temperature for d in devices],
                    name='Temperature (°C)',
                    marker_color='indianred'
                ))
                fig.add_trace(go.Bar(
                    x=[d.device_id for d in devices],
                    y=[d.humidity for d in devices],
                    name='Humidity (%)',
                    marker_color='lightsalmon'
                ))
                fig.update_layout(
                    title='Device Metrics Comparison',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No devices registered. Register your first IoT device!")
        
        session.close()

# ============================================================================
# TAB 3: AI FORECASTING
# ============================================================================

with tab3:
    st.header("AI-Powered Price Forecasting")
    st.markdown("**Machine Learning Model:** RandomForestRegressor with synthetic historical data")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Forecast Configuration")
        forecast_product = st.selectbox("Select Product for Forecast", [
            "Soymeal", "Groundnut Cake", "Sunflower Cake", 
            "Mustard Cake", "Husk"
        ], key='forecast_product')
        
        forecast_days = st.slider("Forecast Horizon (days)", 7, 60, 30)
        
        if st.button("Train Model & Generate Forecast"):
            with st.spinner("Training RandomForest model..."):
                # Generate synthetic data
                df_hist = generate_synthetic_price_data(forecast_product, days=365)
                
                # Train model
                model, last_day = train_price_forecast_model(df_hist)
                
                # Generate forecast
                future_dates, predictions = forecast_prices(model, last_day, forecast_days)
                
                # Store in session state
                st.session_state['forecast_data'] = {
                    'historical': df_hist,
                    'future_dates': future_dates,
                    'predictions': predictions,
                    'product': forecast_product
                }
                
                st.success("✅ Model trained successfully!")
                st.rerun()
    
    with col2:
        st.subheader("Price Forecast Visualization")
        
        if 'forecast_data' in st.session_state:
            data = st.session_state['forecast_data']
            
            # Create visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=data['historical']['date'],
                y=data['historical']['price'],
                mode='lines',
                name='Historical Prices',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=data['future_dates'],
                y=data['predictions'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f'Price Forecast: {data["product"]}',
                xaxis_title='Date',
                yaxis_title='Price (₹/kg)',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            with stats_col1:
                current_price = data['historical']['price'].iloc[-1]
                st.metric("Current Price", f"₹{current_price:.2f}/kg")
            with stats_col2:
                forecast_avg = np.mean(data['predictions'])
                st.metric("Forecast Avg", f"₹{forecast_avg:.2f}/kg")
            with stats_col3:
                price_change = ((forecast_avg - current_price) / current_price) * 100
                st.metric("Expected Change", f"{price_change:+.1f}%")
            
            # Forecast table
            st.subheader("Detailed Forecast")
            forecast_df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in data['future_dates']],
                'Predicted Price (₹/kg)': [f"{p:.2f}" for p in data['predictions']]
            })
            st.dataframe(forecast_df, use_container_width=True)
            
        else:
            st.info("👈 Configure and train the model to see forecasts")

# ============================================================================
# TAB 4: TRANSACTION LEDGER
# ============================================================================

with tab4:
    st.header("🔐 Blockchain-Style Transaction Ledger")
    st.markdown("**Security:** SHA256 hashing ensures transaction immutability")
    
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
            sales_count = len([t for t in transactions if t.tx_type == 'sale'])
            st.metric("Completed Sales", sales_count)
        
        st.subheader("Recent Transactions")
        
        # Transaction table
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
        
        # Transaction verification tool
        st.subheader("Verify Transaction Integrity")
        selected_tx = st.selectbox("Select Transaction to Verify", 
                                    [f"{t.tx_hash[:16]}... ({t.product})" for t in transactions[:20]])
        
        if selected_tx:
            tx_hash_prefix = selected_tx.split('...')[0]
            tx = [t for t in transactions if t.tx_hash.startswith(tx_hash_prefix)][0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.code(f"Full Hash:\n{tx.tx_hash}", language='text')
                st.json(json.loads(tx.payload))
            with col2:
                # Re-compute hash to verify
                recomputed_hash = hash_transaction(json.loads(tx.payload))
                if recomputed_hash == tx.tx_hash:
                    st.success("✅ Transaction verified! Hash matches.")
                else:
                    st.error("❌ Hash mismatch! Transaction may be tampered.")
                
                st.info("**Blockchain Simulation:** Each transaction is hashed using SHA256, creating an immutable record that can be verified at any time.")
    else:
        st.info("No transactions yet. Create a listing or make a purchase to see transactions.")
    
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
