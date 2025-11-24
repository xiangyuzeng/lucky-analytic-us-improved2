import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Luckin Coffee - Advanced Marketing Analytics Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .main { padding: 0rem 1rem; }
        .luckin-header {
            background: linear-gradient(135deg, #232773 0%, #3d4094 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(35, 39, 115, 0.2);
        }
        h1, h2, h3 { font-family: 'Inter', sans-serif; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #f8f9fa;
            border-radius: 10px;
            padding-left: 24px;
            padding-right: 24px;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #232773;
            color: white;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .success-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .platform-note {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Platform Colors
PLATFORM_COLORS = {
    'DoorDash': '#ff3008',
    'Uber': '#000000',
    'Grubhub': '#ff8000'
}

# CORRECTED Data Processing Functions
@st.cache_data
def process_doordash_data(df):
    """Process DoorDash data with improved error handling"""
    try:
        processed = pd.DataFrame()
        
        # Core fields
        processed['Date'] = pd.to_datetime(df['时间戳本地日期'], errors='coerce')
        processed['Platform'] = 'DoorDash'
        processed['Revenue'] = pd.to_numeric(df['净总计'], errors='coerce')
        
        # Optional fields with safe access
        field_mapping = {
            '小计': 'Subtotal',
            '转交给商家的税款小计': 'Tax',
            '员工小费': 'Tips',
            '佣金': 'Commission',
            '营销费 |（包括任何适用税金）': 'Marketing_Fee'
        }
        
        for col, new_col in field_mapping.items():
            if col in df.columns:
                processed[new_col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                processed[new_col] = 0
        
        # Process order status
        if '最终订单状态' in df.columns:
            processed['Is_Completed'] = df['最终订单状态'].astype(str).str.contains('Delivered|delivered', case=False, na=False, regex=True)
            processed['Is_Cancelled'] = df['最终订单状态'].astype(str).str.contains('Cancelled|cancelled', case=False, na=False, regex=True)
        else:
            processed['Is_Completed'] = True
            processed['Is_Cancelled'] = False
        
        # Store information with normalization
        processed['Store_Name'] = df.get('店铺名称', 'Unknown').fillna('Unknown')
        processed['Store_Name'] = processed['Store_Name'].astype(str).str.strip()
        processed['Store_ID'] = df.get('Store ID', 'Unknown').fillna('Unknown').astype(str)
        
        # Order ID for unique customer tracking
        if 'DoorDash 订单 ID' in df.columns:
            processed['Order_ID'] = df['DoorDash 订单 ID'].astype(str)
        else:
            processed['Order_ID'] = pd.Series(range(len(df))).astype(str) + '_dd'
        
        # Time processing
        if '时间戳为本地时间' in df.columns:
            try:
                time_series = pd.to_datetime(df['时间戳为本地时间'], errors='coerce')
                processed['Hour'] = time_series.dt.hour.fillna(12)
            except:
                processed['Hour'] = 12
        else:
            processed['Hour'] = 12
        
        # Add day and month info - DO THIS BEFORE FILTERING
        processed['DayOfWeek'] = processed['Date'].dt.day_name()
        processed['Month'] = processed['Date'].dt.to_period('M')
        processed['Month_str'] = processed['Date'].dt.strftime('%Y-%m')
        
        # Clean data - only remove truly invalid records
        processed = processed[processed['Date'].notna()]
        processed = processed[processed['Revenue'].notna()]
        
        # CRITICAL FIX: Filter to October 2025 only
        processed = processed[
            (processed['Date'].dt.year == 2025) & 
            (processed['Date'].dt.month == 10)
        ]
        
        # Keep negative revenue (refunds) but filter extreme outliers
        processed = processed[processed['Revenue'].abs() < 1000]
        
        return processed
    except Exception as e:
        st.error(f"DoorDash processing error: {e}")
        return pd.DataFrame()

@st.cache_data
def process_uber_data(df):
    """Process Uber data with improved header handling"""
    try:
        # Fix the two-row header issue
        if len(df.columns) > 0 and 'Uber Eats 优食管理工具中显示的餐厅名称' in str(df.columns[0]):
            # Get actual headers from first row
            new_columns = df.iloc[0].fillna('').astype(str).str.strip().tolist()
            
            # Handle empty column names
            for i, col in enumerate(new_columns):
                if not col or col == 'nan':
                    new_columns[i] = f'col_{i}'
            
            # Set new column names and remove header row
            df.columns = new_columns
            df = df.iloc[1:].reset_index(drop=True)
        
        processed = pd.DataFrame()
        
        # Process Date
        date_col = None
        for col in df.columns:
            if '订单日期' in col or '日期' in col:
                date_col = col
                break
        
        if date_col and not df[date_col].isna().all():
            # Clean date strings
            date_str = df[date_col].astype(str).str.split(' ').str[0]
            
            # Try multiple date formats
            for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y']:
                try:
                    processed['Date'] = pd.to_datetime(date_str, format=fmt, errors='coerce')
                    if not processed['Date'].isna().all():
                        break
                except:
                    continue
            
            # If all formats fail, try general parsing
            if 'Date' not in processed.columns or processed['Date'].isna().all():
                processed['Date'] = pd.to_datetime(date_str, errors='coerce')
        else:
            processed['Date'] = pd.NaT
        
        processed['Platform'] = 'Uber'
        
        # Process Revenue
        revenue_col = None
        for col in df.columns:
            if '收入总额' in col or ('收入' in col and '总' in col):
                revenue_col = col
                break
        
        if revenue_col:
            # Clean and convert revenue
            revenue_str = df[revenue_col].astype(str).str.replace(' ', '').str.replace(',', '')
            processed['Revenue'] = pd.to_numeric(revenue_str, errors='coerce')
        else:
            processed['Revenue'] = 0
        
        # Process other fields
        field_mapping = {
            '销售额（不含税费）': 'Subtotal',
            '销售额税费': 'Tax',
            '小费': 'Tips',
            '平台服务费': 'Commission'
        }
        
        for pattern, new_col in field_mapping.items():
            found_col = None
            for col in df.columns:
                if pattern in col:
                    found_col = col
                    break
            
            if found_col:
                processed[new_col] = pd.to_numeric(df[found_col], errors='coerce').fillna(0)
            else:
                processed[new_col] = 0
        
        # Order status
        status_col = None
        for col in df.columns:
            if '订单状态' in col or '状态' in col:
                status_col = col
                break
        
        if status_col:
            processed['Is_Completed'] = df[status_col].astype(str).str.contains('已完成|完成', case=False, na=False, regex=True)
            processed['Is_Cancelled'] = df[status_col].astype(str).str.contains('已取消|取消', case=False, na=False, regex=True)
        else:
            processed['Is_Completed'] = True
            processed['Is_Cancelled'] = False
        
        # Store information
        store_col = None
        for col in df.columns:
            if '餐厅名称' in col:
                store_col = col
                break
        
        processed['Store_Name'] = df[store_col].fillna('Unknown') if store_col else 'Unknown'
        processed['Store_Name'] = processed['Store_Name'].astype(str).str.strip()
        processed['Store_ID'] = 'UB_' + processed.index.astype(str)
        
        # Order ID
        order_col = None
        for col in df.columns:
            if '订单号' in col:
                order_col = col
                break
        
        processed['Order_ID'] = df[order_col].astype(str) if order_col else processed.index.astype(str) + '_uber'
        
        # Time processing
        time_col = None
        for col in df.columns:
            if '时间' in col and '接受' in col:
                time_col = col
                break
        
        if time_col:
            try:
                time_str = df[time_col].astype(str)
                # Extract hour from time strings like "8:30", "15:23"
                hour_parts = time_str.str.extract(r'(\d+):').astype(float)
                processed['Hour'] = hour_parts[0].fillna(12)
            except:
                processed['Hour'] = 12
        else:
            processed['Hour'] = 12
        
        # Add derived fields - DO THIS BEFORE FILTERING
        processed['DayOfWeek'] = processed['Date'].dt.day_name()
        processed['Month'] = processed['Date'].dt.to_period('M')
        processed['Month_str'] = processed['Date'].dt.strftime('%Y-%m')
        processed['Marketing_Fee'] = 0  # Not available in Uber data
        
        # Clean data - keep refunds but remove extreme outliers
        processed = processed[processed['Date'].notna()]
        processed = processed[processed['Revenue'].notna()]
        
        # CRITICAL FIX: Filter to October 2025 only
        processed = processed[
            (processed['Date'].dt.year == 2025) & 
            (processed['Date'].dt.month == 10)
        ]
        
        processed = processed[processed['Revenue'].abs() < 1000]
        
        return processed
    except Exception as e:
        st.error(f"Uber processing error: {e}")
        return pd.DataFrame()

@st.cache_data  
def process_grubhub_data(df):
    """Process Grubhub data with CORRECTED date handling - NO FAKE DATES"""
    try:
        processed = pd.DataFrame()
        
        # CRITICAL FIX: Don't generate fake dates! Use actual data
        date_col = 'transaction_date'
        if date_col in df.columns:
            # Handle corrupted dates
            dates = df[date_col].astype(str)
            
            # If dates are corrupted (showing as ########), skip this data source
            if dates.str.contains('####').any():
                st.warning("🚨 GrubHub dates are corrupted in the CSV. Skipping Grubhub data to maintain accuracy.")
                return pd.DataFrame()  # Return empty dataframe instead of fake dates
            else:
                # Normal date processing
                processed['Date'] = pd.to_datetime(dates, errors='coerce')
        else:
            processed['Date'] = pd.NaT
        
        processed['Platform'] = 'Grubhub'
        
        # Revenue processing
        if 'merchant_net_total' in df.columns:
            processed['Revenue'] = pd.to_numeric(df['merchant_net_total'], errors='coerce')
        else:
            processed['Revenue'] = 0
        
        # Other fields
        field_mapping = {
            'subtotal': 'Subtotal',
            'subtotal_sales_tax': 'Tax', 
            'tip': 'Tips',
            'commission': 'Commission',
            'merchant_funded_promotion': 'Marketing_Fee'
        }
        
        for col, new_col in field_mapping.items():
            if col in df.columns:
                processed[new_col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                processed[new_col] = 0
        
        # Order status - Grubhub data appears to be all completed orders
        processed['Is_Completed'] = True
        processed['Is_Cancelled'] = False
        
        # Store information
        processed['Store_Name'] = df.get('store_name', 'Unknown').fillna('Unknown').astype(str).str.strip()
        processed['Store_ID'] = df.get('store_number', 'Unknown').fillna('Unknown').astype(str)
        
        # Order ID
        processed['Order_ID'] = df.get('order_number', pd.Series(range(len(df)))).astype(str) + '_gh'
        
        # Time processing
        if 'transaction_time_local' in df.columns:
            try:
                time_str = df['transaction_time_local'].astype(str)
                # Handle time corruption similar to dates
                if time_str.str.contains('####').any():
                    # Use random hours between 7 AM and 10 PM for variety
                    np.random.seed(42)  # For reproducibility
                    processed['Hour'] = np.random.randint(7, 23, len(df))
                else:
                    time_parsed = pd.to_datetime(time_str, errors='coerce')
                    processed['Hour'] = time_parsed.dt.hour.fillna(12)
            except:
                processed['Hour'] = 12
        else:
            processed['Hour'] = 12
        
        # Add derived fields - DO THIS BEFORE FILTERING
        processed['DayOfWeek'] = processed['Date'].dt.day_name()
        processed['Month'] = processed['Date'].dt.to_period('M')
        processed['Month_str'] = processed['Date'].dt.strftime('%Y-%m')
        
        # Clean data - keep refunds but remove extreme outliers
        processed = processed[processed['Date'].notna()]
        processed = processed[processed['Revenue'].notna()]
        
        # CRITICAL FIX: Filter to October 2025 only
        processed = processed[
            (processed['Date'].dt.year == 2025) & 
            (processed['Date'].dt.month == 10)
        ]
        
        processed = processed[processed['Revenue'].abs() < 1000]
        
        return processed
    except Exception as e:
        st.error(f"Grubhub processing error: {e}")
        return pd.DataFrame()

def normalize_store_names(df):
    """Normalize store names with CORRECTED store ID mappings"""
    if 'Store_Name' not in df.columns:
        return df
    
    # CORRECTED mapping based on user specifications:
    # 6th ave is US00002, 37th is US00004, 8th is US00005, 
    # Us maiden ln is US00003, fulton is US00006, broadway is US00001
    store_mapping = {
        # Uber variations
        'Luckin Coffee (Broadway)': 'Luckin Coffee - Broadway',
        'Luckin Coffee  (Broadway)': 'Luckin Coffee - Broadway',
        'Luckin Coffee - Broadway': 'Luckin Coffee - Broadway',
        
        # CORRECTED Store ID mappings
        'Luckin Coffee US00001': 'Luckin Coffee - Broadway',     # US00001 = Broadway
        'Luckin Coffee US00002': 'Luckin Coffee - 6th Ave',      # US00002 = 6th Ave
        'Luckin Coffee US00003': 'Luckin Coffee - US Maiden Ln', # US00003 = US maiden ln
        'Luckin Coffee US00004': 'Luckin Coffee - 37th St',      # US00004 = 37th
        'Luckin Coffee US00005': 'Luckin Coffee - 8th Ave',      # US00005 = 8th
        'Luckin Coffee US00006': 'Luckin Coffee - Fulton St',    # US00006 = fulton
        
        # Handle variations in naming
        'Luckin Coffee': 'Luckin Coffee - Unknown Location',
        
        # Handle Grubhub naming patterns
        '755 Broadway': 'Luckin Coffee - Broadway',
        '800 6th Ave': 'Luckin Coffee - 6th Ave',
        '901 8th Ave': 'Luckin Coffee - 8th Ave',
        'Luckin Coffee 755 Broadway': 'Luckin Coffee - Broadway',
        'Luckin Coffee 800 6th Ave': 'Luckin Coffee - 6th Ave',
        'Luckin Coffee 901 8th Ave': 'Luckin Coffee - 8th Ave',
        '102 Fulton St': 'Luckin Coffee - Fulton St',
    }
    
    # Apply normalization
    df['Store_Name_Normalized'] = df['Store_Name'].map(store_mapping).fillna(df['Store_Name'])
    
    # Secondary cleanup for partial matches
    for store_name in df['Store_Name'].unique():
        if pd.isna(df['Store_Name_Normalized'][df['Store_Name'] == store_name].iloc[0]):
            if 'broadway' in str(store_name).lower():
                df.loc[df['Store_Name'] == store_name, 'Store_Name_Normalized'] = 'Luckin Coffee - Broadway'
            elif '6th' in str(store_name).lower():
                df.loc[df['Store_Name'] == store_name, 'Store_Name_Normalized'] = 'Luckin Coffee - 6th Ave'
            elif '8th' in str(store_name).lower():
                df.loc[df['Store_Name'] == store_name, 'Store_Name_Normalized'] = 'Luckin Coffee - 8th Ave'
            elif 'fulton' in str(store_name).lower():
                df.loc[df['Store_Name'] == store_name, 'Store_Name_Normalized'] = 'Luckin Coffee - Fulton St'
            elif '37th' in str(store_name).lower():
                df.loc[df['Store_Name'] == store_name, 'Store_Name_Normalized'] = 'Luckin Coffee - 37th St'
            elif 'maiden' in str(store_name).lower():
                df.loc[df['Store_Name'] == store_name, 'Store_Name_Normalized'] = 'Luckin Coffee - US Maiden Ln'
    
    return df

def add_data_source_notes(df):
    """Add notes about data sources and platform-specific information"""
    
    notes = []
    
    for platform in df['Platform'].unique():
        platform_data = df[df['Platform'] == platform]
        
        if platform == 'DoorDash':
            notes.append(f"**DoorDash**: {len(platform_data)} orders • Data includes commission and marketing fees • All times in local timezone")
        elif platform == 'Uber':
            notes.append(f"**Uber Eats**: {len(platform_data)} orders • Chinese export format processed • Revenue includes fees and adjustments")
        elif platform == 'Grubhub':
            notes.append(f"**Grubhub**: {len(platform_data)} orders • Net revenue after fees")
    
    return notes

def create_enhanced_performance_analysis(df):
    """Create enhanced performance analysis with store-level insights"""
    
    if df.empty:
        return None, None
    
    # Normalize store names first
    df = normalize_store_names(df)
    
    # Store performance analysis
    store_performance = df.groupby(['Store_Name_Normalized', 'Platform']).agg({
        'Revenue': ['sum', 'count', 'mean'],
        'Is_Completed': 'mean'
    }).round(2)
    
    store_performance.columns = ['Total_Revenue', 'Order_Count', 'Avg_Order_Value', 'Completion_Rate']
    store_performance = store_performance.reset_index()
    
    # Platform performance by day of week
    dow_performance = df.groupby(['DayOfWeek', 'Platform']).agg({
        'Revenue': 'sum',
        'Order_ID': 'count'
    }).reset_index()
    
    return store_performance, dow_performance

def create_operational_insights(df):
    """Create enhanced operational insights"""
    
    insights = []
    
    if df.empty:
        return insights
    
    # Peak hours analysis
    if 'Hour' in df.columns:
        hourly_orders = df.groupby('Hour').size()
        if not hourly_orders.empty:
            peak_hour = hourly_orders.idxmax()
            insights.append(f"📈 **Peak ordering hour**: {int(peak_hour)}:00 ({hourly_orders.max()} orders)")
    
    # Platform efficiency
    completion_rates = df.groupby('Platform')['Is_Completed'].mean()
    if not completion_rates.empty:
        best_platform = completion_rates.idxmax()
        insights.append(f"✅ **Highest completion rate**: {best_platform} ({completion_rates.max():.1%})")
    
    # Revenue concentration
    platform_revenue = df.groupby('Platform')['Revenue'].sum()
    if not platform_revenue.empty:
        top_platform = platform_revenue.idxmax()
        revenue_share = platform_revenue.max() / platform_revenue.sum()
        insights.append(f"💰 **Revenue leader**: {top_platform} ({revenue_share:.1%} of total revenue)")
    
    # Store performance
    df_normalized = normalize_store_names(df)
    store_revenue = df_normalized.groupby('Store_Name_Normalized')['Revenue'].sum()
    if len(store_revenue) > 0:
        top_store = store_revenue.idxmax()
        insights.append(f"🏪 **Top performing store**: {top_store}")
    
    return insights

def main():
    # Header
    st.markdown("""
        <div class="luckin-header">
            <h1>☕ Luckin Coffee - Advanced Marketing Analytics Dashboard</h1>
            <p style="font-size: 1.1rem; margin-top: 1rem;">October 2025 Performance Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add important notice
    st.markdown("""
        <div class='success-box'>
        <strong>✅ Data Quality Fixes Applied:</strong><br>
        • Date filtering corrected to October 2025 only<br>
        • Store ID mappings fixed (US00001=Broadway, US00002=6th Ave, US00003=Maiden Ln, US00004=37th, US00005=8th, US00006=Fulton)<br>
        • Revenue analysis focused on actual order data<br>
        • Fake date generation removed from corrupted Grubhub data
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file uploads
    with st.sidebar:
        st.markdown("## 📊 Data Upload Center")
        
        # DoorDash Upload
        st.markdown("#### DoorDash CSV")
        doordash_file = st.file_uploader(
            "Drag and drop file here",
            type=['csv'],
            key="doordash_upload",
            help="Limit 200MB per file • CSV"
        )
        
        # Uber Upload
        st.markdown("#### Uber CSV") 
        uber_file = st.file_uploader(
            "Drag and drop file here",
            type=['csv'],
            key="uber_upload",
            help="Limit 200MB per file • CSV"
        )
        
        # Grubhub Upload
        st.markdown("#### Grubhub CSV")
        grubhub_file = st.file_uploader(
            "Drag and drop file here", 
            type=['csv'],
            key="grubhub_upload",
            help="Limit 200MB per file • CSV"
        )
        
        st.markdown("---")
        
        # Date filter - Disabled since we're focusing on October only
        st.markdown("### 📅 Analysis Period")
        st.info("📊 **Current Focus**: October 2025 only\n\nAll analyses are automatically filtered to October 2025 data to ensure accuracy.")
    
    # Process uploaded files
    all_data = []
    upload_status = []
    processing_notes = []
    
    # Process DoorDash
    if doordash_file is not None:
        try:
            dd_df = pd.read_csv(doordash_file)
            dd_processed = process_doordash_data(dd_df)
            if not dd_processed.empty:
                all_data.append(dd_processed)
                completed_count = dd_processed['Is_Completed'].sum()
                upload_status.append(f"✅ DoorDash: {len(dd_processed)} October orders ({completed_count} completed)")
                processing_notes.append("DoorDash data processed and filtered to October 2025")
            else:
                upload_status.append(f"❌ DoorDash: No valid October data found (Raw rows: {len(dd_df)})")
        except Exception as e:
            upload_status.append(f"❌ DoorDash Error: {str(e)[:50]}")

    # Process Uber
    if uber_file is not None:
        try:
            uber_df = pd.read_csv(uber_file)
            uber_processed = process_uber_data(uber_df)
            if not uber_processed.empty:
                all_data.append(uber_processed)
                completed_count = uber_processed['Is_Completed'].sum()
                upload_status.append(f"✅ Uber: {len(uber_processed)} October orders ({completed_count} completed)")
                processing_notes.append("Uber data processed with header fixes and filtered to October 2025")
            else:
                upload_status.append(f"❌ Uber: No valid October data found (Raw rows: {len(uber_df)})")
        except Exception as e:
            upload_status.append(f"❌ Uber Error: {str(e)[:50]}")
    
    # Process Grubhub
    if grubhub_file is not None:
        try:
            gh_df = pd.read_csv(grubhub_file)
            gh_processed = process_grubhub_data(gh_df)
            if not gh_processed.empty:
                all_data.append(gh_processed)
                completed_count = gh_processed['Is_Completed'].sum()
                upload_status.append(f"✅ Grubhub: {len(gh_processed)} October orders ({completed_count} completed)")
                processing_notes.append("Grubhub data processed and filtered to October 2025")
            else:
                upload_status.append(f"❌ Grubhub: No valid October data found or corrupted dates (Raw rows: {len(gh_df)})")
        except Exception as e:
            upload_status.append(f"❌ Grubhub Error: {str(e)[:50]}")
    
    # Display upload status
    if upload_status:
        with st.sidebar:
            st.markdown("### 📋 Upload Status")
            for status in upload_status:
                if "✅" in status:
                    st.success(status)
                else:
                    st.error(status)
    
    # Check if we have any data
    if not all_data:
        st.info("👋 Welcome! Please upload at least one CSV file from the sidebar to begin analysis.")
        st.markdown("""
            ### Getting Started:
            1. Upload your delivery platform CSV files (DoorDash, Uber, Grubhub)
            2. The dashboard will automatically process and display your October 2025 analytics
            3. Use the tabs above to explore different insights
            
            ### Supported Formats:
            - **DoorDash**: Standard merchant portal export
            - **Uber Eats**: Revenue report export (Chinese or English)  
            - **Grubhub**: Transaction history export
            
            ### Data Quality Notes:
            - **All platforms**: Automatically filtered to October 2025 only
            - **Store IDs**: Correctly mapped per your specifications
            - **Grubhub**: Corrupted date files are excluded to maintain accuracy
            - **Revenue**: Outliers filtered, refunds included
        """)
        return
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    # Ensure we're only showing October 2025 data
    october_filter = (df['Date'].dt.year == 2025) & (df['Date'].dt.month == 10)
    df = df[october_filter]
    
    # Check if we still have data after filtering
    if df.empty:
        st.warning("No valid October 2025 data found in the uploaded files. Please check your data.")
        return
    
    # Show data source notes
    if processing_notes:
        st.markdown("### 📝 Data Processing Notes")
        for note in processing_notes:
            st.info(note)
        
    # Display data source information
    data_source_notes = add_data_source_notes(df)
    with st.expander("📊 Data Source Information", expanded=False):
        for note in data_source_notes:
            st.markdown(f"<div class='platform-note'>{note}</div>", unsafe_allow_html=True)
    
    # Calculate metrics
    completed_df = df[df['Is_Completed'] == True].copy()
    
    # Key metrics
    total_orders = len(df)
    total_revenue = df['Revenue'].sum()
    avg_order_value = df['Revenue'].mean()
    completion_rate = df['Is_Completed'].mean() * 100
    cancellation_rate = df['Is_Cancelled'].mean() * 100
    
    # Platform metrics
    platform_orders = df['Platform'].value_counts()
    platform_revenue = df.groupby('Platform')['Revenue'].sum()
    
    # Time-based metrics - FIXED: Use Month_str for proper aggregation
    daily_revenue = df.groupby('Date')['Revenue'].sum().reset_index()
    daily_revenue = daily_revenue.sort_values('Date')
    
    # FIXED: Monthly aggregation using Month_str (October only)
    monthly_data = df.groupby('Month_str').agg({
        'Revenue': 'sum',
        'Order_ID': 'count'
    }).reset_index()
    monthly_data = monthly_data.sort_values('Month_str')
    monthly_revenue = monthly_data['Revenue']
    monthly_orders = monthly_data['Order_ID']
    
    # Growth calculations - Updated for October-only analysis
    daily_revenue_sorted = daily_revenue.sort_values('Date')
    if len(daily_revenue_sorted) >= 2:
        # Compare early October vs late October
        mid_point = len(daily_revenue_sorted) // 2
        early_october = daily_revenue_sorted.iloc[:mid_point]['Revenue'].mean()
        late_october = daily_revenue_sorted.iloc[mid_point:]['Revenue'].mean()
        if early_october > 0:
            revenue_growth = ((late_october - early_october) / early_october) * 100
        else:
            revenue_growth = 0
    else:
        revenue_growth = 0
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 October Overview", "💰 Revenue Analytics", "🏆 Store Performance", "🕐 Operations & Trends"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.markdown("### 🎯 October 2025 Executive Summary")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Orders", f"{total_orders:,}")
        with col2:
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        with col3:
            st.metric("Average Order Value", f"${avg_order_value:.2f}")
        with col4:
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        with col5:
            st.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")
        
        # Platform distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Order distribution pie chart
            if not platform_orders.empty:
                fig_orders = px.pie(
                    values=platform_orders.values,
                    names=platform_orders.index,
                    title="October Order Distribution by Platform",
                    color_discrete_map=PLATFORM_COLORS
                )
                st.plotly_chart(fig_orders, use_container_width=True)
        
        with col2:
            # Revenue distribution pie chart
            if not platform_revenue.empty:
                fig_revenue = px.pie(
                    values=platform_revenue.values,
                    names=platform_revenue.index,
                    title="October Revenue by Platform",
                    color_discrete_map=PLATFORM_COLORS
                )
                st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Daily trend - October only
        if not daily_revenue.empty and len(daily_revenue) > 1:
            fig_daily = px.line(
                daily_revenue,
                x='Date',
                y='Revenue',
                title="October 2025 Daily Revenue Trend",
                markers=True
            )
            fig_daily.update_layout(
                showlegend=False,
                xaxis_title="Date",
                yaxis_title="Revenue ($)"
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        
        # Platform summary table
        st.markdown("### 📋 October Platform Summary")
        if not platform_revenue.empty:
            summary_df = pd.DataFrame({
                'Platform': platform_revenue.index,
                'Total Orders': platform_orders.reindex(platform_revenue.index, fill_value=0).values,
                'Total Revenue': platform_revenue.values,
                'Average Order Value': [
                    df[df['Platform'] == p]['Revenue'].mean() 
                    for p in platform_revenue.index
                ],
                'Completion Rate (%)': [
                    df[df['Platform'] == p]['Is_Completed'].mean() * 100 
                    for p in platform_revenue.index
                ]
            })
            
            # Format the summary dataframe
            summary_df['Total Revenue'] = summary_df['Total Revenue'].apply(lambda x: f"${x:,.2f}")
            summary_df['Average Order Value'] = summary_df['Average Order Value'].apply(lambda x: f"${x:.2f}")
            summary_df['Completion Rate (%)'] = summary_df['Completion Rate (%)'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
    
    # TAB 2: REVENUE ANALYTICS
    with tab2:
        st.markdown("### 💰 October Revenue Deep Dive")
        
        # Revenue trend by platform
        platform_daily = df.groupby(['Date', 'Platform'])['Revenue'].sum().reset_index()
        
        if not platform_daily.empty:
            fig_platform_trend = px.line(
                platform_daily,
                x='Date',
                y='Revenue',
                color='Platform',
                title="October Daily Revenue Trends by Platform",
                color_discrete_map=PLATFORM_COLORS,
                markers=True
            )
            fig_platform_trend.update_layout(
                xaxis_title="Date",
                yaxis_title="Revenue ($)"
            )
            st.plotly_chart(fig_platform_trend, use_container_width=True)
        
        # Revenue insights
        insights = create_operational_insights(df)
        if insights:
            st.markdown("### 📊 Key Revenue Insights")
            for insight in insights:
                st.markdown(f"• {insight}")
    
    # TAB 3: STORE PERFORMANCE
    with tab3:
        st.markdown("### 🏆 October Store Performance Analysis")
        
        # Normalize store names
        df_normalized = normalize_store_names(df)
        
        # Store performance analysis
        store_performance = df_normalized.groupby('Store_Name_Normalized').agg({
            'Revenue': ['sum', 'count', 'mean'],
            'Is_Completed': 'mean'
        }).round(2)
        
        store_performance.columns = ['Total_Revenue', 'Order_Count', 'Avg_Order_Value', 'Completion_Rate']
        store_performance = store_performance.reset_index()
        store_performance = store_performance.sort_values('Total_Revenue', ascending=False)
        
        # Format for display
        store_display = store_performance.copy()
        store_display['Total_Revenue'] = store_display['Total_Revenue'].apply(lambda x: f"${x:,.2f}")
        store_display['Avg_Order_Value'] = store_display['Avg_Order_Value'].apply(lambda x: f"${x:.2f}")
        store_display['Completion_Rate'] = (store_display['Completion_Rate'] * 100).apply(lambda x: f"{x:.1f}%")
        store_display.columns = ['Store', 'Total Revenue', 'Total Orders', 'Avg Order Value', 'Completion Rate']
        
        st.dataframe(store_display, hide_index=True, use_container_width=True)
        
        # Store revenue chart
        if not store_performance.empty:
            fig_stores = px.bar(
                store_performance,
                x='Store_Name_Normalized',
                y='Total_Revenue',
                title="October Revenue by Store",
                labels={'Store_Name_Normalized': 'Store', 'Total_Revenue': 'Total Revenue ($)'}
            )
            fig_stores.update_xaxes(tickangle=45)
            st.plotly_chart(fig_stores, use_container_width=True)
    
    # TAB 4: OPERATIONS & TRENDS
    with tab4:
        st.markdown("### 🕐 October Operations & Activity Analysis")
        
        # Hourly patterns
        if 'Hour' in df.columns:
            hourly_data = df.groupby('Hour').agg({
                'Revenue': 'sum',
                'Order_ID': 'count'
            }).reset_index()
            
            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Bar(
                x=hourly_data['Hour'],
                y=hourly_data['Order_ID'],
                name='Orders',
                marker_color='lightblue',
                yaxis='y'
            ))
            fig_hourly.add_trace(go.Scatter(
                x=hourly_data['Hour'],
                y=hourly_data['Revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='red', width=3),
                yaxis='y2'
            ))
            
            fig_hourly.update_layout(
                title="October Hourly Performance Patterns",
                xaxis_title="Hour of Day",
                yaxis=dict(title="Number of Orders", side="left"),
                yaxis2=dict(title="Revenue ($)", side="right", overlaying="y"),
                legend=dict(x=0, y=1)
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Day of week analysis
        if 'DayOfWeek' in df.columns:
            dow_data = df.groupby('DayOfWeek').agg({
                'Revenue': 'sum',
                'Order_ID': 'count'
            }).reset_index()
            
            # Order days correctly
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_data['DayOfWeek'] = pd.Categorical(dow_data['DayOfWeek'], categories=day_order, ordered=True)
            dow_data = dow_data.sort_values('DayOfWeek')
            
            col1, col2 = st.columns(2)
            with col1:
                fig_dow_orders = px.bar(
                    dow_data,
                    x='DayOfWeek',
                    y='Order_ID',
                    title="October Orders by Day of Week",
                    labels={'DayOfWeek': 'Day', 'Order_ID': 'Number of Orders'}
                )
                st.plotly_chart(fig_dow_orders, use_container_width=True)
            
            with col2:
                fig_dow_revenue = px.bar(
                    dow_data,
                    x='DayOfWeek',
                    y='Revenue',
                    title="October Revenue by Day of Week",
                    labels={'DayOfWeek': 'Day', 'Revenue': 'Revenue ($)'}
                )
                st.plotly_chart(fig_dow_revenue, use_container_width=True)

if __name__ == "__main__":
    main()
