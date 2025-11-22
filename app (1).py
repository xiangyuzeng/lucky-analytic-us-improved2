import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import base64
import io
import xlsxwriter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from operator import attrgetter
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Luckin Coffee - Advanced Marketing Analytics Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styles ---
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
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        
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
            text-align: center;
        }
        
        .big-font {
            font-size: 2.5em !important;
            font-weight: bold;
            color: #232773;
        }
        
        .improvement-green {
            color: #28a745;
            font-weight: bold;
        }
        
        .decline-red {
            color: #dc3545;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Platform Configuration ---
PLATFORM_COLORS = {
    'DoorDash': '#ff3008',
    'Uber': '#000000',
    'Grubhub': '#ff8000'
}

# --- Data Processing Functions ---
@st.cache_data
def process_doordash_data(df):
    """Process DoorDash data with proper column mapping"""
    try:
        # Map Chinese column headers to English
        column_mapping = {
            '时间戳本地日期': 'Date',
            '小计': 'Subtotal',
            '净总计': 'Net_Total',
            '最终订单状态': 'Order_Status',
            '佣金': 'Commission',
            '员工小费': 'Tips',
            'Store ID': 'Store_ID',
            '店铺名称': 'Store_Name'
        }
        
        # Apply column mapping
        for chinese, english in column_mapping.items():
            if chinese in df.columns:
                df = df.rename(columns={chinese: english})
        
        # Clean and process data
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Platform'] = 'DoorDash'
        df['Revenue'] = pd.to_numeric(df['Net_Total'], errors='coerce')
        df['Is_Completed'] = df['Order_Status'].str.contains('Delivered|completed', case=False, na=False)
        df['Is_Cancelled'] = df['Order_Status'].str.contains('Cancelled|cancel', case=False, na=False)
        
        # Clean numeric columns
        numeric_cols = ['Subtotal', 'Net_Total', 'Commission', 'Tips']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error processing DoorDash data: {e}")
        return pd.DataFrame()

@st.cache_data
def process_uber_data(df):
    """Process Uber data with proper column mapping"""
    try:
        # Map Chinese column headers to English
        column_mapping = {
            '订单日期': 'Date',
            '销售额（含税）': 'Revenue',
            '调整后的总销售额（含税费）': 'Adjusted_Revenue',
            '收入总额': 'Total_Income',
            '订单状态': 'Order_Status',
            '平台服务费': 'Service_Fee',
            '小费': 'Tips',
            '餐厅名称': 'Store_Name'
        }
        
        # Apply column mapping
        for chinese, english in column_mapping.items():
            if chinese in df.columns:
                df = df.rename(columns={chinese: english})
        
        # Clean and process data
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Platform'] = 'Uber'
        
        # Use Total_Income if available, otherwise use Revenue
        if 'Total_Income' in df.columns:
            df['Revenue'] = pd.to_numeric(df['Total_Income'], errors='coerce')
        else:
            df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
            
        df['Is_Completed'] = df['Order_Status'].str.contains('已完成|completed', case=False, na=False)
        df['Is_Cancelled'] = df['Order_Status'].str.contains('已取消|cancelled', case=False, na=False)
        
        # Clean numeric columns
        numeric_cols = ['Adjusted_Revenue', 'Service_Fee', 'Tips']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error processing Uber data: {e}")
        return pd.DataFrame()

@st.cache_data
def process_grubhub_data(df):
    """Process Grubhub data with proper column mapping"""
    try:
        # Clean and process data
        df['Date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['Platform'] = 'Grubhub'
        df['Revenue'] = pd.to_numeric(df['merchant_net_total'], errors='coerce')
        df['Is_Completed'] = df['transaction_type'].str.contains('Prepaid|Order', case=False, na=False)
        df['Is_Cancelled'] = False  # Grubhub data doesn't seem to include cancelled orders
        df['Store_Name'] = df['store_name']
        
        # Clean numeric columns
        numeric_cols = ['subtotal', 'commission', 'tip', 'processing_fee']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error processing Grubhub data: {e}")
        return pd.DataFrame()

def calculate_marketing_metrics(df):
    """Calculate comprehensive marketing metrics"""
    try:
        completed_df = df[df['Is_Completed']].copy()
        
        metrics = {
            'Total_Customers': len(df),
            'Total_Orders': len(completed_df),
            'Total_Revenue': completed_df['Revenue'].sum(),
            'AOV': completed_df['Revenue'].mean(),
            'Cancellation_Rate': df['Is_Cancelled'].mean() * 100 if 'Is_Cancelled' in df.columns else 0,
            'Completion_Rate': df['Is_Completed'].mean() * 100,
            'Revenue_Growth': calculate_revenue_growth(completed_df),
            'Order_Growth': calculate_order_growth(completed_df)
        }
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating marketing metrics: {e}")
        return {}

def calculate_revenue_growth(df):
    """Calculate month-over-month revenue growth"""
    try:
        monthly_revenue = df.groupby(df['Date'].dt.to_period('M'))['Revenue'].sum()
        if len(monthly_revenue) >= 2:
            current_month = monthly_revenue.iloc[-1]
            previous_month = monthly_revenue.iloc[-2]
            return ((current_month - previous_month) / previous_month) * 100
        return 0
    except:
        return 0

def calculate_order_growth(df):
    """Calculate month-over-month order growth"""
    try:
        monthly_orders = df.groupby(df['Date'].dt.to_period('M')).size()
        if len(monthly_orders) >= 2:
            current_month = monthly_orders.iloc[-1]
            previous_month = monthly_orders.iloc[-2]
            return ((current_month - previous_month) / previous_month) * 100
        return 0
    except:
        return 0

def create_cohort_analysis(df):
    """Create cohort analysis for customer retention"""
    try:
        # Use order date and customer ID (if available) for cohort analysis
        # For now, we'll use a simplified approach based on dates and stores
        df['OrderMonth'] = df['Date'].dt.to_period('M')
        
        # Group by month and calculate retention metrics
        monthly_customers = df.groupby('OrderMonth').agg({
            'Store_Name': 'nunique',
            'Revenue': 'sum'
        }).reset_index()
        
        monthly_customers['OrderMonth'] = monthly_customers['OrderMonth'].astype(str)
        
        return monthly_customers
        
    except Exception as e:
        st.error(f"Error in cohort analysis: {e}")
        return pd.DataFrame()

def create_rfm_analysis(df):
    """Create RFM (Recency, Frequency, Monetary) analysis"""
    try:
        # Simplified RFM analysis based on available data
        current_date = df['Date'].max()
        
        # Group by store (as proxy for customer)
        rfm = df.groupby('Store_Name').agg({
            'Date': lambda x: (current_date - x.max()).days,  # Recency
            'Revenue': ['count', 'sum']  # Frequency and Monetary
        }).round(2)
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Calculate RFM scores (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1,2,3,4,5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
        
        # Combined RFM Score
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        return rfm
        
    except Exception as e:
        st.error(f"Error in RFM analysis: {e}")
        return pd.DataFrame()

def create_churn_analysis(df):
    """Create customer churn analysis"""
    try:
        # Calculate days since last order for each store
        current_date = df['Date'].max()
        
        last_order = df.groupby('Store_Name')['Date'].max().reset_index()
        last_order['Days_Since_Last_Order'] = (current_date - last_order['Date']).dt.days
        
        # Define churn (stores with no orders in last 30 days)
        last_order['Is_Churned'] = last_order['Days_Since_Last_Order'] > 30
        
        churn_rate = last_order['Is_Churned'].mean() * 100
        
        return last_order, churn_rate
        
    except Exception as e:
        st.error(f"Error in churn analysis: {e}")
        return pd.DataFrame(), 0

def create_advanced_revenue_charts(df):
    """Create advanced revenue visualization charts"""
    charts = {}
    
    try:
        # Revenue by platform over time
        daily_platform_revenue = df.groupby(['Date', 'Platform'])['Revenue'].sum().reset_index()
        
        fig = px.line(daily_platform_revenue, x='Date', y='Revenue', color='Platform',
                     title="Daily Revenue Trend by Platform",
                     color_discrete_map=PLATFORM_COLORS)
        fig.update_layout(hovermode='x unified')
        charts['revenue_by_platform'] = fig
        
        # Revenue distribution
        fig2 = px.box(df, x='Platform', y='Revenue', color='Platform',
                     title="Revenue Distribution by Platform",
                     color_discrete_map=PLATFORM_COLORS)
        charts['revenue_distribution'] = fig2
        
        # Monthly revenue comparison
        monthly_revenue = df.groupby([df['Date'].dt.to_period('M'), 'Platform'])['Revenue'].sum().reset_index()
        monthly_revenue['Date'] = monthly_revenue['Date'].astype(str)
        
        fig3 = px.bar(monthly_revenue, x='Date', y='Revenue', color='Platform',
                     title="Monthly Revenue Comparison",
                     color_discrete_map=PLATFORM_COLORS)
        charts['monthly_comparison'] = fig3
        
        return charts
        
    except Exception as e:
        st.error(f"Error creating revenue charts: {e}")
        return {}

def create_performance_charts(df):
    """Create advanced performance visualization charts"""
    charts = {}
    
    try:
        # Order completion rates
        completion_rates = df.groupby('Platform')['Is_Completed'].mean() * 100
        
        fig = px.bar(x=completion_rates.index, y=completion_rates.values,
                    color=completion_rates.index,
                    title="Order Completion Rate by Platform",
                    color_discrete_map=PLATFORM_COLORS)
        fig.update_layout(yaxis_title="Completion Rate (%)")
        charts['completion_rates'] = fig
        
        # Average order value by platform
        aov = df.groupby('Platform')['Revenue'].mean()
        
        fig2 = px.bar(x=aov.index, y=aov.values,
                     color=aov.index,
                     title="Average Order Value by Platform",
                     color_discrete_map=PLATFORM_COLORS)
        fig2.update_layout(yaxis_title="Average Order Value ($)")
        charts['aov_comparison'] = fig2
        
        # Order volume trends
        daily_orders = df.groupby(['Date', 'Platform']).size().reset_index(name='Orders')
        
        fig3 = px.line(daily_orders, x='Date', y='Orders', color='Platform',
                      title="Daily Order Volume by Platform",
                      color_discrete_map=PLATFORM_COLORS)
        charts['order_volume'] = fig3
        
        return charts
        
    except Exception as e:
        st.error(f"Error creating performance charts: {e}")
        return {}

def main():
    # Header
    st.markdown("""
        <div class="luckin-header">
            <h1>☕ Luckin Coffee - Advanced Marketing Analytics Dashboard</h1>
            <p>Comprehensive cross-platform performance analysis and customer insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file uploads
    st.sidebar.markdown("### 📁 Data Upload")
    st.sidebar.markdown("Upload your platform CSV files below:")
    
    # File uploaders with unique keys
    uber_file = st.sidebar.file_uploader("📱 Uber Eats CSV", type="csv", key="uber_upload")
    doordash_file = st.sidebar.file_uploader("🚗 DoorDash CSV", type="csv", key="doordash_upload") 
    grubhub_file = st.sidebar.file_uploader("🍔 Grubhub CSV", type="csv", key="grubhub_upload")
    
    # Process uploaded files
    dfs = []
    platforms_loaded = []
    
    if uber_file is not None:
        try:
            uber_df = pd.read_csv(uber_file)
            uber_processed = process_uber_data(uber_df)
            if not uber_processed.empty:
                dfs.append(uber_processed)
                platforms_loaded.append("Uber")
                st.sidebar.success("✅ Uber data loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading Uber data: {e}")
    
    if doordash_file is not None:
        try:
            doordash_df = pd.read_csv(doordash_file)
            doordash_processed = process_doordash_data(doordash_df)
            if not doordash_processed.empty:
                dfs.append(doordash_processed)
                platforms_loaded.append("DoorDash")
                st.sidebar.success("✅ DoorDash data loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading DoorDash data: {e}")
    
    if grubhub_file is not None:
        try:
            grubhub_df = pd.read_csv(grubhub_file)
            grubhub_processed = process_grubhub_data(grubhub_df)
            if not grubhub_processed.empty:
                dfs.append(grubhub_processed)
                platforms_loaded.append("Grubhub")
                st.sidebar.success("✅ Grubhub data loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading Grubhub data: {e}")
    
    if not dfs:
        st.warning("👆 Please upload at least one CSV file to begin analysis")
        return
    
    # Combine all data
    try:
        df = pd.concat(dfs, ignore_index=True)
        df = df.dropna(subset=['Date', 'Revenue'])
        df = df[df['Revenue'] > 0]  # Filter out invalid revenue entries
        
        st.sidebar.markdown(f"### 📊 Data Summary")
        st.sidebar.write(f"**Platforms loaded:** {', '.join(platforms_loaded)}")
        st.sidebar.write(f"**Total records:** {len(df):,}")
        st.sidebar.write(f"**Date range:** {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        
    except Exception as e:
        st.error(f"Error combining data: {e}")
        return
    
    # Calculate marketing metrics
    marketing_metrics = calculate_marketing_metrics(df)
    
    # Create tabs with enhanced functionality
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", "💰 Revenue", "🏆 Performance", 
        "⚡ Operations", "📈 Growth", "🎯 Attribution", "🔄 Retention"
    ])
    
    # Tab 1: Enhanced Overview
    with tab1:
        st.markdown("### 📊 Platform Performance Overview")
        
        completed_df = df[df['Is_Completed']].copy()
        
        # Key metrics with improved layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Orders", f"{marketing_metrics['Total_Orders']:,}", 
                     delta=f"{marketing_metrics['Order_Growth']:.1f}% MoM")
        with col2:
            st.metric("Total Revenue", f"${marketing_metrics['Total_Revenue']:,.2f}",
                     delta=f"{marketing_metrics['Revenue_Growth']:.1f}% MoM")
        with col3:
            st.metric("Average Order Value", f"${marketing_metrics['AOV']:.2f}")
        with col4:
            st.metric("Completion Rate", f"{marketing_metrics['Completion_Rate']:.1f}%")
        
        st.markdown("---")
        
        # Platform distribution with enhanced charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Orders by platform
            platform_orders = completed_df['Platform'].value_counts()
            fig_orders = px.pie(
                values=platform_orders.values,
                names=platform_orders.index,
                title="Orders Distribution by Platform",
                color_discrete_map=PLATFORM_COLORS
            )
            st.plotly_chart(fig_orders, use_container_width=True, key='overview_orders_pie_chart')
        
        with col2:
            # Revenue by platform
            platform_revenue = completed_df.groupby('Platform')['Revenue'].sum().sort_values(ascending=False)
            fig_revenue = px.bar(
                x=platform_revenue.index,
                y=platform_revenue.values,
                title="Revenue by Platform",
                color=platform_revenue.index,
                color_discrete_map=PLATFORM_COLORS
            )
            st.plotly_chart(fig_revenue, use_container_width=True, key='overview_revenue_bar_chart')
        
        # Platform performance comparison table
        st.markdown("### 📈 Platform Performance Comparison")
        
        platform_summary = completed_df.groupby('Platform').agg({
            'Revenue': ['count', 'sum', 'mean'],
            'Date': ['min', 'max']
        }).round(2)
        
        platform_summary.columns = ['Orders', 'Total Revenue', 'AOV', 'First Order', 'Last Order']
        platform_summary['Market Share'] = (platform_summary['Orders'] / platform_summary['Orders'].sum() * 100).round(1)
        
        st.dataframe(platform_summary, use_container_width=True)

    # Tab 2: Enhanced Revenue Analytics
    with tab2:
        st.markdown("### 💰 Advanced Revenue Analytics")
        
        # Create advanced revenue charts
        revenue_charts = create_advanced_revenue_charts(completed_df)
        
        # Daily revenue trend
        if 'revenue_by_platform' in revenue_charts:
            st.plotly_chart(revenue_charts['revenue_by_platform'], use_container_width=True, 
                           key='revenue_daily_trend_enhanced')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue distribution
            if 'revenue_distribution' in revenue_charts:
                st.plotly_chart(revenue_charts['revenue_distribution'], use_container_width=True,
                               key='revenue_distribution_box')
        
        with col2:
            # Weekly revenue pattern
            completed_df['Day_of_Week'] = completed_df['Date'].dt.day_name()
            weekly_revenue = completed_df.groupby(['Day_of_Week', 'Platform'])['Revenue'].sum().reset_index()
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_revenue['Day_of_Week'] = pd.Categorical(weekly_revenue['Day_of_Week'], categories=day_order, ordered=True)
            weekly_revenue = weekly_revenue.sort_values('Day_of_Week')
            
            fig_weekly = px.bar(weekly_revenue, x='Day_of_Week', y='Revenue', color='Platform',
                               title="Revenue by Day of Week",
                               color_discrete_map=PLATFORM_COLORS)
            st.plotly_chart(fig_weekly, use_container_width=True, key='revenue_weekly_pattern')
        
        # Monthly comparison
        if 'monthly_comparison' in revenue_charts:
            st.plotly_chart(revenue_charts['monthly_comparison'], use_container_width=True,
                           key='revenue_monthly_comparison')

    # Tab 3: Enhanced Performance Analytics  
    with tab3:
        st.markdown("### 🏆 Platform Performance Deep Dive")
        
        # Create performance charts
        performance_charts = create_performance_charts(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Completion rates
            if 'completion_rates' in performance_charts:
                st.plotly_chart(performance_charts['completion_rates'], use_container_width=True,
                               key='performance_completion_rates')
        
        with col2:
            # AOV comparison
            if 'aov_comparison' in performance_charts:
                st.plotly_chart(performance_charts['aov_comparison'], use_container_width=True,
                               key='performance_aov_comparison')
        
        # Order volume trends
        if 'order_volume' in performance_charts:
            st.plotly_chart(performance_charts['order_volume'], use_container_width=True,
                           key='performance_order_volume')
        
        # Performance heatmap
        st.markdown("### 📊 Store Performance Heatmap")
        
        store_performance = completed_df.groupby(['Store_Name', 'Platform']).agg({
            'Revenue': ['sum', 'count', 'mean']
        }).round(2)
        
        store_performance.columns = ['Total_Revenue', 'Order_Count', 'AOV']
        store_performance = store_performance.reset_index()
        
        # Create pivot for heatmap
        heatmap_data = store_performance.pivot_table(
            index='Store_Name', 
            columns='Platform', 
            values='Total_Revenue', 
            fill_value=0
        )
        
        fig_heatmap = px.imshow(heatmap_data, 
                               title="Store Revenue Performance by Platform",
                               aspect="auto",
                               color_continuous_scale='Blues')
        st.plotly_chart(fig_heatmap, use_container_width=True, key='performance_store_heatmap_enhanced')

    # Tab 4: Enhanced Operations Analytics
    with tab4:
        st.markdown("### ⚡ Operational Intelligence")
        
        # Hour-of-day analysis
        completed_df['Hour'] = completed_df['Date'].dt.hour
        completed_df['Day_of_Week'] = completed_df['Date'].dt.day_name()
        
        # Order patterns by hour and day
        hour_day_orders = completed_df.groupby(['Hour', 'Day_of_Week']).size().unstack(fill_value=0)
        
        fig_hour_heatmap = px.imshow(
            hour_day_orders.T,
            labels=dict(x="Hour of Day", y="Day of Week", color="Orders"),
            title="Order Patterns: Hour vs Day of Week",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_hour_heatmap, use_container_width=True, key='operations_hour_day_heatmap')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Peak hours analysis
            hourly_orders = completed_df.groupby('Hour').size()
            
            fig_hourly = px.bar(x=hourly_orders.index, y=hourly_orders.values,
                               title="Orders by Hour of Day",
                               labels={'x': 'Hour', 'y': 'Number of Orders'})
            st.plotly_chart(fig_hourly, use_container_width=True, key='operations_hourly_orders')
        
        with col2:
            # Cancellation analysis
            if 'Is_Cancelled' in df.columns:
                cancellation_rate = df.groupby('Platform')['Is_Cancelled'].mean() * 100
                
                fig_cancel = px.bar(
                    x=cancellation_rate.index,
                    y=cancellation_rate.values,
                    title="Cancellation Rate by Platform",
                    color=cancellation_rate.index,
                    color_discrete_map=PLATFORM_COLORS
                )
                fig_cancel.update_layout(yaxis_title="Cancellation Rate (%)")
                st.plotly_chart(fig_cancel, use_container_width=True, key='operations_cancellation_analysis')

    # Tab 5: Enhanced Growth Analytics
    with tab5:
        st.markdown("### 📈 Growth Metrics & Trends")
        
        # Calculate daily metrics
        daily_revenue = completed_df.groupby('Date')['Revenue'].sum().reset_index()
        daily_orders = completed_df.groupby('Date').size().reset_index(name='Orders')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily orders trend
            fig_daily_orders = px.line(daily_orders, x='Date', y='Orders',
                                     title="Daily Orders Trend")
            fig_daily_orders.add_scatter(x=daily_orders['Date'], y=daily_orders['Orders'].rolling(7).mean(),
                                        mode='lines', name='7-day MA', line=dict(color='red'))
            st.plotly_chart(fig_daily_orders, use_container_width=True, key='growth_daily_orders_enhanced')
        
        with col2:
            # Daily revenue trend
            fig_daily_revenue = px.line(daily_revenue, x='Date', y='Revenue',
                                      title="Daily Revenue Trend")
            fig_daily_revenue.add_scatter(x=daily_revenue['Date'], y=daily_revenue['Revenue'].rolling(7).mean(),
                                         mode='lines', name='7-day MA', line=dict(color='red'))
            st.plotly_chart(fig_daily_revenue, use_container_width=True, key='growth_daily_revenue_enhanced')
        
        # Growth rate calculations
        daily_orders['Orders_Growth'] = daily_orders['Orders'].pct_change() * 100
        daily_revenue['Revenue_Growth'] = daily_revenue['Revenue'].pct_change() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_order_growth = px.bar(daily_orders.dropna(), x='Date', y='Orders_Growth',
                                    title="Daily Orders Growth Rate (%)")
            st.plotly_chart(fig_order_growth, use_container_width=True, key='growth_orders_growth_rate_enhanced')
        
        with col2:
            fig_revenue_growth = px.bar(daily_revenue.dropna(), x='Date', y='Revenue_Growth',
                                      title="Daily Revenue Growth Rate (%)")
            st.plotly_chart(fig_revenue_growth, use_container_width=True, key='growth_revenue_growth_rate_enhanced')
        
        # Growth insights
        st.markdown("### 🔮 Growth Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_daily_growth = daily_revenue['Revenue_Growth'].mean()
            st.metric("Average Daily Growth", f"{avg_daily_growth:.2f}%")
        
        with col2:
            best_day_revenue = daily_revenue.loc[daily_revenue['Revenue'].idxmax()]
            st.metric("Best Revenue Day", f"${best_day_revenue['Revenue']:,.2f}",
                     delta=best_day_revenue['Date'].strftime('%Y-%m-%d'))
        
        with col3:
            total_growth = ((daily_revenue['Revenue'].iloc[-1] - daily_revenue['Revenue'].iloc[0]) / 
                           daily_revenue['Revenue'].iloc[0] * 100)
            st.metric("Total Period Growth", f"{total_growth:.1f}%")

    # Tab 6: Enhanced Attribution Analytics
    with tab6:
        st.markdown("### 🎯 Customer Attribution & Segmentation")
        
        # RFM Analysis
        rfm_data = create_rfm_analysis(completed_df)
        
        if not rfm_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # RFM Score Distribution
                rfm_counts = rfm_data['RFM_Score'].value_counts().head(10)
                
                fig_rfm = px.bar(x=rfm_counts.index, y=rfm_counts.values,
                                title="Top RFM Segments",
                                labels={'x': 'RFM Score', 'y': 'Number of Stores'})
                st.plotly_chart(fig_rfm, use_container_width=True, key='attribution_rfm_segments')
            
            with col2:
                # Monetary vs Frequency scatter
                fig_scatter = px.scatter(rfm_data, x='Frequency', y='Monetary',
                                       title="Customer Value Segmentation",
                                       labels={'Frequency': 'Order Frequency', 'Monetary': 'Total Spend'})
                st.plotly_chart(fig_scatter, use_container_width=True, key='attribution_value_segmentation')
        
        # Customer Lifetime Value estimation
        st.markdown("### 💎 Customer Lifetime Value Analysis")
        
        clv_data = completed_df.groupby('Store_Name').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Date': ['min', 'max']
        }).round(2)
        
        clv_data.columns = ['Total_Spend', 'Avg_Order_Value', 'Order_Count', 'First_Order', 'Last_Order']
        clv_data['Days_Active'] = (clv_data['Last_Order'] - clv_data['First_Order']).dt.days + 1
        clv_data['Order_Frequency'] = clv_data['Order_Count'] / clv_data['Days_Active'] * 30  # Orders per month
        
        # Estimate CLV (simplified)
        clv_data['Estimated_CLV'] = clv_data['Avg_Order_Value'] * clv_data['Order_Frequency'] * 12  # Annual
        
        fig_clv = px.histogram(clv_data, x='Estimated_CLV', nbins=20,
                              title="Customer Lifetime Value Distribution")
        st.plotly_chart(fig_clv, use_container_width=True, key='attribution_clv_distribution_enhanced')

    # Tab 7: Enhanced Retention & Attrition Analytics
    with tab7:
        st.markdown("### 🔄 Customer Retention & Attrition Analysis")
        
        # Churn analysis
        churn_data, churn_rate = create_churn_analysis(completed_df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Churn Rate", f"{churn_rate:.1f}%", 
                     help="Stores with no orders in last 30 days")
        
        with col2:
            active_stores = len(churn_data[~churn_data['Is_Churned']])
            st.metric("Active Stores", f"{active_stores}")
        
        with col3:
            avg_days_since_order = churn_data['Days_Since_Last_Order'].mean()
            st.metric("Avg Days Since Last Order", f"{avg_days_since_order:.0f}")
        
        # Churn risk analysis
        st.markdown("### ⚠️ Churn Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Days since last order distribution
            fig_churn_dist = px.histogram(churn_data, x='Days_Since_Last_Order', nbins=20,
                                         title="Days Since Last Order Distribution")
            fig_churn_dist.add_vline(x=30, line_dash="dash", line_color="red",
                                    annotation_text="Churn Threshold (30 days)")
            st.plotly_chart(fig_churn_dist, use_container_width=True, key='retention_churn_distribution')
        
        with col2:
            # Churn by platform
            if not churn_data.empty and 'Platform' in completed_df.columns:
                # Merge churn data with platform info
                store_platform = completed_df.groupby('Store_Name')['Platform'].first().reset_index()
                churn_platform = churn_data.merge(store_platform, on='Store_Name', how='left')
                
                platform_churn = churn_platform.groupby('Platform')['Is_Churned'].mean() * 100
                
                fig_platform_churn = px.bar(x=platform_churn.index, y=platform_churn.values,
                                           color=platform_churn.index,
                                           title="Churn Rate by Platform",
                                           color_discrete_map=PLATFORM_COLORS)
                fig_platform_churn.update_layout(yaxis_title="Churn Rate (%)")
                st.plotly_chart(fig_platform_churn, use_container_width=True, key='retention_platform_churn')
        
        # Cohort analysis
        st.markdown("### 📊 Cohort Retention Analysis")
        
        cohort_data = create_cohort_analysis(completed_df)
        
        if not cohort_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly active stores
                fig_cohort_stores = px.line(cohort_data, x='OrderMonth', y='Store_Name',
                                           title="Monthly Active Stores")
                fig_cohort_stores.update_layout(xaxis_title="Month", yaxis_title="Number of Stores")
                st.plotly_chart(fig_cohort_stores, use_container_width=True, key='retention_monthly_stores')
            
            with col2:
                # Monthly revenue trend for retention
                fig_cohort_revenue = px.line(cohort_data, x='OrderMonth', y='Revenue',
                                           title="Monthly Revenue from Active Stores")
                fig_cohort_revenue.update_layout(xaxis_title="Month", yaxis_title="Revenue ($)")
                st.plotly_chart(fig_cohort_revenue, use_container_width=True, key='retention_monthly_revenue')
        
        # Customer segmentation based on purchase behavior
        st.markdown("### 🎯 Customer Behavior Segmentation")
        
        # Prepare data for segmentation
        if not completed_df.empty:
            customer_features = completed_df.groupby('Store_Name').agg({
                'Revenue': ['sum', 'mean', 'std'],
                'Date': ['count', 'nunique']
            }).round(2)
            
            customer_features.columns = ['Total_Revenue', 'Avg_Order_Value', 'Revenue_Std', 'Total_Orders', 'Unique_Days']
            customer_features['Revenue_Consistency'] = 1 / (1 + customer_features['Revenue_Std'])
            customer_features = customer_features.fillna(0)
            
            # Perform clustering if we have enough data
            if len(customer_features) >= 3:
                try:
                    features_for_clustering = customer_features[['Total_Revenue', 'Avg_Order_Value', 'Total_Orders']].fillna(0)
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features_for_clustering)
                    
                    n_clusters = min(4, len(customer_features))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    customer_features['Segment'] = kmeans.fit_predict(features_scaled)
                    
                    # Visualize segments
                    fig_segments = px.scatter(customer_features, x='Total_Revenue', y='Avg_Order_Value',
                                            color='Segment', size='Total_Orders',
                                            title="Customer Segmentation Analysis",
                                            labels={'Total_Revenue': 'Total Revenue ($)', 
                                                   'Avg_Order_Value': 'Average Order Value ($)'})
                    st.plotly_chart(fig_segments, use_container_width=True, key='retention_customer_segments')
                    
                    # Segment summary
                    segment_summary = customer_features.groupby('Segment').agg({
                        'Total_Revenue': 'mean',
                        'Avg_Order_Value': 'mean',
                        'Total_Orders': 'mean'
                    }).round(2)
                    
                    segment_summary.index = [f"Segment {i+1}" for i in segment_summary.index]
                    st.dataframe(segment_summary, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not perform customer segmentation: {e}")
    
    # Footer with data export functionality
    st.markdown("---")
    st.markdown("### 📤 Export Analytics Report")
    
    if st.button("Generate Comprehensive Report", key="export_report_button"):
        # Create a comprehensive report
        report_data = {
            'Summary': marketing_metrics,
            'Platform_Performance': completed_df.groupby('Platform').agg({
                'Revenue': ['count', 'sum', 'mean'],
                'Date': ['min', 'max']
            }).round(2),
            'Daily_Trends': daily_revenue,
            'Churn_Analysis': churn_data if not churn_data.empty else pd.DataFrame()
        }
        
        # Convert to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for sheet_name, data in report_data.items():
                if isinstance(data, dict):
                    pd.DataFrame([data]).to_excel(writer, sheet_name=sheet_name, index=False)
                elif isinstance(data, pd.DataFrame) and not data.empty:
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        st.download_button(
            label="📥 Download Excel Report",
            data=output.getvalue(),
            file_name=f"luckin_analytics_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Usage instructions
    with st.expander("📋 How to Use This Dashboard"):
        st.markdown("""
        ### Platform Integration Guide
        
        **Supported Platforms:**
        - 🍔 **Grubhub**: Upload transaction reports with order details
        - 📱 **Uber Eats**: Upload revenue reports from Uber Eats Manager
        - 🚗 **DoorDash**: Upload order history with commission data
        
        **Key Features:**
        - 📊 **Multi-platform Analytics**: Unified view across all delivery platforms
        - 💰 **Revenue Intelligence**: Advanced revenue tracking and forecasting  
        - 🏆 **Performance Optimization**: Platform-specific performance insights
        - 🎯 **Customer Analytics**: RFM analysis, segmentation, and CLV calculation
        - 🔄 **Retention Insights**: Cohort analysis and churn prediction
        - 📈 **Growth Tracking**: Trend analysis and growth forecasting
        """)

if __name__ == "__main__":
    main()
