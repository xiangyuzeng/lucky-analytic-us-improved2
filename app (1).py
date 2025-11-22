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
    </style>
""", unsafe_allow_html=True)

# Platform Colors
PLATFORM_COLORS = {
    'DoorDash': '#ff3008',
    'Uber': '#000000',
    'Grubhub': '#ff8000'
}

# Data Processing Functions
@st.cache_data
def process_doordash_data(df):
    """Process DoorDash data"""
    try:
        df['Date'] = pd.to_datetime(df['时间戳本地日期'], errors='coerce')
        df['Platform'] = 'DoorDash'
        df['Revenue'] = pd.to_numeric(df['净总计'], errors='coerce')
        df['Is_Completed'] = df['最终订单状态'].str.contains('Delivered|delivered', case=False, na=False)
        df['Is_Cancelled'] = df['最终订单状态'].str.contains('Cancelled|cancelled', case=False, na=False)
        df['Store_Name'] = df['店铺名称'] if '店铺名称' in df.columns else 'Unknown'
        df = df[df['Date'].notna() & df['Revenue'].notna() & (df['Revenue'] > 0)]
        return df[['Date', 'Platform', 'Revenue', 'Is_Completed', 'Is_Cancelled', 'Store_Name']]
    except Exception as e:
        st.error(f"DoorDash processing error: {e}")
        return pd.DataFrame()

@st.cache_data
def process_uber_data(df):
    """Process Uber data"""
    try:
        df['Date'] = pd.to_datetime(df['订单日期'], errors='coerce')
        df['Platform'] = 'Uber'
        df['Revenue'] = pd.to_numeric(df['收入总额'], errors='coerce')
        df['Is_Completed'] = df['订单状态'].str.contains('已完成|completed', case=False, na=False)
        df['Is_Cancelled'] = df['订单状态'].str.contains('已取消|cancelled', case=False, na=False)
        df['Store_Name'] = df['餐厅名称'] if '餐厅名称' in df.columns else 'Unknown'
        df = df[df['Date'].notna() & df['Revenue'].notna() & (df['Revenue'] > 0)]
        return df[['Date', 'Platform', 'Revenue', 'Is_Completed', 'Is_Cancelled', 'Store_Name']]
    except Exception as e:
        st.error(f"Uber processing error: {e}")
        return pd.DataFrame()

@st.cache_data
def process_grubhub_data(df):
    """Process Grubhub data"""
    try:
        df['Date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['Platform'] = 'Grubhub'
        df['Revenue'] = pd.to_numeric(df['merchant_net_total'], errors='coerce')
        df['Is_Completed'] = df['transaction_type'].str.contains('Prepaid|Order', case=False, na=False)
        df['Is_Cancelled'] = False
        df['Store_Name'] = df['store_name'] if 'store_name' in df.columns else 'Unknown'
        df = df[df['Date'].notna() & df['Revenue'].notna() & (df['Revenue'] > 0)]
        return df[['Date', 'Platform', 'Revenue', 'Is_Completed', 'Is_Cancelled', 'Store_Name']]
    except Exception as e:
        st.error(f"Grubhub processing error: {e}")
        return pd.DataFrame()

def calculate_metrics(df):
    """Calculate key marketing metrics"""
    completed_df = df[df['Is_Completed']].copy()
    
    # Calculate growth rates
    monthly_revenue = completed_df.groupby(completed_df['Date'].dt.to_period('M'))['Revenue'].sum()
    revenue_growth = 0
    if len(monthly_revenue) >= 2:
        revenue_growth = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]) / monthly_revenue.iloc[-2]) * 100
    
    monthly_orders = completed_df.groupby(completed_df['Date'].dt.to_period('M')).size()
    order_growth = 0
    if len(monthly_orders) >= 2:
        order_growth = ((monthly_orders.iloc[-1] - monthly_orders.iloc[-2]) / monthly_orders.iloc[-2]) * 100
    
    return {
        'Total_Orders': len(completed_df),
        'Total_Revenue': completed_df['Revenue'].sum(),
        'AOV': completed_df['Revenue'].mean(),
        'Completion_Rate': df['Is_Completed'].mean() * 100,
        'Cancellation_Rate': df['Is_Cancelled'].mean() * 100,
        'Revenue_Growth': revenue_growth,
        'Order_Growth': order_growth
    }

def main():
    # Header
    st.markdown("""
        <div class="luckin-header">
            <h1>☕ Luckin Coffee - Advanced Marketing Analytics Dashboard</h1>
            <p style="font-size:18px;">瑞幸咖啡高级营销分析仪表板</p>
            <p>Comprehensive Customer Attribution, Retention & Marketing Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("### 📁 数据上传 (Data Upload)")
    
    uber_file = st.sidebar.file_uploader("📱 Uber Eats CSV", type="csv", key="uber_file_uploader")
    doordash_file = st.sidebar.file_uploader("🚗 DoorDash CSV", type="csv", key="doordash_file_uploader")
    grubhub_file = st.sidebar.file_uploader("🍔 Grubhub CSV", type="csv", key="grubhub_file_uploader")
    
    # Process files
    dfs = []
    platforms = []
    
    if uber_file:
        try:
            uber_df = pd.read_csv(uber_file)
            uber_processed = process_uber_data(uber_df)
            if not uber_processed.empty:
                dfs.append(uber_processed)
                platforms.append("Uber")
                st.sidebar.success("✅ Uber data loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Uber error: {e}")
    
    if doordash_file:
        try:
            doordash_df = pd.read_csv(doordash_file)
            doordash_processed = process_doordash_data(doordash_df)
            if not doordash_processed.empty:
                dfs.append(doordash_processed)
                platforms.append("DoorDash")
                st.sidebar.success("✅ DoorDash data loaded")
        except Exception as e:
            st.sidebar.error(f"❌ DoorDash error: {e}")
    
    if grubhub_file:
        try:
            grubhub_df = pd.read_csv(grubhub_file)
            grubhub_processed = process_grubhub_data(grubhub_df)
            if not grubhub_processed.empty:
                dfs.append(grubhub_processed)
                platforms.append("Grubhub")
                st.sidebar.success("✅ Grubhub data loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Grubhub error: {e}")
    
    if not dfs:
        st.warning("👆 Please upload at least one CSV file to begin analysis")
        st.info("""
        **Supported Platforms:**
        - 🍔 Grubhub: Transaction reports
        - 📱 Uber Eats: Revenue reports
        - 🚗 DoorDash: Order history
        """)
        return
    
    # Combine data
    df = pd.concat(dfs, ignore_index=True)
    completed_df = df[df['Is_Completed']].copy()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Platforms:** {', '.join(platforms)}")
    st.sidebar.markdown(f"**Total Records:** {len(df):,}")
    st.sidebar.markdown(f"**Date Range:** {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Create Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", "💰 Revenue", "🏆 Performance", 
        "⚡ Operations", "📈 Growth", "🎯 Attribution", "🔄 Retention"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.markdown("### 📊 Platform Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Orders", f"{metrics['Total_Orders']:,}", 
                     delta=f"{metrics['Order_Growth']:.1f}% MoM" if metrics['Order_Growth'] != 0 else None)
        with col2:
            st.metric("Total Revenue", f"${metrics['Total_Revenue']:,.2f}",
                     delta=f"{metrics['Revenue_Growth']:.1f}% MoM" if metrics['Revenue_Growth'] != 0 else None)
        with col3:
            st.metric("Average Order Value", f"${metrics['AOV']:.2f}")
        with col4:
            st.metric("Completion Rate", f"{metrics['Completion_Rate']:.1f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            platform_orders = completed_df['Platform'].value_counts()
            fig1 = px.pie(values=platform_orders.values, names=platform_orders.index,
                         title="Orders Distribution by Platform",
                         color_discrete_map=PLATFORM_COLORS)
            st.plotly_chart(fig1, use_container_width=True, key='tab1_orders_pie')
        
        with col2:
            platform_revenue = completed_df.groupby('Platform')['Revenue'].sum().sort_values(ascending=False)
            fig2 = px.bar(x=platform_revenue.index, y=platform_revenue.values,
                         title="Revenue by Platform",
                         color=platform_revenue.index,
                         color_discrete_map=PLATFORM_COLORS)
            fig2.update_layout(showlegend=False, xaxis_title="Platform", yaxis_title="Revenue ($)")
            st.plotly_chart(fig2, use_container_width=True, key='tab1_revenue_bar')
        
        st.markdown("### 📈 Platform Performance Summary")
        summary = completed_df.groupby('Platform').agg({
            'Revenue': ['count', 'sum', 'mean'],
            'Date': ['min', 'max']
        }).round(2)
        summary.columns = ['Orders', 'Total Revenue ($)', 'AOV ($)', 'First Order', 'Last Order']
        st.dataframe(summary, use_container_width=True)
    
    # TAB 2: REVENUE
    with tab2:
        st.markdown("### 💰 Advanced Revenue Analytics")
        
        daily_revenue = completed_df.groupby(['Date', 'Platform'])['Revenue'].sum().reset_index()
        fig3 = px.line(daily_revenue, x='Date', y='Revenue', color='Platform',
                      title="Daily Revenue Trend by Platform",
                      color_discrete_map=PLATFORM_COLORS)
        fig3.update_layout(hovermode='x unified')
        st.plotly_chart(fig3, use_container_width=True, key='tab2_daily_revenue_trend')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig4 = px.box(completed_df, x='Platform', y='Revenue', color='Platform',
                         title="Revenue Distribution by Platform",
                         color_discrete_map=PLATFORM_COLORS)
            fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, use_container_width=True, key='tab2_revenue_box')
        
        with col2:
            completed_df['Day_of_Week'] = completed_df['Date'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_revenue = completed_df.groupby(['Day_of_Week', 'Platform'])['Revenue'].sum().reset_index()
            weekly_revenue['Day_of_Week'] = pd.Categorical(weekly_revenue['Day_of_Week'], 
                                                           categories=day_order, ordered=True)
            weekly_revenue = weekly_revenue.sort_values('Day_of_Week')
            
            fig5 = px.bar(weekly_revenue, x='Day_of_Week', y='Revenue', color='Platform',
                         title="Revenue by Day of Week",
                         color_discrete_map=PLATFORM_COLORS)
            st.plotly_chart(fig5, use_container_width=True, key='tab2_weekly_revenue')
        
        monthly_revenue = completed_df.groupby([completed_df['Date'].dt.to_period('M'), 'Platform'])['Revenue'].sum().reset_index()
        monthly_revenue['Date'] = monthly_revenue['Date'].astype(str)
        fig6 = px.bar(monthly_revenue, x='Date', y='Revenue', color='Platform',
                     title="Monthly Revenue Comparison",
                     color_discrete_map=PLATFORM_COLORS)
        st.plotly_chart(fig6, use_container_width=True, key='tab2_monthly_revenue')
    
    # TAB 3: PERFORMANCE
    with tab3:
        st.markdown("### 🏆 Platform Performance Deep Dive")
        
        col1, col2 = st.columns(2)
        
        with col1:
            completion_rates = df.groupby('Platform')['Is_Completed'].mean() * 100
            fig7 = px.bar(x=completion_rates.index, y=completion_rates.values,
                         color=completion_rates.index,
                         title="Order Completion Rate by Platform",
                         color_discrete_map=PLATFORM_COLORS)
            fig7.update_layout(showlegend=False, yaxis_title="Completion Rate (%)")
            st.plotly_chart(fig7, use_container_width=True, key='tab3_completion_rates')
        
        with col2:
            aov_by_platform = completed_df.groupby('Platform')['Revenue'].mean()
            fig8 = px.bar(x=aov_by_platform.index, y=aov_by_platform.values,
                         color=aov_by_platform.index,
                         title="Average Order Value by Platform",
                         color_discrete_map=PLATFORM_COLORS)
            fig8.update_layout(showlegend=False, yaxis_title="AOV ($)")
            st.plotly_chart(fig8, use_container_width=True, key='tab3_aov_comparison')
        
        daily_orders = completed_df.groupby(['Date', 'Platform']).size().reset_index(name='Orders')
        fig9 = px.line(daily_orders, x='Date', y='Orders', color='Platform',
                      title="Daily Order Volume by Platform",
                      color_discrete_map=PLATFORM_COLORS)
        st.plotly_chart(fig9, use_container_width=True, key='tab3_order_volume')
        
        st.markdown("### 📊 Store Performance Analysis")
        store_perf = completed_df.groupby(['Store_Name', 'Platform']).agg({
            'Revenue': ['sum', 'count', 'mean']
        }).round(2)
        store_perf.columns = ['Total_Revenue', 'Order_Count', 'AOV']
        store_perf = store_perf.reset_index()
        
        heatmap_data = store_perf.pivot_table(index='Store_Name', columns='Platform', 
                                              values='Total_Revenue', fill_value=0)
        fig10 = px.imshow(heatmap_data, title="Store Revenue Heatmap by Platform",
                         aspect="auto", color_continuous_scale='Blues')
        st.plotly_chart(fig10, use_container_width=True, key='tab3_store_heatmap')
    
    # TAB 4: OPERATIONS
    with tab4:
        st.markdown("### ⚡ Operational Intelligence")
        
        completed_df['Hour'] = completed_df['Date'].dt.hour
        completed_df['Day_Name'] = completed_df['Date'].dt.day_name()
        
        hour_day_orders = completed_df.groupby(['Hour', 'Day_Name']).size().unstack(fill_value=0)
        fig11 = px.imshow(hour_day_orders.T,
                         labels=dict(x="Hour of Day", y="Day of Week", color="Orders"),
                         title="Order Patterns: Hour vs Day of Week",
                         color_continuous_scale='Blues')
        st.plotly_chart(fig11, use_container_width=True, key='tab4_hour_heatmap')
        
        col1, col2 = st.columns(2)
        
        with col1:
            hourly_orders = completed_df.groupby('Hour').size()
            fig12 = px.bar(x=hourly_orders.index, y=hourly_orders.values,
                          title="Orders by Hour of Day",
                          labels={'x': 'Hour', 'y': 'Orders'})
            st.plotly_chart(fig12, use_container_width=True, key='tab4_hourly_orders')
        
        with col2:
            cancel_rate = df.groupby('Platform')['Is_Cancelled'].mean() * 100
            fig13 = px.bar(x=cancel_rate.index, y=cancel_rate.values,
                          color=cancel_rate.index,
                          title="Cancellation Rate by Platform",
                          color_discrete_map=PLATFORM_COLORS)
            fig13.update_layout(showlegend=False, yaxis_title="Cancellation Rate (%)")
            st.plotly_chart(fig13, use_container_width=True, key='tab4_cancellation')
        
        st.markdown("### 📊 Peak Performance Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            peak_hour = completed_df.groupby('Hour').size().idxmax()
            st.metric("Peak Hour", f"{peak_hour}:00")
        with col2:
            peak_day = completed_df.groupby('Day_of_Week').size().idxmax()
            st.metric("Peak Day", peak_day)
        with col3:
            avg_daily_orders = completed_df.groupby('Date').size().mean()
            st.metric("Avg Daily Orders", f"{avg_daily_orders:.0f}")
    
    # TAB 5: GROWTH
    with tab5:
        st.markdown("### 📈 Growth Metrics & Trends")
        
        daily_agg_revenue = completed_df.groupby('Date')['Revenue'].sum().reset_index()
        daily_agg_orders = completed_df.groupby('Date').size().reset_index(name='Orders')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig14 = px.line(daily_agg_orders, x='Date', y='Orders', title="Daily Orders Trend")
            fig14.add_scatter(x=daily_agg_orders['Date'], 
                            y=daily_agg_orders['Orders'].rolling(7).mean(),
                            mode='lines', name='7-day MA', line=dict(color='red', dash='dash'))
            st.plotly_chart(fig14, use_container_width=True, key='tab5_daily_orders')
        
        with col2:
            fig15 = px.line(daily_agg_revenue, x='Date', y='Revenue', title="Daily Revenue Trend")
            fig15.add_scatter(x=daily_agg_revenue['Date'], 
                            y=daily_agg_revenue['Revenue'].rolling(7).mean(),
                            mode='lines', name='7-day MA', line=dict(color='red', dash='dash'))
            st.plotly_chart(fig15, use_container_width=True, key='tab5_daily_revenue')
        
        daily_agg_orders['Orders_Growth'] = daily_agg_orders['Orders'].pct_change() * 100
        daily_agg_revenue['Revenue_Growth'] = daily_agg_revenue['Revenue'].pct_change() * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig16 = px.bar(daily_agg_orders.dropna(), x='Date', y='Orders_Growth',
                          title="Daily Orders Growth Rate (%)")
            st.plotly_chart(fig16, use_container_width=True, key='tab5_order_growth')
        
        with col2:
            fig17 = px.bar(daily_agg_revenue.dropna(), x='Date', y='Revenue_Growth',
                          title="Daily Revenue Growth Rate (%)")
            st.plotly_chart(fig17, use_container_width=True, key='tab5_revenue_growth')
        
        st.markdown("### 🔮 Growth Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_growth = daily_agg_revenue['Revenue_Growth'].mean()
            st.metric("Avg Daily Growth", f"{avg_growth:.2f}%")
        with col2:
            best_day = daily_agg_revenue.loc[daily_agg_revenue['Revenue'].idxmax()]
            st.metric("Best Revenue Day", f"${best_day['Revenue']:,.2f}")
        with col3:
            total_growth = ((daily_agg_revenue['Revenue'].iloc[-1] - daily_agg_revenue['Revenue'].iloc[0]) / 
                          daily_agg_revenue['Revenue'].iloc[0] * 100)
            st.metric("Total Period Growth", f"{total_growth:.1f}%")
    
    # TAB 6: ATTRIBUTION
    with tab6:
        st.markdown("### 🎯 Customer Attribution & Segmentation")
        
        current_date = completed_df['Date'].max()
        rfm = completed_df.groupby('Store_Name').agg({
            'Date': lambda x: (current_date - x.max()).days,
            'Revenue': ['count', 'sum']
        }).round(2)
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig18 = px.scatter(rfm, x='Frequency', y='Monetary', size='Recency',
                              title="Customer Value Segmentation (RFM)",
                              labels={'Frequency': 'Purchase Frequency', 
                                     'Monetary': 'Total Spend ($)',
                                     'Recency': 'Days Since Last Order'})
            st.plotly_chart(fig18, use_container_width=True, key='tab6_rfm_scatter')
        
        with col2:
            clv_data = completed_df.groupby('Store_Name').agg({
                'Revenue': ['sum', 'mean', 'count'],
                'Date': ['min', 'max']
            }).round(2)
            clv_data.columns = ['Total_Spend', 'AOV', 'Orders', 'First_Date', 'Last_Date']
            clv_data['Days_Active'] = (clv_data['Last_Date'] - clv_data['First_Date']).dt.days + 1
            clv_data['Order_Frequency'] = clv_data['Orders'] / clv_data['Days_Active'] * 30
            clv_data['Est_CLV'] = clv_data['AOV'] * clv_data['Order_Frequency'] * 12
            
            fig19 = px.histogram(clv_data, x='Est_CLV', nbins=20,
                                title="Customer Lifetime Value Distribution")
            st.plotly_chart(fig19, use_container_width=True, key='tab6_clv_hist')
        
        st.markdown("### 💡 Marketing Insights")
        col1, col2, col3 = st.columns(3)
        with col1:
            high_value = (clv_data['Est_CLV'] > clv_data['Est_CLV'].quantile(0.75)).sum()
            st.metric("High-Value Customers", high_value)
        with col2:
            avg_clv = clv_data['Est_CLV'].mean()
            st.metric("Average CLV", f"${avg_clv:,.2f}")
        with col3:
            repeat_rate = (clv_data['Orders'] > 1).mean() * 100
            st.metric("Repeat Purchase Rate", f"{repeat_rate:.1f}%")
    
    # TAB 7: RETENTION
    with tab7:
        st.markdown("### 🔄 Customer Retention & Attrition Analysis")
        
        churn_data = completed_df.groupby('Store_Name')['Date'].max().reset_index()
        churn_data['Days_Since_Last_Order'] = (current_date - churn_data['Date']).dt.days
        churn_data['Is_Churned'] = churn_data['Days_Since_Last_Order'] > 30
        
        churn_rate = churn_data['Is_Churned'].mean() * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Rate", f"{churn_rate:.1f}%", 
                     help="Stores with no orders in last 30 days")
        with col2:
            active = (~churn_data['Is_Churned']).sum()
            st.metric("Active Stores", active)
        with col3:
            avg_days = churn_data['Days_Since_Last_Order'].mean()
            st.metric("Avg Days Since Order", f"{avg_days:.0f}")
        
        st.markdown("### ⚠️ Churn Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig20 = px.histogram(churn_data, x='Days_Since_Last_Order', nbins=20,
                                title="Days Since Last Order Distribution")
            fig20.add_vline(x=30, line_dash="dash", line_color="red",
                           annotation_text="Churn Threshold")
            st.plotly_chart(fig20, use_container_width=True, key='tab7_churn_dist')
        
        with col2:
            store_platform = completed_df.groupby('Store_Name')['Platform'].first().reset_index()
            churn_platform = churn_data.merge(store_platform, on='Store_Name')
            platform_churn = churn_platform.groupby('Platform')['Is_Churned'].mean() * 100
            
            fig21 = px.bar(x=platform_churn.index, y=platform_churn.values,
                          color=platform_churn.index,
                          title="Churn Rate by Platform",
                          color_discrete_map=PLATFORM_COLORS)
            fig21.update_layout(showlegend=False, yaxis_title="Churn Rate (%)")
            st.plotly_chart(fig21, use_container_width=True, key='tab7_platform_churn')
        
        st.markdown("### 📊 Cohort Analysis")
        
        monthly_cohort = completed_df.groupby(completed_df['Date'].dt.to_period('M')).agg({
            'Store_Name': 'nunique',
            'Revenue': 'sum'
        }).reset_index()
        monthly_cohort['Date'] = monthly_cohort['Date'].astype(str)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig22 = px.line(monthly_cohort, x='Date', y='Store_Name',
                           title="Monthly Active Stores")
            fig22.update_layout(yaxis_title="Number of Stores")
            st.plotly_chart(fig22, use_container_width=True, key='tab7_monthly_stores')
        
        with col2:
            fig23 = px.line(monthly_cohort, x='Date', y='Revenue',
                           title="Monthly Revenue from Active Stores")
            fig23.update_layout(yaxis_title="Revenue ($)")
            st.plotly_chart(fig23, use_container_width=True, key='tab7_monthly_revenue')
        
        st.markdown("### 🎯 Customer Segmentation")
        
        if len(clv_data) >= 3:
            try:
                features = clv_data[['Total_Spend', 'AOV', 'Orders']].fillna(0)
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                n_clusters = min(4, len(features))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clv_data['Segment'] = kmeans.fit_predict(features_scaled)
                
                fig24 = px.scatter(clv_data, x='Total_Spend', y='AOV',
                                  color='Segment', size='Orders',
                                  title="Customer Behavioral Segmentation",
                                  labels={'Total_Spend': 'Total Spend ($)', 
                                         'AOV': 'Average Order Value ($)'})
                st.plotly_chart(fig24, use_container_width=True, key='tab7_segmentation')
                
                segment_summary = clv_data.groupby('Segment').agg({
                    'Total_Spend': 'mean',
                    'AOV': 'mean',
                    'Orders': 'mean'
                }).round(2)
                segment_summary.index = [f"Segment {i+1}" for i in segment_summary.index]
                
                st.markdown("### 📋 Segment Summary")
                st.dataframe(segment_summary, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Segmentation analysis requires more data: {e}")
    
    # Export functionality
    st.markdown("---")
    st.markdown("### 📤 Export Analytics Report")
    
    if st.button("Generate Excel Report", key="export_button"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_excel(writer, sheet_name='Summary', index=False)
            
            platform_summary = completed_df.groupby('Platform').agg({
                'Revenue': ['count', 'sum', 'mean']
            }).round(2)
            platform_summary.to_excel(writer, sheet_name='Platform_Performance')
            
            daily_agg_revenue.to_excel(writer, sheet_name='Daily_Revenue', index=False)
        
        st.download_button(
            label="📥 Download Report",
            data=output.getvalue(),
            file_name=f"luckin_analytics_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )
    
    # Help section
    with st.expander("📋 How to Use This Dashboard"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Upload Data**: Use the sidebar to upload CSV files from Uber, DoorDash, or Grubhub
        2. **Explore Analytics**: Navigate through 7 comprehensive tabs
        3. **Export Reports**: Generate Excel reports for stakeholders
        
        **Features:**
        - 📊 **Overview**: Key metrics and platform distribution
        - 💰 **Revenue**: Advanced revenue analytics and trends
        - 🏆 **Performance**: Platform and store performance metrics
        - ⚡ **Operations**: Operational intelligence and peak hours
        - 📈 **Growth**: Growth tracking and forecasting
        - 🎯 **Attribution**: Customer segmentation and RFM analysis
        - 🔄 **Retention**: Churn analysis and cohort tracking
        """)

if __name__ == "__main__":
    main()
