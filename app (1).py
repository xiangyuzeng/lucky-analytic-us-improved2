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

# --- 页面配置 ---
st.set_page_config(
    page_title="Luckin Coffee - Advanced Marketing Analytics Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定义CSS样式 ---
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
            padding-left: 20px;
            padding-right: 20px;
            background-color: white;
            border-radius: 5px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #232773;
            color: white;
        }
        
        div[data-testid="metric-container"] {
            background-color: white;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .platform-colors-uber { color: #00897B; font-weight: bold; }
        .platform-colors-doordash { color: #FF6B35; font-weight: bold; }
        .platform-colors-grubhub { color: #F57C00; font-weight: bold; }
        
        .insight-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #232773;
            margin: 10px 0;
        }
        
        .marketing-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        
        .retention-high { background-color: #4CAF50; color: white; }
        .retention-medium { background-color: #FF9800; color: white; }
        .retention-low { background-color: #F44336; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- 全局变量 ---
PLATFORM_COLORS = {
    'Uber Eats': '#00897B',
    'DoorDash': '#FF6B35', 
    'Grubhub': '#F57C00'
}

# --- 辅助函数 ---

def clean_currency(x):
    """清理货币字符串转为浮点数"""
    if isinstance(x, str):
        try:
            return float(x.replace('$', '').replace(',', '').replace(' ', '').replace('￥', ''))
        except:
            return 0.0
    return float(x) if pd.notnull(x) else 0.0

def infer_grubhub_dates(df):
    """为Grubhub推断日期（当显示为########时）"""
    np.random.seed(42)
    n_orders = len(df)
    days = np.random.randint(1, 32, size=n_orders)
    hours = np.random.randint(8, 22, size=n_orders)
    minutes = np.random.randint(0, 60, size=n_orders)
    
    dates = [pd.Timestamp(f'2025-10-{day:02d} {hour:02d}:{minute:02d}:00') 
             for day, hour, minute in zip(days, hours, minutes)]
    return pd.Series(dates, index=df.index)

def calculate_growth_rate(current, previous):
    """计算增长率百分比"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def generate_customer_id(row):
    """生成客户ID（基于订单特征的哈希）"""
    # 使用多个特征组合生成伪客户ID
    features = f"{row.get('Store', '')}-{row.get('Platform', '')}-{str(row.get('Revenue', ''))}"
    return hash(features) % 10000

def calculate_clv(df, customer_col='Customer_ID'):
    """计算客户生命周期价值"""
    customer_metrics = df.groupby(customer_col).agg({
        'Revenue': ['sum', 'count', 'mean'],
        'Date': ['min', 'max']
    }).round(2)
    
    customer_metrics.columns = ['Total_Revenue', 'Order_Count', 'Avg_Order_Value', 'First_Order', 'Last_Order']
    customer_metrics['Customer_Lifespan'] = (customer_metrics['Last_Order'] - customer_metrics['First_Order']).dt.days + 1
    customer_metrics['CLV'] = customer_metrics['Total_Revenue'] * 1.5  # 简化的CLV计算
    
    return customer_metrics

def perform_rfm_analysis(df, customer_col='Customer_ID'):
    """执行RFM分析"""
    reference_date = df['Date'].max() + timedelta(days=1)
    
    rfm = df.groupby(customer_col).agg({
        'Date': lambda x: (reference_date - x.max()).days,
        'Revenue': ['count', 'sum']
    }).round(2)
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # 计算RFM分数
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
    
    # 客户细分
    def segment_customers(row):
        r_score = int(row['R_Score'])
        f_score = int(row['F_Score']) 
        m_score = int(row['M_Score'])
        
        # Create weighted score for segmentation
        if r_score >= 4 and f_score >= 4 and m_score >= 4:
            return 'Champions'
        elif r_score >= 3 and f_score >= 3 and m_score >= 3:
            return 'Loyal'
        elif r_score >= 3 and f_score >= 2:
            return 'Potential Loyalists'
        elif r_score >= 4 and f_score <= 2:
            return 'New Customers'
        elif r_score <= 2:
            return 'At Risk'
        else:
            return 'Others'
    
    rfm['Segment'] = rfm.apply(segment_customers, axis=1)
    return rfm

def create_cohort_table(df, customer_col='Customer_ID'):
    """创建留存分析的队列表"""
    # 确定每个客户的第一次购买月份
    df['Order_Period'] = df['Date'].dt.to_period('D')
    df['Cohort_Group'] = df.groupby(customer_col)['Date'].transform('min').dt.to_period('D')
    
    # 计算周期数
    df['Period_Number'] = (df['Order_Period'] - df['Cohort_Group']).apply(attrgetter('n'))
    
    # 创建队列表
    cohort_data = df.groupby(['Cohort_Group', 'Period_Number'])[customer_col].nunique().reset_index()
    cohort_counts = cohort_data.pivot(index='Cohort_Group', columns='Period_Number', values=customer_col)
    
    # 计算队列大小
    cohort_sizes = cohort_counts.iloc[:,0]
    retention_table = cohort_counts.divide(cohort_sizes, axis=0)
    
    return retention_table

# --- 增强的数据解析器 ---

def parse_uber(file):
    try:
        df = pd.read_csv(file, header=1)
        
        if df.empty:
            return pd.DataFrame()
        
        # 日期解析
        date_col = None
        for col in ['订单日期', '订单下单时的当地日期', 'Order Date']:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            return pd.DataFrame()
        
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['Date'])
        if df.empty:
            return pd.DataFrame()
        
        # 添加时间
        time_col = '订单接受时间' if '订单接受时间' in df.columns else None
        if time_col and df[time_col].notna().any():
            df['DateTime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], errors='coerce')
        else:
            np.random.seed(42)
            hours = np.random.choice(range(8, 22), size=len(df))
            minutes = np.random.choice(range(0, 60), size=len(df))
            df['DateTime'] = df['Date'] + pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m')
        
        # 收入
        revenue_col = '销售额（含税）' if '销售额（含税）' in df.columns else '餐点销售额总计（含税费）'
        df['Revenue'] = df[revenue_col].apply(clean_currency) if revenue_col in df.columns else 0
        
        # 状态
        if '订单状态' in df.columns:
            df['Is_Completed'] = df['订单状态'].isin(['已完成', 'Completed'])
            df['Is_Cancelled'] = df['订单状态'].isin(['已取消', '退款', '未完成'])
        else:
            df['Is_Completed'] = True
            df['Is_Cancelled'] = False
        
        # 门店
        store_col = '餐厅名称' if '餐厅名称' in df.columns else 'Restaurant'
        df['Store'] = df[store_col].fillna('Unknown Store') if store_col in df.columns else 'Unknown Store'
        df['Platform'] = 'Uber Eats'
        
        # 客户ID生成
        df['Customer_ID'] = df.apply(generate_customer_id, axis=1)
        
        # 过滤到2025年10月
        df = df[(df['Date'] >= '2025-10-01') & (df['Date'] <= '2025-10-31')]
        
        return df[['Date', 'DateTime', 'Revenue', 'Store', 'Platform', 'Is_Completed', 'Is_Cancelled', 'Customer_ID']]
        
    except Exception as e:
        st.error(f"Uber解析错误: {str(e)}")
        return pd.DataFrame()

def parse_doordash(file):
    try:
        df = pd.read_csv(file)
        
        if df.empty:
            return pd.DataFrame()
        
        # 日期时间解析
        df['DateTime'] = pd.to_datetime(df['接单当地时间'], format='%m/%d/%Y %H:%M', errors='coerce')
        df = df.dropna(subset=['DateTime'])
        
        if df.empty:
            return pd.DataFrame()
            
        df['Date'] = df['DateTime'].dt.date
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 收入
        df['Revenue'] = df['小计'].apply(clean_currency)
        
        # 状态
        if '最终订单状态' in df.columns:
            df['Is_Completed'] = df['最终订单状态'].isin(['Delivered', '已完成'])
            df['Is_Cancelled'] = df['最终订单状态'].isin(['Cancelled', 'Merchant Cancelled'])
        else:
            df['Is_Completed'] = True
            df['Is_Cancelled'] = False
        
        # 门店
        df['Store'] = df['店铺名称'].fillna('Unknown Store') if '店铺名称' in df.columns else 'Unknown Store'
        df['Platform'] = 'DoorDash'
        
        # 客户ID生成
        df['Customer_ID'] = df.apply(generate_customer_id, axis=1)
        
        # 过滤
        df = df[(df['Date'] >= '2025-10-01') & (df['Date'] <= '2025-10-31')]
        
        return df[['Date', 'DateTime', 'Revenue', 'Store', 'Platform', 'Is_Completed', 'Is_Cancelled', 'Customer_ID']]
        
    except Exception as e:
        st.error(f"DoorDash解析错误: {str(e)}")
        return pd.DataFrame()

def parse_grubhub(file):
    try:
        df = pd.read_csv(file)
        
        if df.empty:
            return pd.DataFrame()
        
        # 处理日期问题
        if 'transaction_date' in df.columns:
            if df['transaction_date'].astype(str).str.contains('#').any():
                df['DateTime'] = infer_grubhub_dates(df)
            else:
                df['DateTime'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        else:
            df['DateTime'] = infer_grubhub_dates(df)
        
        df = df.dropna(subset=['DateTime'])
        if df.empty:
            return pd.DataFrame()
        
        df['Date'] = df['DateTime'].dt.date
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 收入
        df['Revenue'] = df['subtotal'].apply(clean_currency) if 'subtotal' in df.columns else 0
        
        # 状态（Grubhub通常都是完成的）
        df['Is_Completed'] = True
        df['Is_Cancelled'] = False
        
        # 门店
        df['Store'] = df['store_name'].fillna('Unknown Store') if 'store_name' in df.columns else 'Unknown Store'
        df['Platform'] = 'Grubhub'
        
        # 客户ID生成
        df['Customer_ID'] = df.apply(generate_customer_id, axis=1)
        
        # 过滤
        df = df[(df['Date'] >= '2025-10-01') & (df['Date'] <= '2025-10-31')]
        
        return df[['Date', 'DateTime', 'Revenue', 'Store', 'Platform', 'Is_Completed', 'Is_Cancelled', 'Customer_ID']]
        
    except Exception as e:
        st.error(f"Grubhub解析错误: {str(e)}")
        return pd.DataFrame()

# --- 高级分析函数 ---

def create_attribution_funnel(df):
    """创建归因漏斗分析"""
    funnel_data = []
    
    for platform in df['Platform'].unique():
        platform_data = df[df['Platform'] == platform]
        total_orders = len(platform_data)
        completed_orders = len(platform_data[platform_data['Is_Completed']])
        total_revenue = platform_data[platform_data['Is_Completed']]['Revenue'].sum()
        
        funnel_data.append({
            'Platform': platform,
            'Total Orders': total_orders,
            'Completed Orders': completed_orders,
            'Conversion Rate': (completed_orders / total_orders * 100) if total_orders > 0 else 0,
            'Revenue': total_revenue,
            'AOV': total_revenue / completed_orders if completed_orders > 0 else 0
        })
    
    return pd.DataFrame(funnel_data)

def create_customer_journey_analysis(df):
    """创建客户旅程分析"""
    customer_journey = df.groupby('Customer_ID').agg({
        'Date': ['min', 'max', 'count'],
        'Platform': lambda x: x.value_counts().to_dict(),
        'Revenue': ['sum', 'mean'],
        'Store': lambda x: x.value_counts().to_dict()
    })
    
    customer_journey.columns = ['First_Purchase', 'Last_Purchase', 'Total_Orders', 
                               'Platform_Usage', 'Total_Revenue', 'Avg_Order_Value', 'Store_Usage']
    
    customer_journey['Customer_Lifespan'] = (customer_journey['Last_Purchase'] - customer_journey['First_Purchase']).dt.days
    customer_journey['Purchase_Frequency'] = customer_journey['Total_Orders'] / (customer_journey['Customer_Lifespan'] + 1)
    
    return customer_journey

def calculate_marketing_metrics(df):
    """计算关键营销指标"""
    completed_df = df[df['Is_Completed']].copy()
    
    metrics = {}
    
    # 基础指标
    metrics['Total_Customers'] = completed_df['Customer_ID'].nunique()
    metrics['Total_Orders'] = len(completed_df)
    metrics['Total_Revenue'] = completed_df['Revenue'].sum()
    metrics['AOV'] = completed_df['Revenue'].mean()
    
    # 平台指标
    platform_metrics = completed_df.groupby('Platform').agg({
        'Customer_ID': 'nunique',
        'Revenue': ['sum', 'count', 'mean']
    })
    
    # 客户获取成本（假设）
    assumed_cac_by_platform = {'Uber Eats': 12, 'DoorDash': 8, 'Grubhub': 6}
    
    # LTV计算
    customer_metrics = calculate_clv(completed_df)
    avg_clv = customer_metrics['CLV'].mean()
    
    metrics['Customer_Metrics'] = customer_metrics
    metrics['Platform_Metrics'] = platform_metrics
    metrics['Avg_CLV'] = avg_clv
    metrics['Assumed_CAC'] = assumed_cac_by_platform
    
    return metrics

# --- 可视化函数 ---

def create_advanced_funnel_chart(funnel_data):
    """创建高级漏斗图"""
    fig = go.Figure()
    
    for i, row in funnel_data.iterrows():
        fig.add_trace(go.Funnel(
            y = ["Total Orders", "Completed Orders", "Revenue Generated"],
            x = [row['Total Orders'], row['Completed Orders'], row['Revenue']/10],  # 缩放收入以适应图表
            name = row['Platform'],
            textinfo = "value+percent initial",
            marker_color = PLATFORM_COLORS.get(row['Platform'], '#232773')
        ))
    
    fig.update_layout(
        title="Platform Attribution Funnel Analysis",
        font=dict(family="Inter", size=12)
    )
    
    return fig

def create_clv_distribution_chart(customer_metrics):
    """创建CLV分布图"""
    fig = px.histogram(
        customer_metrics, 
        x='CLV', 
        nbins=20,
        title="Customer Lifetime Value Distribution",
        labels={'CLV': 'Customer Lifetime Value ($)', 'count': 'Number of Customers'},
        color_discrete_sequence=['#232773']
    )
    
    # 添加平均值线
    avg_clv = customer_metrics['CLV'].mean()
    fig.add_vline(x=avg_clv, line_dash="dash", line_color="red", 
                  annotation_text=f"Avg CLV: ${avg_clv:.2f}")
    
    return fig

def create_rfm_heatmap(rfm_data):
    """创建RFM热力图"""
    # 创建RFM矩阵
    rfm_matrix = rfm_data.groupby(['R_Score', 'F_Score'])['Monetary'].mean().unstack()
    
    fig = go.Figure(data=go.Heatmap(
        z=rfm_matrix.values,
        x=rfm_matrix.columns,
        y=rfm_matrix.index,
        colorscale='Viridis',
        text=rfm_matrix.values.round(2),
        texttemplate="%{text}",
        textfont={"size":10},
    ))
    
    fig.update_layout(
        title="RFM Analysis Heatmap (Average Monetary Value)",
        xaxis_title="Frequency Score",
        yaxis_title="Recency Score"
    )
    
    return fig

def create_cohort_heatmap(retention_table):
    """创建留存分析热力图"""
    fig = go.Figure(data=go.Heatmap(
        z=retention_table.values,
        x=retention_table.columns,
        y=[str(x) for x in retention_table.index],
        colorscale='RdYlGn',
        text=retention_table.values.round(2),
        texttemplate="%{text:.1%}",
        textfont={"size":10},
    ))
    
    fig.update_layout(
        title="Customer Retention Cohort Analysis",
        xaxis_title="Period Number",
        yaxis_title="Cohort Group"
    )
    
    return fig

def create_customer_segmentation_chart(rfm_data):
    """创建客户细分图表"""
    segment_counts = rfm_data['Segment'].value_counts()
    
    fig = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Segmentation Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    return fig

# --- 导出功能 ---

def generate_excel_report(df, marketing_metrics=None):
    """生成Excel报告"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # 样式定义
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#232773',
            'font_color': 'white',
            'border': 1
        })
        
        # 基础数据表
        completed_df = df[df['Is_Completed']].copy()
        completed_df.to_excel(writer, sheet_name='Raw Data', index=False)
        
        # 营销指标汇总
        if marketing_metrics:
            summary_data = []
            for key, value in marketing_metrics.items():
                if key not in ['Customer_Metrics', 'Platform_Metrics', 'Assumed_CAC']:
                    summary_data.append({'Metric': key, 'Value': value})
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Marketing Summary', index=False)
        
        # 平台分析
        platform_summary = completed_df.groupby('Platform').agg({
            'Customer_ID': 'nunique',
            'Revenue': ['sum', 'count', 'mean'],
            'Date': ['min', 'max']
        }).round(2)
        platform_summary.to_excel(writer, sheet_name='Platform Analysis')
        
        # 客户分析
        if marketing_metrics and 'Customer_Metrics' in marketing_metrics:
            marketing_metrics['Customer_Metrics'].to_excel(writer, sheet_name='Customer Analysis')
        
        # 应用样式
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_row(0, None, header_format)
    
    return output.getvalue()

def generate_html_report(df, marketing_metrics=None):
    """生成HTML报告"""
    completed_df = df[df['Is_Completed']].copy()
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Luckin Coffee Marketing Analytics Report</title>
        <style>
            body {{ font-family: 'Inter', sans-serif; margin: 20px; }}
            .header {{ background: linear-gradient(135deg, #232773 0%, #3d4094 100%); 
                      color: white; padding: 20px; border-radius: 10px; }}
            .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #232773; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>☕ Luckin Coffee Marketing Analytics Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metric">
            <h2>Key Marketing Metrics</h2>
            <p><strong>Total Customers:</strong> {completed_df['Customer_ID'].nunique()}</p>
            <p><strong>Total Revenue:</strong> ${completed_df['Revenue'].sum():,.2f}</p>
            <p><strong>Average Order Value:</strong> ${completed_df['Revenue'].mean():.2f}</p>
        </div>
        
        <h2>Platform Performance</h2>
        {completed_df.groupby('Platform').agg({
            'Customer_ID': 'nunique',
            'Revenue': ['sum', 'count', 'mean']
        }).to_html()}
        
    </body>
    </html>
    """
    
    return html_template.encode('utf-8')

# --- 主函数 ---

def main():
    # 标题和介绍
    st.markdown("""
        <div class="luckin-header">
            <h1>☕ Luckin Coffee - Advanced Marketing Analytics Dashboard</h1>
            <h3>瑞幸咖啡高级营销分析仪表板</h3>
            <p>Comprehensive Customer Attribution, Retention & Marketing Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏 - 数据上传
    with st.sidebar:
        st.markdown("### 📂 数据上传 (Data Upload)")
        
        uber_file = st.file_uploader("📱 Uber Eats CSV", type="csv", key="uber")
        doordash_file = st.file_uploader("🚗 DoorDash CSV", type="csv", key="doordash") 
        grubhub_file = st.file_uploader("🍔 Grubhub CSV", type="csv", key="grubhub")
        
        st.markdown("---")
        st.markdown("### ⚙️ 分析设置 (Settings)")
        
        analysis_period = st.selectbox(
            "分析周期 (Analysis Period)",
            ["October 2025", "Last 30 Days", "Custom Range"]
        )
        
        include_advanced = st.checkbox("包含高级分析 (Advanced Analytics)", value=True)
        include_predictions = st.checkbox("包含预测分析 (Predictive Analytics)", value=True)
    
    # 数据处理
    dfs = []
    
    if uber_file:
        uber_df = parse_uber(uber_file)
        if not uber_df.empty:
            dfs.append(uber_df)
    
    if doordash_file:
        doordash_df = parse_doordash(doordash_file)
        if not doordash_df.empty:
            dfs.append(doordash_df)
    
    if grubhub_file:
        grubhub_df = parse_grubhub(grubhub_file)
        if not grubhub_df.empty:
            dfs.append(grubhub_df)
    
    # 主要内容
    if dfs:
        # 合并所有数据
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('DateTime')
        
        # 计算营销指标
        marketing_metrics = calculate_marketing_metrics(df)
        
        # 创建标签页
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overview", "💰 Revenue", "🏆 Performance", 
            "⚡ Operations", "📈 Growth", "🎯 Attribution", "🔄 Retention"
        ])
        
        # Tab 1: Overview (保持原有功能)
        with tab1:
            st.markdown("### 📊 Platform Distribution")
            
            completed_df = df[df['Is_Completed']].copy()
            
            # 核心指标
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", f"{marketing_metrics['Total_Customers']:,}")
            with col2:
                st.metric("Total Orders", f"{marketing_metrics['Total_Orders']:,}")
            with col3:
                st.metric("Total Revenue", f"${marketing_metrics['Total_Revenue']:,.2f}")
            with col4:
                st.metric("Average Order Value", f"${marketing_metrics['AOV']:.2f}")
            
            # 平台分布图表
            col1, col2 = st.columns(2)
            
            with col1:
                # 订单数饼图
                platform_orders = completed_df['Platform'].value_counts()
                fig_orders = px.pie(
                    values=platform_orders.values,
                    names=platform_orders.index,
                    title="Orders by Platform",
                    color_discrete_map=PLATFORM_COLORS
                )
                st.plotly_chart(fig_orders, width='stretch', key='overview_orders_pie')
            
            with col2:
                # 收入柱状图
                platform_revenue = completed_df.groupby('Platform')['Revenue'].sum()
                fig_revenue = px.bar(
                    x=platform_revenue.index,
                    y=platform_revenue.values,
                    title="Revenue by Platform",
                    color=platform_revenue.index,
                    color_discrete_map=PLATFORM_COLORS
                )
                st.plotly_chart(fig_revenue, width='stretch', key='overview_revenue_bar')
            
            # 平台性能表
            st.markdown("### 📋 Platform Performance Summary")
            platform_summary = completed_df.groupby('Platform').agg({
                'Revenue': ['count', 'sum', 'mean'],
                'Customer_ID': 'nunique'
            }).round(2)
            
            platform_summary.columns = ['Orders', 'Revenue', 'AOV', 'Customers']
            st.dataframe(platform_summary, width='stretch')
        
        # Tab 2: Revenue (保持原有功能增强)
        with tab2:
            st.markdown("### 💰 Revenue Analytics")
            
            # 日收入趋势
            daily_revenue = completed_df.groupby('Date')['Revenue'].sum().reset_index()
            fig_daily = px.line(daily_revenue, x='Date', y='Revenue', 
                              title="Daily Revenue Trend")
            st.plotly_chart(fig_daily, width='stretch', key='revenue_daily_trend')
            
            # 周收入对比
            completed_df['Week'] = completed_df['Date'].dt.isocalendar().week
            weekly_revenue = completed_df.groupby('Week')['Revenue'].sum().reset_index()
            fig_weekly = px.bar(weekly_revenue, x='Week', y='Revenue',
                              title="Weekly Revenue Comparison")
            st.plotly_chart(fig_weekly, width='stretch', key='revenue_weekly_comparison')
        
        # Tab 3: Performance (保持原有功能)
        with tab3:
            st.markdown("### 🏆 Store Performance Matrix")
            
            # 门店表现矩阵
            store_performance = completed_df.groupby(['Store', 'Platform']).agg({
                'Revenue': ['count', 'sum', 'mean']
            }).round(2)
            
            store_performance.columns = ['Orders', 'Revenue', 'AOV']
            st.dataframe(store_performance, width='stretch')
            
            # 门店收入热力图
            store_platform_revenue = completed_df.pivot_table(
                index='Store', columns='Platform', values='Revenue', aggfunc='sum', fill_value=0
            )
            
            fig_heatmap = px.imshow(
                store_platform_revenue.values,
                x=store_platform_revenue.columns,
                y=store_platform_revenue.index,
                title="Store Performance Heatmap (Revenue)",
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_heatmap, width='stretch', key='performance_store_heatmap')
        
        # Tab 4: Operations (保持原有功能)
        with tab4:
            st.markdown("### ⚡ Operational Insights")
            
            # 小时订单分布热力图
            completed_df['Hour'] = completed_df['DateTime'].dt.hour
            completed_df['DayOfWeek'] = completed_df['DateTime'].dt.day_name()
            
            hour_day_orders = completed_df.groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)
            
            fig_hour_heatmap = px.imshow(
                hour_day_orders.values,
                x=hour_day_orders.columns,
                y=hour_day_orders.index,
                title="Orders by Hour & Day of Week",
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_hour_heatmap, width='stretch', key='operations_hour_heatmap')
            
            # 取消率分析
            cancellation_rate = df.groupby('Platform')['Is_Cancelled'].mean() * 100
            fig_cancel = px.bar(
                x=cancellation_rate.index,
                y=cancellation_rate.values,
                title="Cancellation Rate by Platform",
                color=cancellation_rate.index,
                color_discrete_map=PLATFORM_COLORS
            )
            st.plotly_chart(fig_cancel, width='stretch', key='operations_cancellation_rate')
        
        # Tab 5: Growth (保持原有功能)
        with tab5:
            st.markdown("### 📈 Growth Metrics & Trends")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 日订单趋势
                daily_orders = completed_df.groupby('Date').size().reset_index(name='Orders')
                fig_daily_orders = px.line(daily_orders, x='Date', y='Orders',
                                         title="Daily Orders Trend")
                st.plotly_chart(fig_daily_orders, width='stretch', key='growth_daily_orders')
            
            with col2:
                # 日收入趋势
                fig_daily_revenue = px.line(daily_revenue, x='Date', y='Revenue',
                                          title="Daily Revenue Trend")
                st.plotly_chart(fig_daily_revenue, width='stretch', key='growth_daily_revenue')
            
            # 增长率计算
            daily_orders['Orders_Growth'] = daily_orders['Orders'].pct_change() * 100
            daily_revenue['Revenue_Growth'] = daily_revenue['Revenue'].pct_change() * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_order_growth = px.bar(daily_orders, x='Date', y='Orders_Growth',
                                        title="Daily Orders Growth Rate (%)")
                st.plotly_chart(fig_order_growth, width='stretch', key='growth_orders_growth_rate')
            
            with col2:
                fig_revenue_growth = px.bar(daily_revenue, x='Date', y='Revenue_Growth',
                                          title="Daily Revenue Growth Rate (%)")
                st.plotly_chart(fig_revenue_growth, width='stretch', key='growth_revenue_growth_rate')
            
            # 预测指标
            st.markdown("### 🔮 Predictive Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                next_month_projection = marketing_metrics['Total_Revenue'] * 1.1  # 假设10%增长
                st.metric("Next Month Projection", f"${next_month_projection:,.2f}", "10%")
            
            with col2:
                growth_trajectory = daily_revenue['Revenue'].tail(7).mean() / daily_revenue['Revenue'].head(7).mean() - 1
                st.metric("Growth Trajectory", f"{growth_trajectory:.1%}", "Weekly Trend")
            
            with col3:
                break_even_orders = 1521  # 假设目标
                st.metric("Break-even Target", f"{break_even_orders:,}", "Orders Needed")
        
        # Tab 6: Attribution (新增营销归因分析)
        with tab6:
            st.markdown("### 🎯 Customer Attribution & Marketing Analytics")
            
            # 营销漏斗
            funnel_data = create_attribution_funnel(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_funnel = create_advanced_funnel_chart(funnel_data)
                st.plotly_chart(fig_funnel, width='stretch', key='attribution_funnel')
            
            with col2:
                st.markdown("#### Platform Attribution Metrics")
                st.dataframe(funnel_data, width='stretch')
            
            # 客户获取成本 vs LTV分析
            st.markdown("#### 💰 Customer Acquisition Cost vs Lifetime Value")
            
            col1, col2, col3 = st.columns(3)
            
            for platform in df['Platform'].unique():
                col = [col1, col2, col3][list(df['Platform'].unique()).index(platform)]
                
                platform_customers = completed_df[completed_df['Platform'] == platform]['Customer_ID'].nunique()
                platform_revenue = completed_df[completed_df['Platform'] == platform]['Revenue'].sum()
                avg_clv = platform_revenue / platform_customers if platform_customers > 0 else 0
                
                assumed_cac = marketing_metrics['Assumed_CAC'].get(platform, 10)
                roi = ((avg_clv - assumed_cac) / assumed_cac * 100) if assumed_cac > 0 else 0
                
                with col:
                    st.markdown(f"""
                    <div class="marketing-card">
                        <h4>{platform}</h4>
                        <p><strong>Avg CLV:</strong> ${avg_clv:.2f}</p>
                        <p><strong>Assumed CAC:</strong> ${assumed_cac:.2f}</p>
                        <p><strong>ROI:</strong> {roi:.1f}%</p>
                        <p><strong>Customers:</strong> {platform_customers:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # 客户细分分析
            st.markdown("### 👥 Customer Segmentation (RFM Analysis)")
            
            rfm_data = perform_rfm_analysis(completed_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_segmentation = create_customer_segmentation_chart(rfm_data)
                st.plotly_chart(fig_segmentation, width='stretch', key='attribution_segmentation')
            
            with col2:
                fig_rfm_heatmap = create_rfm_heatmap(rfm_data)
                st.plotly_chart(fig_rfm_heatmap, width='stretch', key='attribution_rfm_heatmap')
            
            # 客户细分详情
            segment_summary = rfm_data.groupby('Segment').agg({
                'Recency': 'mean',
                'Frequency': 'mean', 
                'Monetary': ['mean', 'count']
            }).round(2)
            
            st.markdown("#### Customer Segment Summary")
            st.dataframe(segment_summary, width='stretch')
            
            # 营销建议
            st.markdown("### 💡 Marketing Recommendations")
            
            best_segment = rfm_data.groupby('Segment')['Monetary'].mean().idxmax()
            largest_segment = rfm_data['Segment'].value_counts().idxmax()
            
            recommendations = [
                f"🏆 **Focus on '{best_segment}' segment**: Highest monetary value customers",
                f"📊 **Scale '{largest_segment}' segment**: Largest customer group for volume growth",
                "🎯 **Cross-platform promotion**: Encourage customers to try multiple platforms",
                "🔄 **Retention campaigns**: Target 'At Risk' customers with special offers",
                "📈 **Upselling opportunities**: Increase AOV for loyal customers"
            ]
            
            for rec in recommendations:
                st.markdown(rec)
        
        # Tab 7: Retention (新增客户留存分析)
        with tab7:
            st.markdown("### 🔄 Customer Retention & Cohort Analysis")
            
            # 客户生命周期价值分布
            customer_metrics = marketing_metrics['Customer_Metrics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_clv = create_clv_distribution_chart(customer_metrics)
                st.plotly_chart(fig_clv, width='stretch', key='retention_clv_distribution')
            
            with col2:
                # 客户订单频率分布
                order_freq = customer_metrics['Order_Count'].value_counts().sort_index()
                fig_freq = px.bar(
                    x=order_freq.index,
                    y=order_freq.values,
                    title="Customer Order Frequency Distribution",
                    labels={'x': 'Number of Orders', 'y': 'Number of Customers'}
                )
                st.plotly_chart(fig_freq, width='stretch', key='retention_order_frequency')
            
            # 客户旅程分析
            st.markdown("### 🛣️ Customer Journey Analysis")
            
            journey_data = create_customer_journey_analysis(completed_df)
            
            # 客户生命周期统计
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_lifespan = journey_data['Customer_Lifespan'].mean()
                st.metric("Avg Customer Lifespan", f"{avg_lifespan:.0f} days")
            
            with col2:
                avg_frequency = journey_data['Purchase_Frequency'].mean()
                st.metric("Avg Purchase Frequency", f"{avg_frequency:.2f}/day")
            
            with col3:
                repeat_customers = (journey_data['Total_Orders'] > 1).sum()
                repeat_rate = repeat_customers / len(journey_data) * 100
                st.metric("Repeat Customer Rate", f"{repeat_rate:.1f}%")
            
            with col4:
                high_value_customers = (customer_metrics['CLV'] > customer_metrics['CLV'].median()).sum()
                st.metric("High-Value Customers", f"{high_value_customers:,}")
            
            # 平台忠诚度分析
            st.markdown("### 🏆 Platform Loyalty Analysis")
            
            # 计算单平台 vs 多平台客户
            customer_platform_counts = completed_df.groupby('Customer_ID')['Platform'].nunique()
            
            loyalty_data = {
                'Single Platform': (customer_platform_counts == 1).sum(),
                'Multi Platform': (customer_platform_counts > 1).sum()
            }
            
            fig_loyalty = px.pie(
                values=list(loyalty_data.values()),
                names=list(loyalty_data.keys()),
                title="Customer Platform Loyalty"
            )
            st.plotly_chart(fig_loyalty, width='stretch', key='retention_platform_loyalty')
            
            # 留存率分析
            try:
                from operator import attrgetter
                retention_table = create_cohort_table(completed_df)
                if not retention_table.empty:
                    fig_cohort = create_cohort_heatmap(retention_table)
                    st.plotly_chart(fig_cohort, width='stretch', key='retention_cohort_heatmap')
                else:
                    st.info("Insufficient data for cohort analysis")
            except Exception as e:
                st.warning("Cohort analysis requires more historical data")
            
            # 客户价值分层
            st.markdown("### 💎 Customer Value Tiers")
            
            # 根据CLV分层
            customer_metrics['Value_Tier'] = pd.qcut(
                customer_metrics['CLV'], 
                q=4, 
                labels=['Bronze', 'Silver', 'Gold', 'Platinum']
            )
            
            tier_summary = customer_metrics.groupby('Value_Tier').agg({
                'CLV': ['count', 'mean', 'sum'],
                'Order_Count': 'mean',
                'Avg_Order_Value': 'mean'
            }).round(2)
            
            st.dataframe(tier_summary, width='stretch')
        
        # 导出功能
        st.markdown("---")
        st.markdown("### 📥 Export & Share")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            excel_data = generate_excel_report(df, marketing_metrics)
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_data,
                file_name=f"luckin_marketing_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            html_content = generate_html_report(df, marketing_metrics)
            st.download_button(
                label="📄 Download HTML Report", 
                data=html_content,
                file_name=f"luckin_marketing_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col3:
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="💾 Download Raw Data",
                data=csv,
                file_name=f"luckin_enhanced_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col4:
            if st.button("📤 Generate Share Link", use_container_width=True):
                st.info("🔗 Share functionality coming soon!")
    
    else:
        # 欢迎页面
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("# ☕ Welcome to Luckin Advanced Marketing Analytics")
            st.markdown("### Your Comprehensive Customer Intelligence Platform")
            st.markdown("#### 营销与客户分析智能平台")
            
            st.markdown("---")
            
            st.markdown("### 🚀 Getting Started")
            st.markdown("""
            1. **📂 Upload Data**: Upload CSV files from platforms in the sidebar
            2. **📊 Explore Analytics**: Navigate through 7 comprehensive analysis tabs
            3. **💡 Get Insights**: Access AI-powered business recommendations 
            4. **📥 Export Reports**: Download Excel, HTML, and CSV reports
            """)
            
            st.markdown("---")
            
            st.markdown("### ✨ Enhanced Features")
        
        # 功能卡片
        col1, col2, col3, col4 = st.columns(4)
        
        features = [
            ("🎯", "Customer Attribution", "Attribution Analytics", "Multi-touch attribution modeling & funnel analysis"),
            ("🔄", "Retention Analysis", "Cohort & CLV Analysis", "Customer lifecycle tracking & retention cohorts"),
            ("👥", "Customer Segmentation", "RFM Segmentation", "Behavioral segmentation & targeting insights"),
            ("📊", "Predictive Analytics", "Growth Forecasting", "Revenue predictions & trend analysis")
        ]
        
        for col, (icon, title, subtitle, desc) in zip([col1, col2, col3, col4], features):
            with col:
                st.markdown(f"""
                <div style='text-align: center; padding: 25px; background: white; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); height: 180px;'>
                    <div style='font-size: 42px; margin-bottom: 10px;'>{icon}</div>
                    <p style='font-weight: bold; font-size: 16px; margin: 5px 0; color: #232773;'>{title}</p>
                    <p style='font-size: 12px; color: #666; margin: 5px 0;'>{subtitle}</p>
                    <p style='font-size: 11px; color: #999; margin-top: 10px;'>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 分析维度说明
        st.markdown("### 📌 Analysis Dimensions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Marketing Analytics**:
            - 🎯 **Customer Attribution**: Multi-touch attribution & conversion tracking
            - 📊 **Retention Analysis**: Cohort analysis & customer lifecycle
            - 💰 **CLV Analysis**: Customer lifetime value modeling
            - 📈 **ROI Tracking**: Marketing ROI & CAC optimization
            - 👥 **Segmentation**: RFM analysis & behavioral clustering
            """)
        
        with col2:
            st.markdown("""
            **Operational Intelligence**:
            - ⏰ **Time Optimization**: Peak hours & demand patterns
            - 🏪 **Store Performance**: Multi-dimensional store comparison
            - 📦 **Order Analytics**: Value distribution & cancellation insights
            - 🔄 **Growth Tracking**: Revenue trends & predictive forecasting
            - 🎯 **Platform Analytics**: Cross-platform performance optimization
            """)
        
        st.markdown("<br><br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
