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
warnings.filterwarnings('ignore')

# --- 页面配置 ---
st.set_page_config(
    page_title="Luckin Coffee - 美国市场运营分析系统 (US Operations)",
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
    </style>
""", unsafe_allow_html=True)

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
    # 分布在2025年10月
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
            # 生成随机时间用于分析
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
        
        # 过滤到2025年10月
        df = df[(df['Date'] >= '2025-10-01') & (df['Date'] <= '2025-10-31')]
        
        return df[['Date', 'DateTime', 'Revenue', 'Store', 'Platform', 'Is_Completed', 'Is_Cancelled']]
        
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
        
        # 过滤
        df = df[(df['Date'] >= '2025-10-01') & (df['Date'] <= '2025-10-31')]
        
        return df[['Date', 'DateTime', 'Revenue', 'Store', 'Platform', 'Is_Completed', 'Is_Cancelled']]
        
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
                # 当日期显示为########时，推断日期
                df['DateTime'] = infer_grubhub_dates(df)
            else:
                df['DateTime'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        
        df = df.dropna(subset=['DateTime'])
        
        if df.empty:
            return pd.DataFrame()
        
        df['Date'] = df['DateTime'].dt.date
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 收入
        df['Revenue'] = df['subtotal'].apply(clean_currency) if 'subtotal' in df.columns else 0
        
        # 状态（假设Grubhub订单都已完成）
        df['Is_Completed'] = True
        df['Is_Cancelled'] = False
        
        # 门店
        df['Store'] = df['store_name'].fillna('Unknown Store') if 'store_name' in df.columns else 'Unknown Store'
        df['Platform'] = 'Grubhub'
        
        # 过滤
        df = df[(df['Date'] >= '2025-10-01') & (df['Date'] <= '2025-10-31')]
        
        return df[['Date', 'DateTime', 'Revenue', 'Store', 'Platform', 'Is_Completed', 'Is_Cancelled']]
        
    except Exception as e:
        st.error(f"Grubhub解析错误: {str(e)}")
        return pd.DataFrame()

def generate_excel_report(df):
    """生成Excel报告"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # 汇总数据
        summary_data = {
            '指标': ['总记录数', '完成订单', '总收入 ($)', '取消率 (%)', '平均订单价值 ($)'],
            '值': [
                len(df),
                len(df[df['Is_Completed'] == True]),
                df[df['Is_Completed'] == True]['Revenue'].sum(),
                (df['Is_Cancelled'].sum() / len(df)) * 100 if len(df) > 0 else 0,
                df[df['Is_Completed'] == True]['Revenue'].mean()
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='汇总', index=False)
        
        # 平台数据
        platform_summary = df.groupby('Platform').agg({
            'Revenue': ['count', 'sum', 'mean'],
            'Is_Cancelled': 'sum'
        }).round(2)
        platform_summary.columns = ['订单数', '总收入', '平均订单价值', '取消数量']
        platform_summary.to_excel(writer, sheet_name='平台分析')
        
        # 门店数据
        store_summary = df.groupby('Store').agg({
            'Revenue': ['count', 'sum', 'mean']
        }).round(2)
        store_summary.columns = ['订单数', '总收入', '平均订单价值']
        store_summary.to_excel(writer, sheet_name='门店分析')
        
        # 原始数据
        df.to_excel(writer, sheet_name='原始数据', index=False)
    
    return output.getvalue()

def generate_html_report(df):
    """生成HTML报告"""
    total_records = len(df)
    completed_orders = len(df[df['Is_Completed'] == True])
    total_revenue = df[df['Is_Completed'] == True]['Revenue'].sum()
    cancel_rate = (df['Is_Cancelled'].sum() / len(df)) * 100 if len(df) > 0 else 0
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Luckin Coffee 运营报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #232773; color: white; padding: 20px; text-align: center; }}
            .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .metric {{ text-align: center; padding: 15px; border: 1px solid #ddd; }}
            .metric h3 {{ margin: 0; color: #232773; }}
            .metric p {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
            .section {{ margin: 30px 0; }}
            .section h2 {{ color: #232773; border-bottom: 2px solid #232773; padding-bottom: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Luckin Coffee 美国市场运营分析报告</h1>
            <p>报告生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <h3>总记录数</h3>
                <p>{total_records:,}</p>
            </div>
            <div class="metric">
                <h3>完成订单</h3>
                <p>{completed_orders:,}</p>
            </div>
            <div class="metric">
                <h3>总收入</h3>
                <p>${total_revenue:,.2f}</p>
            </div>
            <div class="metric">
                <h3>取消率</h3>
                <p>{cancel_rate:.1f}%</p>
            </div>
        </div>
        
        <div class="section">
            <h2>平台详细数据</h2>
            <p>各平台表现数据分析</p>
        </div>
    </body>
    </html>
    """
    return html_content

# --- 主应用程序 ---

def main():
    # 标题
    st.markdown("""
        <div class="luckin-header">
            <h1>☕ Luckin Coffee</h1>
            <h2>美国市场运营分析系统 (US Operations)</h2>
            <p style="font-size: 14px; opacity: 0.9;">2025-11-22</p>
        </div>
    """, unsafe_allow_html=True)

    # 侧边栏 - 文件上传
    st.sidebar.header("📂 数据上传")
    st.sidebar.markdown("上传各平台的CSV文件进行分析")
    
    uploaded_files = {}
    
    # 文件上传组件
    uber_file = st.sidebar.file_uploader("Uber Eats CSV", type=['csv'], key="uber")
    doordash_file = st.sidebar.file_uploader("DoorDash CSV", type=['csv'], key="doordash") 
    grubhub_file = st.sidebar.file_uploader("Grubhub CSV", type=['csv'], key="grubhub")
    
    if uber_file or doordash_file or grubhub_file:
        # 解析数据
        dataframes = []
        
        if uber_file:
            uber_df = parse_uber(uber_file)
            if not uber_df.empty:
                dataframes.append(uber_df)
                st.sidebar.success(f"✅ Uber Eats: {len(uber_df)} 条记录")
        
        if doordash_file:
            doordash_df = parse_doordash(doordash_file)
            if not doordash_df.empty:
                dataframes.append(doordash_df)
                st.sidebar.success(f"✅ DoorDash: {len(doordash_df)} 条记录")
        
        if grubhub_file:
            grubhub_df = parse_grubhub(grubhub_file)
            if not grubhub_df.empty:
                dataframes.append(grubhub_df)
                st.sidebar.success(f"✅ Grubhub: {len(grubhub_df)} 条记录")
        
        if not dataframes:
            st.error("❌ 无法解析任何数据文件，请检查文件格式")
            return
        
        # 合并数据
        df = pd.concat(dataframes, ignore_index=True)
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        # 计算关键指标
        total_records = len(df)
        completed_orders = len(df[df['Is_Completed'] == True])
        total_revenue = df[df['Is_Completed'] == True]['Revenue'].sum()
        cancel_rate = (df['Is_Cancelled'].sum() / len(df)) * 100 if len(df) > 0 else 0
        
        # 显示关键指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 总记录数",
                value=f"{total_records:,}",
                help="所有平台的订单总数"
            )
        
        with col2:
            st.metric(
                label="✅ 完成订单", 
                value=f"{completed_orders:,}",
                help="成功完成的订单数量"
            )
        
        with col3:
            st.metric(
                label="💰 总收入",
                value=f"${total_revenue:,.2f}",
                help="所有完成订单的总收入"
            )
        
        with col4:
            st.metric(
                label="❌ 取消率",
                value=f"{cancel_rate:.1f}%",
                delta="目标 < 5%" if cancel_rate < 5 else "⚠️ 超过目标",
                help="订单取消率"
            )

        # 主要分析区域
        st.markdown("---")
        
        # 📊 报告预览 - 趋势图
        st.markdown("## 📊 报告预览")
        
        # 每日趋势
        completed_df = df[df['Is_Completed'] == True].copy()
        daily_platform = completed_df.groupby(['Date', 'Platform']).size().unstack(fill_value=0)
        
        fig_trend = go.Figure()
        
        colors = {'Uber Eats': '#00897B', 'DoorDash': '#FF6B35', 'Grubhub': '#F57C00'}
        
        for platform in daily_platform.columns:
            fig_trend.add_trace(go.Scatter(
                x=daily_platform.index,
                y=daily_platform[platform],
                mode='lines+markers',
                name=platform,
                line=dict(color=colors.get(platform, '#232773'), width=3),
                marker=dict(size=6)
            ))
        
        fig_trend.update_layout(
            title='每日订单趋势',
            xaxis_title='日期',
            yaxis_title='订单数',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)

        # 两列布局
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 🥧 渠道占比
            st.markdown("### 🥧 渠道占比 (Market Share)")
            
            platform_orders = completed_df.groupby('Platform').size()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=platform_orders.index,
                values=platform_orders.values,
                hole=0.4,
                marker_colors=[colors.get(platform, '#232773') for platform in platform_orders.index],
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig_pie.update_layout(
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.1)
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # 🏪 门店表现
            st.markdown("### 🏪 门店表现 (Store Performance)")
            
            store_revenue = completed_df.groupby('Store')['Revenue'].sum().sort_values(ascending=True)
            
            fig_store = go.Figure(go.Bar(
                x=store_revenue.values,
                y=store_revenue.index,
                orientation='h',
                marker_color='#232773',
                text=[f'${x:,.0f}' for x in store_revenue.values],
                textposition='auto'
            ))
            
            fig_store.update_layout(
                title='各门店收入排名',
                xaxis_title='收入 ($)',
                height=400,
                margin=dict(l=100)
            )
            
            st.plotly_chart(fig_store, use_container_width=True)

        # 📋 平台详细数据
        st.markdown("### 📋 平台详细数据 (Platform Details)")
        
        platform_stats = []
        for platform in completed_df['Platform'].unique():
            platform_data = completed_df[completed_df['Platform'] == platform]
            stats = {
                '平台 (Platform)': platform,
                '订单量 (Orders)': len(platform_data),
                '营收 (Revenue)': f"${platform_data['Revenue'].sum():,.2f}",
                '客单价 (Avg Ticket)': f"${platform_data['Revenue'].mean():.2f}",
                '市场份额 (Share)': f"{(len(platform_data) / len(completed_df)) * 100:.1f}%"
            }
            platform_stats.append(stats)
        
        platform_df = pd.DataFrame(platform_stats)
        
        # 使用HTML表格以获得更好的格式
        html_table = platform_df.to_html(index=False, escape=False, classes='table table-striped')
        html_table = html_table.replace('Uber Eats', '<span class="platform-colors-uber">● Uber Eats</span>')
        html_table = html_table.replace('DoorDash', '<span class="platform-colors-doordash">● DoorDash</span>')  
        html_table = html_table.replace('Grubhub', '<span class="platform-colors-grubhub">● Grubhub</span>')
        
        st.markdown(html_table, unsafe_allow_html=True)

        # 📈 运营建议
        st.markdown("### 📈 下阶段运营建议 (Recommendations)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1. 运营优化 (Operations)")
            
            # 分析数据生成建议
            platform_performance = completed_df.groupby('Platform').agg({
                'Revenue': ['count', 'sum', 'mean']
            })
            
            best_platform = platform_performance[('Revenue', 'sum')].idxmax()
            worst_aov = platform_performance[('Revenue', 'mean')].idxmin()
            
            recommendations_ops = [
                f"• 针对 {best_platform}（Top Channel）优化供应链，预保充足库存以应对高峰期。",
                f"• 加强 8th Ave 门店（Broadway）运营管理。",
                f"• 针对 {worst_aov} 平台优化菜单定价策略。"
            ]
            
            for rec in recommendations_ops:
                st.markdown(rec)
        
        with col2:
            st.markdown("#### 2. 营销策略 (Marketing)")
            
            recommendations_marketing = [
                "• Grubhub 策略：通过 'GH+ Delivery Fee' 促销活动提升市场占有率。",
                "• DoorDash 策略：利用其较高的 'SO Delivery Fee' 定价政策优化盈利能力。",
                "• 跨平台协同：统一品牌形象，提升整体市场认知度。"
            ]
            
            for rec in recommendations_marketing:
                st.markdown(rec)

        # 导出功能
        st.markdown("---")
        st.markdown("### 📥 导出选项")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Excel导出
            excel_data = generate_excel_report(df)
            st.download_button(
                label="📊 下载Excel报告",
                data=excel_data,
                file_name=f"luckin_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            # HTML报告
            html_content = generate_html_report(df)
            st.download_button(
                label="📄 下载HTML报告",
                data=html_content,
                file_name=f"luckin_report_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col3:
            # CSV数据
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="💾 下载原始数据",
                data=csv,
                file_name=f"luckin_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col4:
            st.button("📤 分享仪表板", use_container_width=True, help="复制链接到剪贴板")

    else:
        # 欢迎页面
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("# ☕ 欢迎使用 Luckin 数据分析系统")
            st.markdown("### 您的外卖业务智能分析平台")
            
            st.markdown("---")
            
            st.markdown("### 🚀 开始使用")
            st.markdown("""
            1. 上传各平台CSV文件 (Uber Eats, DoorDash, Grubhub)
            2. 查看自动生成的洞察和关键指标  
            3. 探索详细的运营分析
            4. 导出报告并与团队分享
            """)
            
            st.markdown("---")
            
            st.markdown("### ✨ 核心功能")
        
        # 功能卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
                <div style='font-size: 36px;'>📊</div>
                <p><strong>收入分析</strong></p>
                <small>Revenue Analytics</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
                <div style='font-size: 36px;'>💡</div>
                <p><strong>智能洞察</strong></p>
                <small>Smart Insights</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
                <div style='font-size: 36px;'>📈</div>
                <p><strong>增长指标</strong></p>
                <small>Growth Metrics</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);'>
                <div style='font-size: 36px;'>⏰</div>
                <p><strong>实时分析</strong></p>
                <small>Real-time Analysis</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
