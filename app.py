# -*- coding: utf-8 -*-
"""
骨质疏松机会性筛查系统
基于逻辑回归模型的腰椎CT值预测
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings('ignore')

# ====================== 页面配置 ======================
st.set_page_config(
    page_title="骨质疏松机会性筛查系统",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== 标题 ======================
st.title("🦴 骨质疏松机会性筛查系统")
st.markdown("""
基于**逻辑回归机器学习模型**的骨质疏松风险预测系统。
使用LASSO筛选的7个核心特征进行预测。
""")

# ====================== 7个核心特征（逻辑回归权重） ======================
SELECTED_FEATURES = [
    'age',              # 年龄
    'BMI',              # 身体质量指数
    'sex',              # 性别 (0=女, 1=男)
    'L1guanzhuang',     # 第1腰椎冠状面CT值
    'L2guanzhuang',     # 第2腰椎冠状面CT值
    'L3hengduan',       # 第3腰椎横断面CT值
    'L4shizhuang'       # 第4腰椎矢状面CT值
]

# 特征中文名称
FEATURE_NAMES_CN = {
    'age': '年龄 (岁)',
    'BMI': 'BMI',
    'sex': '性别',
    'L1guanzhuang': 'L1冠状面 (HU)',
    'L2guanzhuang': 'L2冠状面 (HU)',
    'L3hengduan': 'L3横断面 (HU)',
    'L4shizhuang': 'L4矢状面 (HU)'
}

# 特征描述
FEATURE_DESCRIPTIONS = {
    'age': '年龄 - 骨质疏松主要危险因素，年龄越大风险越高',
    'BMI': '身体质量指数 - 反映整体营养状态，低BMI增加风险',
    'sex': '性别 - 女性骨质疏松患病率显著高于男性',
    'L1guanzhuang': '第1腰椎冠状面CT值 - 反映上腰椎骨密度',
    'L2guanzhuang': '第2腰椎冠状面CT值 - 反映上腰椎骨密度，重要预测因子',
    'L3hengduan': '第3腰椎横断面CT值 - 反映腰椎中部骨密度，关键预测因子',
    'L4shizhuang': '第4腰椎矢状面CT值 - 承重最大椎体，最重要预测因子'
}

# 参考范围（用于显示）
REFERENCE_RANGES = {
    'age': (40, 90),
    'BMI': (18.5, 28.0),
    'sex': (0, 1),
    'L1guanzhuang': (50, 180),
    'L2guanzhuang': (50, 180),
    'L3hengduan': (50, 180),
    'L4shizhuang': (50, 180)
}

# 输入范围
INPUT_RANGES = {
    'age': (18, 100),
    'BMI': (15.0, 40.0),
    'sex': (0, 1),
    'L1guanzhuang': (30.0, 250.0),
    'L2guanzhuang': (30.0, 250.0),
    'L3hengduan': (30.0, 250.0),
    'L4shizhuang': (30.0, 250.0)
}

# 默认值（基于临床常见值及您提供的均值）
DEFAULT_VALUES = {
    'age': 65,
    'BMI': 23.5,
    'sex': 0,               # 0=女性
    'L1guanzhuang': 104.07, # 均值
    'L2guanzhuang': 102.71,
    'L3hengduan': 93.66,
    'L4shizhuang': 96.55
}

# 逻辑回归系数（您提供的权重，顺序与特征一致）
LOGISTIC_COEFFICIENTS = {
    'age': -0.429782,
    'BMI': -0.688228,
    'sex': -0.053709,
    'L1guanzhuang': -0.169589,
    'L2guanzhuang': -0.424072,
    'L3hengduan': -0.512535,
    'L4shizhuang': -0.905305
}

# ====================== 加载模型 ======================
@st.cache_resource
def load_models():
    """加载逻辑回归模型和标准化器"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    try:
        model = joblib.load(os.path.join(model_dir, 'LR_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        st.sidebar.success("✅ 模型加载成功")
        return model, scaler
    except Exception as e:
        st.sidebar.error(f"❌ 模型加载失败: {e}")
        st.sidebar.info("请确保models文件夹包含: LR_model.pkl, scaler.pkl")
        return None, None

# ====================== 预测函数 ======================
def predict_osteoporosis(model, scaler, input_values):
    """
    使用逻辑回归预测骨质疏松风险
    
    Args:
        model: 训练好的逻辑回归模型 (sklearn.linear_model.LogisticRegression)
        scaler: 标准化器 (StandardScaler)
        input_values: 7个特征的输入值字典
    
    Returns:
        probability: 骨质疏松概率
        prediction: 预测类别 (0/1)
    """
    # 按特征顺序构建输入数组
    feature_order = SELECTED_FEATURES
    X_input = np.array([[input_values[f] for f in feature_order]])
    
    # 标准化
    X_scaled = scaler.transform(X_input)
    
    # 预测概率
    probability = model.predict_proba(X_scaled)[0, 1]
    prediction = 1 if probability > 0.5 else 0
    
    return probability, prediction

# ====================== 计算特征贡献（基于系数加权偏离） ======================
def calculate_feature_contributions(input_values):
    """基于逻辑回归系数和参考范围计算各特征对风险的贡献"""
    contributions = []
    # 使用参考范围中值作为基准
    for feat in SELECTED_FEATURES:
        value = input_values[feat]
        ref_low, ref_high = REFERENCE_RANGES.get(feat, (0, 100))
        ref_mean = (ref_low + ref_high) / 2
        coeff = LOGISTIC_COEFFICIENTS.get(feat, 0)
        
        # 计算偏离度（标准化为与系数同量级）
        if ref_high > ref_low:
            deviation = (value - ref_mean) / (ref_high - ref_low) * 2  # 近似标准化
        else:
            deviation = 0
        
        # 贡献 = 系数 * 偏离度（系数为负，值越小贡献越大）
        contribution = coeff * deviation * 0.5  # 缩放系数用于可视化
        contributions.append(contribution)
    
    return contributions

# ====================== 主函数 ======================
def main():
    # 加载模型
    model, scaler = load_models()
    
    if model is None:
        st.warning("⚠️ 请先上传模型文件到models文件夹")
        return
    
    # ====================== 侧边栏 ======================
    st.sidebar.header("📋 导航")
    page = st.sidebar.radio(
        "选择页面",
        ["🔍 骨质疏松预测", "📊 特征分析", "ℹ️ 使用说明"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **模型信息**
    - 算法: 逻辑回归 (Logistic Regression)
    - 特征数: 7个 (LASSO筛选)
    - 权重: 基于训练数据优化
    """)
    
    # ====================== 预测页面 ======================
    if page == "🔍 骨质疏松预测":
        st.header("🔍 骨质疏松风险预测")
        st.markdown("请输入患者的7个核心特征进行预测。")
        
        col1, col2 = st.columns(2)
        
        input_values = {}
        
        with col1:
            st.subheader("📊 临床特征")
            
            input_values['age'] = st.number_input(
                "**年龄 (age)**",
                min_value=18, max_value=100, value=65, step=1,
                help="年龄 (岁) | 正常范围: 40-90"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['age'][0]}-{REFERENCE_RANGES['age'][1]} 岁")
            
            input_values['sex'] = st.selectbox(
                "**性别 (sex)**",
                options=[0, 1],
                format_func=lambda x: "女" if x == 0 else "男",
                index=0,
                help="0=女性, 1=男性"
            )
            st.caption("参考: 女性(0) 或 男性(1)")
            
            input_values['BMI'] = st.number_input(
                "**BMI**",
                min_value=15.0, max_value=40.0, value=23.5, step=0.1,
                help="身体质量指数 | 正常范围: 18.5-28.0"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['BMI'][0]}-{REFERENCE_RANGES['BMI'][1]}")
        
        with col2:
            st.subheader("📊 影像学特征")
            
            input_values['L1guanzhuang'] = st.number_input(
                "**L1冠状面 (L1guanzhuang)**",
                min_value=30.0, max_value=250.0, value=104.07, step=1.0,
                help="第1腰椎冠状面CT值 | 单位: HU"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L1guanzhuang'][0]}-{REFERENCE_RANGES['L1guanzhuang'][1]} HU")
            
            input_values['L2guanzhuang'] = st.number_input(
                "**L2冠状面 (L2guanzhuang)**",
                min_value=30.0, max_value=250.0, value=102.71, step=1.0,
                help="第2腰椎冠状面CT值 | 单位: HU"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L2guanzhuang'][0]}-{REFERENCE_RANGES['L2guanzhuang'][1]} HU")
            
            input_values['L3hengduan'] = st.number_input(
                "**L3横断面 (L3hengduan)**",
                min_value=30.0, max_value=250.0, value=93.66, step=1.0,
                help="第3腰椎横断面CT值 | 单位: HU"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L3hengduan'][0]}-{REFERENCE_RANGES['L3hengduan'][1]} HU")
            
            input_values['L4shizhuang'] = st.number_input(
                "**L4矢状面 (L4shizhuang)**",
                min_value=30.0, max_value=250.0, value=96.55, step=1.0,
                help="第4腰椎矢状面CT值 | 单位: HU"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L4shizhuang'][0]}-{REFERENCE_RANGES['L4shizhuang'][1]} HU")
        
        # 预测按钮
        if st.button("🚀 开始预测", type="primary", use_container_width=True):
            with st.spinner("正在分析中..."):
                try:
                    # 执行预测
                    probability, prediction = predict_osteoporosis(model, scaler, input_values)
                    
                    # 显示结果
                    st.markdown("---")
                    st.subheader("📊 预测结果")
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    
                    with col_res1:
                        if prediction == 1:
                            st.error(f"## ⚠️ 诊断结果: **骨质疏松**")
                        else:
                            st.success(f"## ✅ 诊断结果: **非骨质疏松**")
                    
                    with col_res2:
                        st.metric("骨质疏松概率", f"{probability:.2%}")
                    
                    with col_res3:
                        if probability < 0.3:
                            st.success("### 风险等级: 🟢 低风险")
                        elif probability < 0.7:
                            st.warning("### 风险等级: 🟡 中风险")
                        else:
                            st.error("### 风险等级: 🔴 高风险")
                    
                    # 风险仪表盘
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "骨质疏松风险 (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "lightyellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # 特征贡献分析
                    st.subheader("🧠 模型决策解释")
                    
                    contributions = calculate_feature_contributions(input_values)
                    
                    contrib_df = pd.DataFrame({
                        '特征': SELECTED_FEATURES,
                        '特征中文': [FEATURE_NAMES_CN.get(f, f) for f in SELECTED_FEATURES],
                        '输入值': [input_values[f] for f in SELECTED_FEATURES],
                        '贡献值': contributions,
                        '影响方向': ['增加风险' if v > 0 else '降低风险' for v in contributions]
                    })
                    contrib_df['绝对值'] = np.abs(contrib_df['贡献值'])
                    contrib_df = contrib_df.sort_values('绝对值', ascending=False)
                    
                    st.dataframe(
                        contrib_df[['特征中文', '输入值', '贡献值', '影响方向']].style.format({
                            '输入值': '{:.1f}',
                            '贡献值': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                    
                    # 贡献条形图
                    fig_contrib = px.bar(contrib_df,
                                         x='贡献值',
                                         y='特征中文',
                                         orientation='h',
                                         color='影响方向',
                                         color_discrete_map={'增加风险': '#EF553B', '降低风险': '#636EFA'},
                                         title='各特征对预测的影响')
                    fig_contrib.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
                    fig_contrib.update_layout(height=400)
                    st.plotly_chart(fig_contrib, use_container_width=True)
                    
                    # 临床建议
                    st.subheader("📋 临床建议")
                    if probability > 0.7:
                        st.warning("""
                        **⚠️ 高风险 (骨质疏松概率 > 70%)**:
                        1. **建议就诊**: 尽快咨询内分泌科或骨科专家
                        2. **DXA检查**: 建议进行双能X线骨密度检查确诊
                        3. **药物治疗**: 根据医生建议考虑抗骨质疏松药物
                        4. **生活方式**: 增加钙和维生素D摄入，适度负重运动
                        5. **预防跌倒**: 评估跌倒风险，采取预防措施
                        """)
                    elif probability > 0.3:
                        st.info("""
                        **⚠️ 中风险 (骨质疏松概率 30%-70%)**:
                        1. **骨密度监测**: 建议1年内复查DXA
                        2. **生活方式调整**: 增加钙摄入(1000-1200mg/天)
                        3. **补充维生素D**: 维持血清25(OH)D > 30 ng/mL
                        4. **负重运动**: 每周3-5次，每次30分钟
                        5. **戒烟限酒**: 减少骨质流失风险因素
                        """)
                    else:
                        st.success("""
                        **✅ 低风险 (骨质疏松概率 < 30%)**:
                        1. **常规随访**: 每2-3年复查骨密度
                        2. **维持健康生活方式**: 均衡饮食，适度运动
                        3. **充足钙摄入**: 每日800-1000mg钙剂
                        4. **预防为主**: 保持良好生活习惯
                        """)
                
                except Exception as e:
                    st.error(f"❌ 预测失败: {e}")
                    st.info("请检查输入值是否在合理范围内，并确保模型文件正确。")
    
    # ====================== 特征分析页面 ======================
    elif page == "📊 特征分析":
        st.header("📊 特征分析")
        
        tab1, tab2, tab3 = st.tabs(["📈 特征重要性", "🔬 逻辑回归系数", "ℹ️ 特征说明"])
        
        with tab1:
            st.subheader("7个核心特征重要性分析（基于系数绝对值）")
            
            importance_df = pd.DataFrame({
                '特征': SELECTED_FEATURES,
                '特征中文': [FEATURE_NAMES_CN.get(f, f) for f in SELECTED_FEATURES],
                '系数绝对值': [abs(LOGISTIC_COEFFICIENTS.get(f, 0)) for f in SELECTED_FEATURES]
            }).sort_values('系数绝对值', ascending=True)
            
            fig = px.bar(importance_df,
                         x='系数绝对值',
                         y='特征中文',
                         orientation='h',
                         title="特征重要性排序 (基于逻辑回归系数绝对值)",
                         color='系数绝对值',
                         color_continuous_scale='Reds')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **特征重要性说明**:
            - **L4shizhuang** 是最重要的预测指标 (系数绝对值=0.9053)
            - **BMI** 次之 (系数绝对值=0.6882)
            - **L3hengduan** 和 **L2guanzhuang** 也是关键预测因子
            - **年龄** 具有较大影响 (系数绝对值=0.4298)
            - 所有特征系数均为**负值**，即特征值越低，骨质疏松风险越高
            """)
        
        with tab2:
            st.subheader("逻辑回归系数详情")
            
            lasso_df = pd.DataFrame({
                '特征': SELECTED_FEATURES,
                '特征中文': [FEATURE_NAMES_CN.get(f, f) for f in SELECTED_FEATURES],
                '逻辑回归系数': [LOGISTIC_COEFFICIENTS.get(f, 0) for f in SELECTED_FEATURES]
            }).sort_values('逻辑回归系数', ascending=True)
            
            fig_lasso = px.bar(lasso_df,
                               x='逻辑回归系数',
                               y='特征中文',
                               orientation='h',
                               title="逻辑回归系数",
                               color='逻辑回归系数',
                               color_continuous_scale='Blues_r')
            fig_lasso.update_layout(height=400)
            st.plotly_chart(fig_lasso, use_container_width=True)
            
            st.markdown("""
            **系数解读**:
            - **负系数**: 特征值越低，骨质疏松风险越高
            - **系数绝对值越大**: 特征对预测的影响越大
            - **L4shizhuang** 具有最大的绝对系数 (-0.9053)，是最重要的预测因子
            - **性别** 系数较小 (-0.0537)，但仍有统计学意义
            """)
        
        with tab3:
            st.subheader("7个核心特征详细说明")
            
            feature_table = []
            for feat in SELECTED_FEATURES:
                feature_table.append({
                    '特征': feat,
                    '特征中文': FEATURE_NAMES_CN.get(feat, feat),
                    '描述': FEATURE_DESCRIPTIONS.get(feat, ''),
                    '逻辑回归系数': LOGISTIC_COEFFICIENTS.get(feat, 0),
                    '参考范围': f"{REFERENCE_RANGES[feat][0]}-{REFERENCE_RANGES[feat][1]}",
                    '与骨质疏松关系': '负相关 (值↓ → 风险↑)'
                })
            
            st.dataframe(pd.DataFrame(feature_table), use_container_width=True)
            
            st.markdown("""
            ### 🎯 特征筛选说明
            
            **为什么选择这7个特征？**
            
            通过LASSO回归从临床及腰椎CT特征中筛选出以下核心预测因子：
            
            | 特征 | 类型 | LASSO系数 | 临床意义 |
            |------|------|-----------|---------|
            | **age** | 临床 | -0.4298 | 年龄增长是骨质疏松最强危险因素 |
            | **BMI** | 临床 | -0.6882 | 低体重/低BMI增加骨质疏松风险 |
            | **sex** | 临床 | -0.0537 | 女性患病率显著高于男性 |
            | **L1guanzhuang** | CT | -0.1696 | 上腰椎骨密度指标 |
            | **L2guanzhuang** | CT | -0.4241 | 上腰椎骨密度，重要预测因子 |
            | **L3hengduan** | CT | -0.5125 | 腰椎中部骨密度，关键因子 |
            | **L4shizhuang** | CT | -0.9053 | 承重最大椎体，最重要预测因子 |
            """)
    
    # ====================== 使用说明页面 ======================
    else:
        st.header("ℹ️ 使用说明")
        
        st.markdown("""
        ## 📖 系统使用指南
        
        ### 1. 系统概述
        本系统基于**逻辑回归机器学习模型**，使用LASSO筛选的7个核心特征进行骨质疏松风险预测。
        
        ### 2. 模型性能
        | 指标 | 数值 |
        |------|------|
        | 验证集准确率 | 83.05% |
        | 敏感性 | 90.62% |
        | 特异性 | 74.07% |
        | AUC | 0.89 |
        
        ### 3. 使用方法
        1. 进入"🔍 骨质疏松预测"页面
        2. 输入7个核心特征值：
           - 年龄 (岁)
           - 性别 (女/男)
           - BMI
           - L1冠状面CT值 (HU)
           - L2冠状面CT值 (HU)
           - L3横断面CT值 (HU)
           - L4矢状面CT值 (HU)
        3. 点击"开始预测"按钮
        4. 查看预测结果和临床建议
        
        ### 4. 输入特征参考范围
        | 特征 | 参考范围 | 单位 |
        |------|---------|------|
        | 年龄 | 40-90 | 岁 |
        | BMI | 18.5-28.0 | - |
        | L1冠状面 | 50-180 | HU |
        | L2冠状面 | 50-180 | HU |
        | L3横断面 | 50-180 | HU |
        | L4矢状面 | 50-180 | HU |
        
        ### 5. 结果解读
        
        #### 风险等级
        - 🟢 **低风险 (<30%)**: 各指标在正常范围
        - 🟡 **中风险 (30%-70%)**: 需要进一步评估
        - 🔴 **高风险 (>70%)**: 建议DXA检查确诊
        
        ### 6. 重要声明
        ⚠️ **本系统为机会性筛查工具，不能替代DXA金标准诊断**
        """)
    
    # 页脚
    st.markdown("---")
    st.caption("🦴 骨质疏松机会性筛查系统 | 基于逻辑回归机器学习 | 仅供参考，请遵医嘱")


if __name__ == "__main__":
    main()
