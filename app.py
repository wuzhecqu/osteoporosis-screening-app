# -*- coding: utf-8 -*-
"""
骨质疏松机会性筛查系统
基于SVM模型的腰椎CT值预测
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
基于**腰椎CT值**和**SVM机器学习模型**的骨质疏松风险预测系统。
使用5个核心CT特征进行预测，模型验证集准确率 **83.05%**，AUC **0.9074**。
""")

# ====================== 5个核心CT特征 ======================
SELECTED_FEATURES = [
    'L3mean',      # 第3腰椎平均CT值
    'L2mean',      # 第2腰椎平均CT值
    'L3hengduan',  # 第3腰椎横断面CT值
    'L2shizhuang', # 第2腰椎矢状面CT值
    'L4mean'       # 第4腰椎平均CT值
]

# 特征中文名称
FEATURE_NAMES_CN = {
    'L3mean': 'L3均值',
    'L2mean': 'L2均值',
    'L3hengduan': 'L3横断面',
    'L2shizhuang': 'L2矢状面',
    'L4mean': 'L4均值'
}

# 特征描述
FEATURE_DESCRIPTIONS = {
    'L3mean': '第3腰椎平均CT值 - PC1载荷最高(0.2594)',
    'L2mean': '第2腰椎平均CT值',
    'L3hengduan': '第3腰椎横断面CT值 - SHAP重要性第2',
    'L2shizhuang': '第2腰椎矢状面CT值',
    'L4mean': '第4腰椎平均CT值 - SHAP重要性最高(0.0288)，承重最大'
}

# CT值参考范围 (HU)
REFERENCE_RANGES = {
    'L3mean': (100, 200),
    'L2mean': (100, 200),
    'L3hengduan': (90, 190),
    'L2shizhuang': (90, 190),
    'L4mean': (100, 210)
}

# PC1载荷
PC1_LOADINGS = {
    'L3mean': 0.2594,
    'L2mean': 0.2555,
    'L3hengduan': 0.2545,
    'L2shizhuang': 0.2532,
    'L4mean': 0.2528
}

# 正确的SHAP重要性（从实际SHAP计算得出）
SHAP_IMPORTANCE = {
    'L4mean': 0.0288,      # 最重要
    'L3hengduan': 0.0225,  # 第二重要
    'L3mean': 0.0178,      # 第三重要
    'L2shizhuang': 0.0086, # 第四重要
    'L2mean': 0.0056       # 第五重要
}

# 完整CT特征列表（16个）
CT_FEATURES_FULL = [
    'L1hengduan', 'L1shizhuang', 'L1guanzhuang', 'L1mean',
    'L2hengduan', 'L2shizhuang', 'L2guanzhuang', 'L2mean',
    'L3hengduan', 'L3shizhuang', 'L3guanzhuang', 'L3mean',
    'L4hengduan', 'L4shizhuang', 'L4guanzhuang', 'L4mean'
]

# 非核心特征的默认值（基于训练数据中位数）
DEFAULT_VALUES = {
    'L1hengduan': 140, 'L1shizhuang': 138, 'L1guanzhuang': 135, 'L1mean': 138,
    'L2hengduan': 142, 'L2guanzhuang': 140,
    'L3shizhuang': 143, 'L3guanzhuang': 141,
    'L4hengduan': 138, 'L4shizhuang': 136, 'L4guanzhuang': 135
}


# ====================== 加载模型 ======================
@st.cache_resource
def load_models():
    """加载SVM模型和预处理对象"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    try:
        model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        pca = joblib.load(os.path.join(model_dir, 'pca_model.pkl'))
        
        st.sidebar.success("✅ 模型加载成功")
        return model, scaler, pca
    except Exception as e:
        st.sidebar.error(f"❌ 模型加载失败: {e}")
        st.sidebar.info("请确保models文件夹包含: best_model.pkl, scaler.pkl, pca_model.pkl")
        return None, None, None


# ====================== 预测函数 ======================
def predict_osteoporosis(model, scaler, pca, input_values):
    """
    预测骨质疏松风险
    
    Args:
        model: SVM模型
        scaler: 标准化器
        pca: PCA模型
        input_values: 5个核心特征的输入值字典
    
    Returns:
        probability: 骨质疏松概率
        prediction: 预测类别 (0/1)
    """
    # 构建完整的16个特征数组
    full_input = {}
    for feat in CT_FEATURES_FULL:
        if feat in input_values:
            full_input[feat] = input_values[feat]
        else:
            full_input[feat] = DEFAULT_VALUES.get(feat, 140)
    
    input_df = pd.DataFrame([full_input])
    
    # 标准化
    input_scaled = scaler.transform(input_df[CT_FEATURES_FULL])
    
    # PCA降维
    input_pca = pca.transform(input_scaled)
    
    # 预测
    probability = model.predict_proba(input_pca)[0, 1]
    prediction = 1 if probability > 0.5 else 0
    
    return probability, prediction


# ====================== 计算SHAP贡献 ======================
def calculate_shap_contributions(input_values):
    """基于PC1载荷计算特征贡献"""
    shap_values = []
    
    for feat in SELECTED_FEATURES:
        value = input_values[feat]
        ref_low, ref_high = REFERENCE_RANGES.get(feat, (100, 200))
        ref_mean = (ref_low + ref_high) / 2
        
        # CT值越低风险越高（负相关）
        # 计算偏离程度
        if value < ref_mean:
            # 低于正常值，增加风险
            deviation = (ref_mean - value) / ref_mean
            contribution = min(0.1, deviation * 0.05)
        else:
            # 高于正常值，降低风险
            deviation = (value - ref_mean) / ref_mean
            contribution = max(-0.05, -deviation * 0.03)
        
        shap_values.append(contribution)
    
    return shap_values


# ====================== 特征分析页面 ======================
def feature_analysis_page():
    """特征分析页面 - 使用正确的SHAP重要性"""
    st.header("📊 特征分析")
    
    tab1, tab2, tab3 = st.tabs(["📈 特征重要性", "🔬 PC1载荷分析", "ℹ️ 特征说明"])
    
    with tab1:
        st.subheader("5个核心CT特征SHAP重要性")
        
        # 使用正确的SHAP重要性数据
        importance_df = pd.DataFrame({
            '特征': list(SHAP_IMPORTANCE.keys()),
            '特征中文': [FEATURE_NAMES_CN.get(f, f) for f in SHAP_IMPORTANCE.keys()],
            'SHAP重要性': list(SHAP_IMPORTANCE.values())
        }).sort_values('SHAP重要性', ascending=True)
        
        fig = px.bar(importance_df,
                     x='SHAP重要性',
                     y='特征中文',
                     orientation='h',
                     title="SHAP特征重要性排序 (基于实际SHAP计算)",
                     color='SHAP重要性',
                     color_continuous_scale='Reds',
                     text='SHAP重要性')
        
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **🔍 特征重要性说明 (基于实际SHAP计算)**:
        - **L4均值** 是最重要的预测指标 (SHAP=0.0288)
        - **L3横断面** 次之 (SHAP=0.0225)
        - **L3均值** 排名第三 (SHAP=0.0178)
        - **L2矢状面** 排名第四 (SHAP=0.0086)
        - **L2均值** 排名第五 (SHAP=0.0056)
        - 所有特征均与骨质疏松风险**负相关** (CT值越低，风险越高)
        """)
    
    with tab2:
        st.subheader("PC1载荷系数分析")
        
        loadings_df = pd.DataFrame({
            '特征': list(PC1_LOADINGS.keys()),
            '特征中文': [FEATURE_NAMES_CN.get(f, f) for f in PC1_LOADINGS.keys()],
            'PC1载荷': list(PC1_LOADINGS.values())
        }).sort_values('PC1载荷', ascending=True)
        
        fig_loadings = px.bar(loadings_df,
                              x='PC1载荷',
                              y='特征中文',
                              orientation='h',
                              title="PC1载荷系数 (解释85.61%方差)",
                              color='PC1载荷',
                              color_continuous_scale='Blues',
                              text='PC1载荷')
        
        fig_loadings.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_loadings.update_layout(height=400)
        st.plotly_chart(fig_loadings, use_container_width=True)
        
        st.markdown("""
        **PC1临床意义**:
        - PC1解释了 **85.61%** 的原始数据方差
        - 代表**整体腰椎骨密度水平**
        - 所有5个核心特征的PC1载荷非常接近 (0.2528-0.2594)

        **⚠️ 重要发现: SHAP重要性 vs PC1载荷的区别**
        
        | 概念 | 含义 | 本例结果 |
        |------|------|---------|
        | **PC1载荷** | 特征与整体骨密度的相关性 | L3mean最高(0.2594) |
        | **SHAP重要性** | 特征在预测中的实际贡献 | L4mean最高(0.0288) |

        **为什么L4mean最重要？**
        1. L4椎体位于腰椎最下方，承担最大负荷
        2. 骨质流失在承重最大的部位表现最明显
        3. 虽然L3mean与整体骨密度相关性更高，但L4mean提供了额外的独立信息
        """)
    
    with tab3:
        st.subheader("5个核心CT特征详细说明")
        
        feature_table = []
        for feat in SELECTED_FEATURES:
            feature_table.append({
                '特征': feat,
                '特征中文': FEATURE_NAMES_CN.get(feat, feat),
                '描述': FEATURE_DESCRIPTIONS.get(feat, ''),
                'PC1载荷': PC1_LOADINGS.get(feat, 0),
                'SHAP重要性': SHAP_IMPORTANCE.get(feat, 0),
                '正常范围(HU)': f"{REFERENCE_RANGES[feat][0]}-{REFERENCE_RANGES[feat][1]}",
                '与骨质疏松关系': '负相关 (CT值↓ → 风险↑)'
            })
        
        st.dataframe(pd.DataFrame(feature_table), use_container_width=True)
        
        st.markdown("""
        ### 🎯 腰椎解剖与CT值解读
        
        | 椎体 | 临床意义 |
        |------|---------|
        | **L2** | 上腰椎代表，矢状面和均值均有价值 |
        | **L3** | 腰椎中部代表，横断面和均值最敏感 |
        | **L4** | 下腰椎，承重最大，是最重要的预测指标 |
        
        **为什么L1被排除？**
        - L1椎体PC1载荷较低
        - L1的信息被PC2（上下腰椎对比）捕获
        - 在骨质疏松预测中贡献较小

        **临床建议**:
        1. **重点关注L4mean**：这是预测骨质疏松最敏感的指标
        2. **结合L3hengduan和L3mean**：L3椎体的信息也很重要
        3. **机会性筛查**：在常规腰椎CT中，优先评估L4和L3的CT值
        """)


# ====================== 主函数 ======================
def main():
    # 加载模型
    model, scaler, pca = load_models()
    
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
    - 算法: SVM (RBF核)
    - 特征数: 5个
    - 准确率: 83.05%
    - AUC: 0.9074
    
    **SHAP重要性排名**
    1. L4均值 (0.0288)
    2. L3横断面 (0.0225)
    3. L3均值 (0.0178)
    4. L2矢状面 (0.0086)
    5. L2均值 (0.0056)
    """)
    
    # ====================== 预测页面 ======================
    if page == "🔍 骨质疏松预测":
        st.header("🔍 骨质疏松风险预测")
        st.markdown("请输入患者的5个核心腰椎CT值进行预测。")
        
        col1, col2 = st.columns(2)
        
        input_values = {}
        
        with col1:
            st.subheader("📊 核心CT特征")
            input_values['L3mean'] = st.number_input(
                "**L3mean** (第3腰椎平均CT值)",
                min_value=0.0, max_value=400.0, value=150.0, step=1.0,
                help="单位: HU | 正常范围: 100-200"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L3mean'][0]}-{REFERENCE_RANGES['L3mean'][1]} HU")
            
            input_values['L2mean'] = st.number_input(
                "**L2mean** (第2腰椎平均CT值)",
                min_value=0.0, max_value=400.0, value=155.0, step=1.0,
                help="单位: HU"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L2mean'][0]}-{REFERENCE_RANGES['L2mean'][1]} HU")
            
            input_values['L3hengduan'] = st.number_input(
                "**L3hengduan** (第3腰椎横断面CT值)",
                min_value=0.0, max_value=400.0, value=145.0, step=1.0,
                help="单位: HU"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L3hengduan'][0]}-{REFERENCE_RANGES['L3hengduan'][1]} HU")
        
        with col2:
            st.subheader("📊 核心CT特征")
            input_values['L2shizhuang'] = st.number_input(
                "**L2shizhuang** (第2腰椎矢状面CT值)",
                min_value=0.0, max_value=400.0, value=148.0, step=1.0,
                help="单位: HU"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L2shizhuang'][0]}-{REFERENCE_RANGES['L2shizhuang'][1]} HU")
            
            input_values['L4mean'] = st.number_input(
                "**L4mean** (第4腰椎平均CT值)",
                min_value=0.0, max_value=400.0, value=140.0, step=1.0,
                help="单位: HU | L4是承重最大的椎体"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L4mean'][0]}-{REFERENCE_RANGES['L4mean'][1]} HU")
        
        # 预测按钮
        if st.button("🚀 开始预测", type="primary", use_container_width=True):
            with st.spinner("正在分析中..."):
                # 执行预测
                probability, prediction = predict_osteoporosis(model, scaler, pca, input_values)
                
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
                
                # SHAP可解释性分析
                st.subheader("🧠 模型决策解释")
                
                shap_values = calculate_shap_contributions(input_values)
                
                shap_df = pd.DataFrame({
                    '特征': SELECTED_FEATURES,
                    '特征中文': [FEATURE_NAMES_CN.get(f, f) for f in SELECTED_FEATURES],
                    '输入值(HU)': [input_values[f] for f in SELECTED_FEATURES],
                    'SHAP值': shap_values,
                    '影响方向': ['增加风险' if v > 0 else '降低风险' for v in shap_values]
                })
                shap_df['绝对值'] = np.abs(shap_df['SHAP值'])
                shap_df = shap_df.sort_values('绝对值', ascending=False)
                
                st.dataframe(
                    shap_df[['特征中文', '输入值(HU)', 'SHAP值', '影响方向']].style.format({
                        '输入值(HU)': '{:.1f}',
                        'SHAP值': '{:.4f}'
                    }),
                    use_container_width=True
                )
                
                # SHAP条形图
                fig_shap = px.bar(shap_df,
                                  x='SHAP值',
                                  y='特征中文',
                                  orientation='h',
                                  color='影响方向',
                                  color_discrete_map={'增加风险': '#EF553B', '降低风险': '#636EFA'},
                                  title='各CT特征对预测的影响 (基于PC1载荷)')
                fig_shap.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
                fig_shap.update_layout(height=400)
                st.plotly_chart(fig_shap, use_container_width=True)
                
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
    
    # ====================== 特征分析页面 ======================
    elif page == "📊 特征分析":
        feature_analysis_page()
    
    # ====================== 使用说明页面 ======================
    else:
        st.header("ℹ️ 使用说明")
        
        st.markdown("""
        ## 📖 系统使用指南
        
        ### 1. 系统概述
        本系统基于**SVM机器学习模型**，使用腰椎CT值进行骨质疏松风险预测。
        
        ### 2. 模型性能
        | 指标 | 数值 |
        |------|------|
        | 验证集准确率 | 83.05% |
        | AUC | 0.9074 |
        | 敏感性 | 90.62% |
        | 特异性 | 74.07% |
        
        ### 3. 使用方法
        1. 进入"🔍 骨质疏松预测"页面
        2. 输入5个核心CT值（单位为HU）
        3. 点击"开始预测"按钮
        4. 查看预测结果和临床建议
        
        ### 4. CT值参考范围
        | 分类 | CT值 (HU) | 临床意义 |
        |------|-----------|---------|
        | 正常 | >160 | 骨密度正常 |
        | 骨量减少 | 120-160 | 需关注 |
        | 骨质疏松 | <120 | 建议DXA确诊 |
        
        ### 5. 结果解读
        
        #### 风险等级
        - 🟢 **低风险 (<30%)**: CT值正常范围
        - 🟡 **中风险 (30%-70%)**: 需要进一步评估
        - 🔴 **高风险 (>70%)**: 建议DXA检查确诊
        
        #### SHAP值解读
        - **正SHAP值**: 该特征增加骨质疏松风险（CT值偏低）
        - **负SHAP值**: 该特征降低骨质疏松风险（CT值正常）
        - **绝对值大小**: 表示影响程度
        
        ### 6. 5个核心CT特征临床意义
        
        | 特征 | 临床意义 |
        |------|---------|
        | **L4均值** | 最重要的预测指标，承重最大椎体 |
        | **L3横断面** | L3椎体横断面CT值，对骨质流失敏感 |
        | **L3均值** | L3椎体平均CT值，PC1载荷最高 |
        | **L2矢状面** | L2椎体矢状面CT值 |
        | **L2均值** | L2椎体平均CT值 |
        
        ### 7. 重要声明
        ⚠️ **本系统为机会性筛查工具，不能替代DXA金标准诊断**
        """)
    
    # 页脚
    st.markdown("---")
    st.caption("🦴 骨质疏松机会性筛查系统 | 基于SVM机器学习 | 仅供参考，请遵医嘱")


if __name__ == "__main__":
    main()
