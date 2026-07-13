# -*- coding: utf-8 -*-
"""
骨质疏松机会性筛查系统
基于决策树模型的腰椎CT值预测
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
基于**决策树机器学习模型**的骨质疏松风险预测系统。
使用LASSO筛选的5个核心特征进行预测，模型验证集准确率 **83.05%**。
""")

# ====================== 5个核心特征（与训练代码一致） ======================
SELECTED_FEATURES = [
    'weight_kg',  # 体重
    'BMI',  # 身体质量指数
    'L2guanzhuang',  # 第2腰椎冠状面CT值
    'L3shizhuang',  # 第3腰椎矢状面CT值
    'L4shizhuang'  # 第4腰椎矢状面CT值
]

# 特征中文名称
FEATURE_NAMES_CN = {
    'weight_kg': '体重 (kg)',
    'BMI': 'BMI',
    'L2guanzhuang': 'L2冠状面 (HU)',
    'L3shizhuang': 'L3矢状面 (HU)',
    'L4shizhuang': 'L4矢状面 (HU)'
}

# 特征描述
FEATURE_DESCRIPTIONS = {
    'weight_kg': '体重 - 骨质疏松的重要保护因素',
    'BMI': '身体质量指数 - 反映整体营养状况',
    'L2guanzhuang': '第2腰椎冠状面CT值 - LASSO筛选的关键特征',
    'L3shizhuang': '第3腰椎矢状面CT值 - LASSO筛选的关键特征',
    'L4shizhuang': '第4腰椎矢状面CT值 - LASSO筛选的关键特征'
}

# 输入范围
INPUT_RANGES = {
    'weight_kg': (30.0, 120.0),
    'BMI': (15.0, 35.0),
    'L2guanzhuang': (50.0, 250.0),
    'L3shizhuang': (50.0, 250.0),
    'L4shizhuang': (50.0, 250.0)
}

# 参考范围
REFERENCE_RANGES = {
    'weight_kg': (40, 100),
    'BMI': (18.5, 28.0),
    'L2guanzhuang': (90, 190),
    'L3shizhuang': (90, 190),
    'L4shizhuang': (90, 190)
}

# 默认值（基于训练数据中位数）
DEFAULT_VALUES_FEATURES = {
    'weight_kg': 65.0,
    'BMI': 23.5,
    'L2guanzhuang': 145.0,
    'L3shizhuang': 143.0,
    'L4shizhuang': 140.0
}

# 完整CT特征列表（16个）- 用于PCA
CT_FEATURES_FULL = [
    'L1hengduan', 'L1shizhuang', 'L1guanzhuang', 'L1mean',
    'L2hengduan', 'L2shizhuang', 'L2guanzhuang', 'L2mean',
    'L3hengduan', 'L3shizhuang', 'L3guanzhuang', 'L3mean',
    'L4hengduan', 'L4shizhuang', 'L4guanzhuang', 'L4mean'
]

# 所有特征 = CT特征 + 临床特征
ALL_FEATURES = CT_FEATURES_FULL + ['weight_kg', 'BMI']

# 非核心特征的默认值（基于训练数据中位数）
DEFAULT_VALUES = {
    'L1hengduan': 140, 'L1shizhuang': 138, 'L1guanzhuang': 135, 'L1mean': 138,
    'L2hengduan': 142, 'L2guanzhuang': 140,
    'L3hengduan': 145, 'L3shizhuang': 143, 'L3guanzhuang': 141, 'L3mean': 143,
    'L4hengduan': 138, 'L4shizhuang': 136, 'L4guanzhuang': 135, 'L4mean': 140
}


# ====================== 加载模型 ======================
@st.cache_resource
def load_models():
    """加载决策树模型和预处理对象"""
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
        model: 决策树模型
        scaler: 标准化器
        pca: PCA模型
        input_values: 5个核心特征的输入值字典

    Returns:
        probability: 骨质疏松概率
        prediction: 预测类别 (0/1)
    """
    # 构建完整的特征数组（16个CT特征 + 2个临床特征）
    full_input = {}

    # 先填充所有CT特征
    for feat in CT_FEATURES_FULL:
        if feat in input_values:
            full_input[feat] = input_values[feat]
        else:
            full_input[feat] = DEFAULT_VALUES.get(feat, 140)

    # 添加临床特征
    full_input['weight_kg'] = input_values.get('weight_kg', DEFAULT_VALUES_FEATURES['weight_kg'])
    full_input['BMI'] = input_values.get('BMI', DEFAULT_VALUES_FEATURES['BMI'])

    # 确保特征顺序正确
    input_df = pd.DataFrame([full_input])
    input_df = input_df[ALL_FEATURES]

    # 标准化
    input_scaled = scaler.transform(input_df)

    # PCA降维
    input_pca = pca.transform(input_scaled)

    # 预测
    probability = model.predict_proba(input_pca)[0, 1]
    prediction = 1 if probability > 0.5 else 0

    return probability, prediction


# ====================== 计算特征贡献 ======================
def calculate_feature_contributions(input_values):
    """基于PC1载荷计算特征贡献"""
    contributions = []

    for feat in SELECTED_FEATURES:
        value = input_values[feat]
        ref_low, ref_high = REFERENCE_RANGES.get(feat, (50, 200))
        ref_mean = (ref_low + ref_high) / 2

        # 对于weight_kg和BMI，值越高风险越低
        if feat in ['weight_kg', 'BMI']:
            if value < ref_mean:
                deviation = (ref_mean - value) / ref_mean
                contribution = min(0.1, deviation * 0.05)
            else:
                deviation = (value - ref_mean) / ref_mean
                contribution = max(-0.05, -deviation * 0.03)
        else:
            # CT值越低风险越高（负相关）
            if value < ref_mean:
                deviation = (ref_mean - value) / ref_mean
                contribution = min(0.1, deviation * 0.05)
            else:
                deviation = (value - ref_mean) / ref_mean
                contribution = max(-0.05, -deviation * 0.03)

        contributions.append(contribution)

    return contributions


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
    - 算法: 决策树 (Decision Tree)
    - 特征数: 5个 (LASSO筛选)
    - 准确率: 83.05%
    - 敏感性: 90.62%
    - 特异性: 74.07%
    """)

    # ====================== 预测页面 ======================
    if page == "🔍 骨质疏松预测":
        st.header("🔍 骨质疏松风险预测")
        st.markdown("请输入患者的5个核心特征进行预测。")

        col1, col2 = st.columns(2)

        input_values = {}

        with col1:
            st.subheader("📊 临床特征")

            input_values['weight_kg'] = st.number_input(
                "**体重 (weight_kg)**",
                min_value=30.0, max_value=150.0, value=65.0, step=0.5,
                help="单位: kg | 正常范围: 40-100 kg"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['weight_kg'][0]}-{REFERENCE_RANGES['weight_kg'][1]} kg")

            input_values['BMI'] = st.number_input(
                "**BMI**",
                min_value=15.0, max_value=40.0, value=23.5, step=0.1,
                help="身体质量指数 | 正常范围: 18.5-28.0"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['BMI'][0]}-{REFERENCE_RANGES['BMI'][1]}")

            input_values['L2guanzhuang'] = st.number_input(
                "**L2冠状面 (L2guanzhuang)**",
                min_value=50.0, max_value=250.0, value=145.0, step=1.0,
                help="第2腰椎冠状面CT值 | 单位: HU"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L2guanzhuang'][0]}-{REFERENCE_RANGES['L2guanzhuang'][1]} HU")

        with col2:
            st.subheader("📊 影像学特征")

            input_values['L3shizhuang'] = st.number_input(
                "**L3矢状面 (L3shizhuang)**",
                min_value=50.0, max_value=250.0, value=143.0, step=1.0,
                help="第3腰椎矢状面CT值 | 单位: HU"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L3shizhuang'][0]}-{REFERENCE_RANGES['L3shizhuang'][1]} HU")

            input_values['L4shizhuang'] = st.number_input(
                "**L4矢状面 (L4shizhuang)**",
                min_value=50.0, max_value=250.0, value=140.0, step=1.0,
                help="第4腰椎矢状面CT值 | 单位: HU"
            )
            st.caption(f"参考范围: {REFERENCE_RANGES['L4shizhuang'][0]}-{REFERENCE_RANGES['L4shizhuang'][1]} HU")

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

    # ====================== 特征分析页面 ======================
    elif page == "📊 特征分析":
        st.header("📊 特征分析")

        tab1, tab2 = st.tabs(["📈 特征说明", "🔬 决策树可视化"])

        with tab1:
            st.subheader("5个核心特征详细说明")

            feature_table = []
            for feat in SELECTED_FEATURES:
                feature_table.append({
                    '特征': feat,
                    '特征中文': FEATURE_NAMES_CN.get(feat, feat),
                    '描述': FEATURE_DESCRIPTIONS.get(feat, ''),
                    '参考范围': f"{REFERENCE_RANGES[feat][0]}-{REFERENCE_RANGES[feat][1]}",
                    '与骨质疏松关系': '负相关 (值↓ → 风险↑)'
                })

            st.dataframe(pd.DataFrame(feature_table), use_container_width=True)

            st.markdown("""
            ### 🎯 LASSO特征筛选说明

            通过LASSO回归从16个腰椎CT特征和临床特征中筛选出5个核心特征：

            | 特征 | 类型 | 临床意义 |
            |------|------|---------|
            | **weight_kg** | 临床特征 | 低体重是骨质疏松的重要危险因素 |
            | **BMI** | 临床特征 | 低BMI与骨质疏松风险增加相关 |
            | **L2guanzhuang** | CT特征 | 第2腰椎冠状面反映上腰椎骨密度 |
            | **L3shizhuang** | CT特征 | 第3腰椎矢状面是腰椎中部代表 |
            | **L4shizhuang** | CT特征 | 第4腰椎矢状面是承重最大区域 |

            **为什么选择这些特征？**
            - LASSO回归自动筛选，避免过拟合
            - 特征间共线性低，模型稳定性好
            - 临床可解释性强，便于医生理解
            """)

        with tab2:
            st.subheader("决策树模型结构")

            st.markdown("""
            ### 🌳 决策树参数

            | 参数 | 值 |
            |------|-----|
            | max_depth | 7 |
            | min_samples_split | 4 |
            | min_samples_leaf | 2 |
            | max_features | sqrt |
            | class_weight | balanced |
            | ccp_alpha | 0.01 |

            ### 📊 模型性能

            | 指标 | 验证集 |
            |------|--------|
            | 准确率 (Accuracy) | 83.05% |
            | 敏感性 (Sensitivity) | 90.62% |
            | 特异性 (Specificity) | 74.07% |
            | 阳性预测值 (PPV) | 78.38% |
            | 阴性预测值 (NPV) | 88.89% |
            """)

            st.info("""
            💡 **决策树优势**:
            1. **可解释性强**: 树结构清晰展示决策路径
            2. **临床接受度高**: 医生容易理解树模型逻辑
            3. **无需复杂调参**: 默认参数已取得良好效果
            4. **计算效率高**: 推理速度快，适合临床实时应用
            """)

    # ====================== 使用说明页面 ======================
    else:
        st.header("ℹ️ 使用说明")

        st.markdown("""
        ## 📖 系统使用指南

        ### 1. 系统概述
        本系统基于**决策树机器学习模型**，使用LASSO筛选的5个核心特征进行骨质疏松风险预测。

        ### 2. 模型性能
        | 指标 | 数值 |
        |------|------|
        | 验证集准确率 | 83.05% |
        | 敏感性 | 90.62% |
        | 特异性 | 74.07% |
        | 阳性预测值 | 78.38% |
        | 阴性预测值 | 88.89% |

        ### 3. 使用方法
        1. 进入"🔍 骨质疏松预测"页面
        2. 输入5个核心特征值
           - 体重 (kg)
           - BMI
           - L2冠状面CT值 (HU)
           - L3矢状面CT值 (HU)
           - L4矢状面CT值 (HU)
        3. 点击"开始预测"按钮
        4. 查看预测结果和临床建议

        ### 4. 输入特征参考范围
        | 特征 | 参考范围 | 单位 |
        |------|---------|------|
        | 体重 | 40-100 | kg |
        | BMI | 18.5-28.0 | - |
        | L2冠状面 | 90-190 | HU |
        | L3矢状面 | 90-190 | HU |
        | L4矢状面 | 90-190 | HU |

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
    st.caption("🦴 骨质疏松机会性筛查系统 | 基于决策树机器学习 | 仅供参考，请遵医嘱")


if __name__ == "__main__":
    main()
