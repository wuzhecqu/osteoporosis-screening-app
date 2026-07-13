# -*- coding: utf-8 -*-
"""
决策树模型训练脚本 - 用于骨质疏松预测
使用LASSO筛选的5个核心特征
超参数: max_depth=7, min_samples_split=4, min_samples_leaf=2,
        max_features='sqrt', class_weight='balanced', ccp_alpha=0.01
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_curve, auc, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ===================== 路径配置 =====================
data_dir = r'F:\Project\zhouzhixi\prediction\datasplit'
output_dir = r'F:\Project\zhouzhixi\prediction\lassoCT_Clinical\models'
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("决策树模型训练 - 骨质疏松预测")
print("LASSO筛选特征: weight_kg, BMI, L2guanzhuang, L3shizhuang, L4shizhuang")
print("=" * 80)

# ===================== 读取数据 =====================
train_df = pd.read_csv(os.path.join(data_dir, 'train_data_binary.csv'), encoding='utf-8-sig')
val_df = pd.read_csv(os.path.join(data_dir, 'validation_data_binary.csv'), encoding='utf-8-sig')

# ===================== LASSO筛选的5个核心特征 =====================
SELECTED_FEATURES = [
    'weight_kg',
    'BMI',
    'L2guanzhuang',
    'L3shizhuang',
    'L4shizhuang'
]

# 完整的16个腰椎CT特征（用于PCA）
CT_FEATURES_FULL = [
    'L1hengduan', 'L1shizhuang', 'L1guanzhuang', 'L1mean',
    'L2hengduan', 'L2shizhuang', 'L2guanzhuang', 'L2mean',
    'L3hengduan', 'L3shizhuang', 'L3guanzhuang', 'L3mean',
    'L4hengduan', 'L4shizhuang', 'L4guanzhuang', 'L4mean'
]

# 所有特征 = CT特征 + 临床特征
ALL_FEATURES = CT_FEATURES_FULL + ['weight_kg', 'BMI']

print(f"\n核心特征 ({len(SELECTED_FEATURES)}个):")
for i, feat in enumerate(SELECTED_FEATURES, 1):
    print(f"  {i}. {feat}")

print(f"\n完整特征数: {len(ALL_FEATURES)}")

# ===================== 数据预处理 =====================
# 训练集
X_train = train_df[ALL_FEATURES].copy()
y_train = train_df['bmd_binary'].copy()

# 验证集
X_val = val_df[ALL_FEATURES].copy()
y_val = val_df['bmd_binary'].copy()

# 填充缺失值（使用训练集均值）
train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_val = X_val.fillna(train_means)

print(f"\n训练集: {X_train.shape}, 验证集: {X_val.shape}")
print(f"训练集标签分布: 非骨质疏松={sum(y_train==0)}, 骨质疏松={sum(y_train==1)}")
print(f"验证集标签分布: 非骨质疏松={sum(y_val==0)}, 骨质疏松={sum(y_val==1)}")

# ===================== 标准化 =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"\n数据标准化完成")

# ===================== PCA降维 =====================
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

print(f"\nPCA降维结果:")
print(f"  原始维度: {X_train_scaled.shape[1]}")
print(f"  降维后维度: {X_train_pca.shape[1]}")
print(f"  各主成分解释方差: {pca.explained_variance_ratio_}")
print(f"  累计解释方差: {sum(pca.explained_variance_ratio_):.4f}")

# ===================== 决策树模型训练 =====================
# 使用您指定的超参数
decision_tree = DecisionTreeClassifier(
    max_depth=7,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    ccp_alpha=0.01
)

print("\n" + "=" * 60)
print("训练决策树模型...")
print(f"超参数: max_depth=7, min_samples_split=4, min_samples_leaf=2")
print(f"       max_features='sqrt', class_weight='balanced', ccp_alpha=0.01")
print("=" * 60)

decision_tree.fit(X_train_pca, y_train)

# ===================== 模型评估 =====================
# 验证集预测
y_pred = decision_tree.predict(X_val_pca)
y_pred_proba = decision_tree.predict_proba(X_val_pca)[:, 1]

# 计算指标
accuracy = accuracy_score(y_val, y_pred)
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
ap_score = average_precision_score(y_val, y_pred_proba)

print(f"\n📊 验证集性能:")
print(f"  准确率 (Accuracy): {accuracy:.4f}")
print(f"  AUC: {roc_auc:.4f}")
print(f"  AP (Average Precision): {ap_score:.4f}")

# 分类报告
print("\n" + "-" * 60)
print("分类报告:")
print("-" * 60)
print(classification_report(y_val, y_pred, target_names=['非骨质疏松', '骨质疏松']))

# 混淆矩阵
cm = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\n混淆矩阵:")
print(f"  TN={tn}, FP={fp}")
print(f"  FN={fn}, TP={tp}")
print(f"  敏感性 (Sensitivity): {tp/(tp+fn):.4f}")
print(f"  特异性 (Specificity): {tn/(tn+fp):.4f}")
print(f"  阳性预测值 (PPV): {tp/(tp+fp):.4f}")
print(f"  阴性预测值 (NPV): {tn/(tn+fn):.4f}")

# ===================== 保存模型文件 =====================
print("\n" + "=" * 60)
print("保存模型文件...")
print("=" * 60)

# 1. 保存决策树模型
joblib.dump(decision_tree, os.path.join(output_dir, 'best_model.pkl'))
print(f"✅ 决策树模型已保存: {os.path.join(output_dir, 'best_model.pkl')}")

# 2. 保存标准化器
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
print(f"✅ 标准化器已保存: {os.path.join(output_dir, 'scaler.pkl')}")

# 3. 保存PCA模型
joblib.dump(pca, os.path.join(output_dir, 'pca_model.pkl'))
print(f"✅ PCA模型已保存: {os.path.join(output_dir, 'pca_model.pkl')}")

# 4. 保存特征列表
with open(os.path.join(output_dir, 'features.txt'), 'w', encoding='utf-8') as f:
    f.write("5个核心特征:\n")
    for feat in SELECTED_FEATURES:
        f.write(f"  {feat}\n")
    f.write(f"\n完整特征数: {len(ALL_FEATURES)}\n")
print(f"✅ 特征列表已保存: {os.path.join(output_dir, 'features.txt')}")

# 5. 保存模型信息
model_info = {
    'model_type': 'DecisionTree',
    'model_params': {
        'max_depth': 7,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'ccp_alpha': 0.01,
        'random_state': 42
    },
    'selected_features': SELECTED_FEATURES,
    'all_features': ALL_FEATURES,
    'n_features': len(ALL_FEATURES),
    'n_components': 5,
    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
    'cumulative_variance': float(sum(pca.explained_variance_ratio_)),
    'train_samples': len(y_train),
    'val_samples': len(y_val),
    'performance': {
        'accuracy': float(accuracy),
        'auc': float(roc_auc),
        'ap_score': float(ap_score),
        'sensitivity': float(tp/(tp+fn)),
        'specificity': float(tn/(tn+fp)),
        'ppv': float(tp/(tp+fp)),
        'npv': float(tn/(tn+fn))
    },
    'confusion_matrix': cm.tolist(),
    'label_map': {'0': '非骨质疏松', '1': '骨质疏松'}
}

with open(os.path.join(output_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)
print(f"✅ 模型信息已保存: {os.path.join(output_dir, 'model_info.json')}")

# ===================== 绘制评估图表 =====================
# 1. ROC曲线 + PR曲线 + 混淆矩阵
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ROC曲线
axes[0].plot(fpr, tpr, lw=2.5, label=f'Decision Tree (AUC = {roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (0.5)')
axes[0].set_xlabel('False Positive Rate (1-Specificity)', fontsize=11)
axes[0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
axes[0].set_title('ROC Curve', fontsize=12, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].grid(alpha=0.3)

# PR曲线
axes[1].plot(recall, precision, lw=2.5, label=f'Decision Tree (AP = {ap_score:.4f})')
axes[1].set_xlabel('Recall (Sensitivity)', fontsize=11)
axes[1].set_ylabel('Precision', fontsize=11)
axes[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
axes[1].legend(loc='lower left', fontsize=9)
axes[1].grid(alpha=0.3)

# 混淆矩阵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['非骨质疏松', '骨质疏松'],
            yticklabels=['非骨质疏松', '骨质疏松'],
            annot_kws={'size': 14})
axes[2].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[2].set_xlabel('预测标签', fontsize=11)
axes[2].set_ylabel('真实标签', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'decision_tree_evaluation.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'decision_tree_evaluation.pdf'), bbox_inches='tight')
plt.close()
print(f"✅ 评估图表已保存: {os.path.join(output_dir, 'decision_tree_evaluation.png')}")

# ===================== 绘制特征重要性（PC1载荷） =====================
fig, ax = plt.subplots(figsize=(12, 8))

# 获取PC1载荷
loadings = pca.components_[0]
feat_loadings = pd.DataFrame({
    '特征': ALL_FEATURES,
    'PC1载荷': loadings
}).sort_values('PC1载荷', ascending=True)

# 标记核心特征
colors = ['#FF6B6B' if f in SELECTED_FEATURES else '#4ECDC4' for f in feat_loadings['特征']]
bars = ax.barh(feat_loadings['特征'], feat_loadings['PC1载荷'], color=colors)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)

# 标记核心特征
for i, (feat, loading) in enumerate(zip(feat_loadings['特征'], feat_loadings['PC1载荷'])):
    if feat in SELECTED_FEATURES:
        ax.text(loading + 0.01, i, '★ 核心特征',
                va='center', fontsize=10, color='red', fontweight='bold')

ax.set_xlabel('PC1载荷系数', fontsize=12)
ax.set_title('PCA第一主成分载荷\n(解释 {:.2f}% 方差)'.format(pca.explained_variance_ratio_[0] * 100),
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_loadings.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'pca_loadings.pdf'), bbox_inches='tight')
plt.close()
print(f"✅ PCA载荷图已保存: {os.path.join(output_dir, 'pca_loadings.png')}")

# ===================== 打印最终结果 =====================
print("\n" + "=" * 80)
print("🎯 训练完成! 模型文件已保存")
print("=" * 80)
print(f"""
输出目录: {output_dir}
├── best_model.pkl              # 决策树模型
├── scaler.pkl                  # 标准化器
├── pca_model.pkl               # PCA模型
├── features.txt                # 特征列表
├── model_info.json             # 模型信息
├── decision_tree_evaluation.png # 评估图表
└── pca_loadings.png            # PCA载荷图

模型性能摘要:
├── 准确率 (Accuracy):  {accuracy:.4f}
├── AUC:                 {roc_auc:.4f}
├── AP:                  {ap_score:.4f}
├── 敏感性 (Sensitivity): {tp/(tp+fn):.4f}
├── 特异性 (Specificity): {tn/(tn+fp):.4f}
├── 阳性预测值 (PPV):    {tp/(tp+fp):.4f}
└── 阴性预测值 (NPV):    {tn/(tn+fn):.4f}
""")

print("=" * 80)
