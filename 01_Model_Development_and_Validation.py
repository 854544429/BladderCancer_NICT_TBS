import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# ==========================================
# 1. 读取数据 (内部队列 + 外部验证队列)
# ==========================================
# 内部数据
df_internal = pd.read_excel('./model/NACI_MeachineLearning_SEC_CQ.xlsx')
# 外部数据 (请根据你的实际文件名修改此路径
df_external = pd.read_excel('./model/NACI_ExternalValidation_RMYY.xlsx')

features = ['Sex','Age','T','N','G','TB', 'TSR', 'TLS']
target = 'CR'

X_internal = df_internal[features]
y_internal = df_internal[target]

X_ext = df_external[features]
y_ext = df_external[target]

# ==========================================
# 2. 内部队列 7:3 拆分
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_internal, y_internal, test_size=0.3, random_state=42, stratify=y_internal
)

# ==========================================
# 3. 模型定义
# ==========================================
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, class_weight='balanced', random_state=42),
    'Support Vector Machine': SVC(probability=True, kernel='linear', class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
}

results_list = []
plt.figure(figsize=(10, 8), dpi=100)
colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']

# 【新增】用于收集各模型在各队列中预测概率的字典，方便后续导出画ROC
roc_export_data = {
    "Train_True_Labels": y_train.values,
    "Test_True_Labels": y_test.values,
    "Ext_True_Labels": y_ext.values
}

# ==========================================
# 4. 核心计算模块与绘图
# ==========================================
def get_metrics(model, X, y, dataset_name, model_name):
    """提取复用的指标计算逻辑"""
    y_pred = model.predict(X)
    y_probs = model.predict_proba(X)[:, 1]
    
    # 强制 labels=[0,1] 避免在极小样本集下只预测出一类而报错
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    youden = sensitivity + specificity - 1
    
    fpr, tpr, _ = roc_curve(y, y_probs)
    auc_val = auc(fpr, tpr)
    
    return {
        'Model': f"{model_name} ({dataset_name})",
        'AUC': round(auc_val, 3),
        'Sensitivity (灵敏度)': round(sensitivity, 3),
        'Specificity (特异度)': round(specificity, 3),
        'PPV (阳性预测值)': round(ppv, 3),
        'NPV (阴性预测值)': round(npv, 3),
        'Accuracy (准确率)': round(accuracy, 3),
        'Youden Index (约登指数)': round(youden, 3)
    }, fpr, tpr, auc_val

for i, (model_name, model) in enumerate(models.items()):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 【新增】收集各个模型的预测概率，用于后续自定义绘制ROC
    roc_export_data[f"{model_name}_Train_Prob"] = model.predict_proba(X_train)[:, 1]
    roc_export_data[f"{model_name}_IntTest_Prob"] = model.predict_proba(X_test)[:, 1]
    roc_export_data[f"{model_name}_External_Prob"] = model.predict_proba(X_ext)[:, 1]

    # 分别计算 Train, Internal Test, External 结果
    metrics_tr, fpr_tr, tpr_tr, auc_tr = get_metrics(model, X_train, y_train, "Train", model_name)
    metrics_te, fpr_te, tpr_te, auc_te = get_metrics(model, X_test, y_test, "Int Test", model_name)
    metrics_ext, fpr_ext, tpr_ext, auc_ext = get_metrics(model, X_ext, y_ext, "External", model_name)
    
    # 将三大数据集指标追加进总表
    results_list.extend([metrics_tr, metrics_te, metrics_ext])
    
    # --- 绘制内置 ROC 曲线 ---
    plt.plot(fpr_tr, tpr_tr, color=colors[i], linestyle=':', lw=1.2, alpha=0.5)
    plt.plot(fpr_te, tpr_te, color=colors[i], linestyle='--', lw=1.5, alpha=0.8,
             label=f'{model_name} (Int AUC = {auc_te:.2f})')
    plt.plot(fpr_ext, tpr_ext, color=colors[i], linestyle='-', lw=2.5,
             label=f'{model_name} (Ext AUC = {auc_ext:.2f})')

# ==========================================
# 5. 输出指标表格与 【新增】导出ROC作图数据
# ==========================================
results_df = pd.DataFrame(results_list)
print("\n--- 全队列预测性能评价 (Train vs Internal Test vs External Validation) ---")
print(results_df.to_string(index=False))

# 【新增模块】将所有模型的预测概率及真实标签导出为 Excel，供外部画图使用
# 由于各个队列的样本量不同（训练集 vs 测试集 vs 外部验证集），这里通过 DataFrame 分别组装后保存，或者存为字典
df_train_roc = pd.DataFrame({
    "True_Label": y_train.values,
    **{name: model.predict_proba(X_train)[:, 1] for name, model in models.items()}
})
df_test_roc = pd.DataFrame({
    "True_Label": y_test.values,
    **{name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}
})
df_ext_roc = pd.DataFrame({
    "True_Label": y_ext.values,
    **{name: model.predict_proba(X_ext)[:, 1] for name, model in models.items()}
})

with pd.ExcelWriter("All_Models_ROC_Data.xlsx") as writer:
    df_train_roc.to_excel(writer, sheet_name="Training_Cohort", index=False)
    df_test_roc.to_excel(writer, sheet_name="Internal_Test_Cohort", index=False)
    df_ext_roc.to_excel(writer, sheet_name="External_Validation_Cohort", index=False)

print("\n[成功] 用于自己画 ROC 曲线的数据已保存至: All_Models_ROC_Data.xlsx")
print("该 Excel 包含三个 Sheet: Training_Cohort, Internal_Test_Cohort, External_Validation_Cohort")

# 图形美化
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves: Internal Test vs External Validation')
plt.legend(loc="lower right", fontsize='small', frameon=True)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ==========================================
# 6. 保存模型 (保存逻辑回归)
# ==========================================
best_model = models['Logistic Regression']
model_filename = 'best_LR_model.pkl'
joblib.dump(best_model, model_filename)
print(f"\n模型已成功保存为: {model_filename}")

# ==========================================
# 7. 提取预测概率并初始化高级分析
# ==========================================
print("\n--- 正在提取预测概率并初始化高级分析 ---")
best_lr = models['Logistic Regression']

prob_test = best_lr.predict_proba(X_test)[:, 1]
prob_ext = best_lr.predict_proba(X_ext)[:, 1]

# ==========================================
# 8. SHAP 可释性分析 (使用原始特征尺度)
# ==========================================
import shap
from sklearn.calibration import calibration_curve

print("\n--- 正在生成并保存 SHAP 可视化分析 (PDF) ---")
explainer = shap.LinearExplainer(best_lr, X_train)
shap_values = explainer(X_ext)

# 1. Summary Plot (蜂群图)
plt.figure(figsize=(10, 6), dpi=300)
shap.summary_plot(shap_values, X_ext, feature_names=features, show=False)
plt.title("SHAP Summary Plot (External Validation)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("SHAP_Summary_Plot.pdf", format='pdf', bbox_inches='tight')
plt.close()
print("已保存: SHAP_Summary_Plot.pdf")

# 2. Bar Plot (特征绝对重要性排名)
plt.figure(figsize=(8, 5), dpi=300)
shap.summary_plot(shap_values, X_ext, feature_names=features, plot_type="bar", show=False)
plt.title("SHAP Feature Importance", fontsize=14, pad=15)
plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=11)
plt.tight_layout()
plt.savefig("SHAP_Bar_Plot.pdf", format='pdf', bbox_inches='tight')
plt.close()
print("已保存: SHAP_Bar_Plot.pdf")

# ==========================================
# 9. 决策曲线分析 (DCA)
# ==========================================
prob_train = best_lr.predict_proba(X_train)[:, 1]
prob_test  = best_lr.predict_proba(X_test)[:, 1]
prob_ext   = best_lr.predict_proba(X_ext)[:, 1]

print("\n--- 正在生成独立的 DCA 决策曲线 ---")

def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    n = len(y_true)
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        if pt == 1.0: 
            nb = 0
        else:
            nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    return np.array(net_benefits)

def smooth_line(y, box_pts=5):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[:box_pts] = y[:box_pts]
    y_smooth[-box_pts:] = y[-box_pts:]
    return y_smooth

thresholds = np.linspace(0.02, 0.90, 100)
nb_none = np.zeros_like(thresholds)

dca_tasks = [
    {"name": "Training Set", "y": y_train, "prob": prob_train, "color": "#4DAF4A", "box_pts": 5, "filename": "DCA_Curve_1_Training.pdf"},
    {"name": "Internal Test", "y": y_test, "prob": prob_test, "color": "#377EB8", "box_pts": 5, "filename": "DCA_Curve_2_InternalTest.pdf"},
    {"name": "External Validation", "y": y_ext, "prob": prob_ext, "color": "#E41A1C", "box_pts": 7, "filename": "DCA_Curve_3_ExternalVal.pdf"}
]

for task in dca_tasks:
    prevalence = np.mean(task["y"])
    nb_raw = calculate_net_benefit(task["y"], task["prob"], thresholds)
    nb_smooth = smooth_line(nb_raw, box_pts=task["box_pts"])
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    
    plt.figure(figsize=(7, 5.5), dpi=300)
    plt.plot(thresholds, nb_smooth, color=task["color"], lw=2.5, label=f'Model ({task["name"]})')
    plt.plot(thresholds, nb_all, color='gray', lw=1.5, linestyle='--', label='Treat All')
    plt.plot(thresholds, nb_none, color='black', lw=1.5, linestyle='-', label='Treat None')
    
    plt.xlim([0, 0.85]) 
    plt.ylim([-0.05, max(prevalence + 0.1, 0.4)]) 
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title(f'Decision Curve Analysis\n({task["name"]})', fontsize=13, pad=12, fontweight='bold')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    
    plt.savefig(task["filename"], format='pdf', bbox_inches='tight')
    plt.close()
    print(f"已保存: {task['filename']}")

# ==========================================
# 10. 校准曲线 (Calibration Curve)
# ==========================================
print("\n--- 正在生成独立的平滑版校准曲线 (PDF) ---")

cal_tasks = [
    {"name": "Training Set", "y": y_train, "prob": prob_train, "color": "#4DAF4A", "marker": "^", "filename": "Calibration_1_Training.pdf"},
    {"name": "Internal Test", "y": y_test, "prob": prob_test, "color": "#377EB8", "marker": "s", "filename": "Calibration_2_InternalTest.pdf"},
    {"name": "External Validation", "y": y_ext, "prob": prob_ext, "color": "#E41A1C", "marker": "o", "filename": "Calibration_3_ExternalVal.pdf"}
]

for task in cal_tasks:
    plt.figure(figsize=(6, 5.5), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Calibration', lw=1.5)
    
    prob_true, prob_pred = calibration_curve(task["y"], task["prob"], n_bins=6, strategy='quantile')
    poly = np.poly1d(np.polyfit(prob_pred, prob_true, 3))
    x_smooth = np.linspace(prob_pred.min(), prob_pred.max(), 100)
    
    plt.plot(x_smooth, poly(x_smooth), color=task["color"], lw=2.5, label=f'Observed ({task["name"]})')
    plt.scatter(prob_pred, prob_true, color=task["color"], alpha=0.8, s=45, marker=task["marker"], zorder=3)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives (Observed)', fontsize=12)
    plt.title(f'Calibration Curve\n({task["name"]})', fontsize=13, pad=12, fontweight='bold')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    
    plt.savefig(task["filename"], format='pdf', bbox_inches='tight')
    plt.close()
    print(f"已成功保存: {task['filename']}")