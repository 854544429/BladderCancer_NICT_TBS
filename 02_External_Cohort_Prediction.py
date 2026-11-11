import pandas as pd
import joblib
import pandas as pd
import numpy as np
# ==========================================
# 1. 配置路径 (请根据实际情况修改)
# ==========================================
# 你的新队列数据路径
new_data_path = './model/TCGA_blca_cohort.xlsx' 
# 之前保存的模型路径
model_filename = 'best_LR_model.pkl'
# 预测结果的输出路径
output_path = './model/TCGA_cohort_Predictions.xlsx'

# ==========================================
# 2. 加载数据与模型
# ==========================================
print(f"正在加载新队列数据: {new_data_path}")
df_new = pd.read_excel(new_data_path)

print(f"正在加载训练好的模型: {model_filename}")
model = joblib.load(model_filename)

# ==========================================
# 3. 提取特征
# ==========================================
# ⚠️ 注意：特征的名称和顺序必须与训练时完全一致！
features = ['Sex', 'Age', 'T', 'N', 'G', 'TB', 'TSR', 'TLS']

# 检查新队列是否包含所有必需的特征
missing_cols = [col for col in features if col not in df_new.columns]
if missing_cols:
    raise ValueError(f"错误：新数据集中缺失以下特征列，请检查数据：{missing_cols}")

X_new = df_new[features]

# 简单检查是否有缺失值 (如果有缺失值，模型预测会报错)
if X_new.isnull().values.any():
    print("⚠️ 警告：新队列中存在缺失值 (NaN)，建议先进行填补或删除，否则预测可能会报错！")

# ==========================================
# 4. 进行预测
# ==========================================
print("正在进行预测...")
# 预测二分类结果 (0 或 1)
predictions = model.predict(X_new)
# 预测概率 (取第二列，即类别 1 的概率)
probabilities = model.predict_proba(X_new)[:, 1]

# ==========================================
# 5. 结果合并与导出
# ==========================================
# 将预测结果作为新列追加到原始数据末尾
df_new['Predicted_Class'] = predictions
df_new['Predicted_Probability'] = probabilities

# 保存为新的 Excel 文件
df_new.to_excel(output_path, index=False)

print("\n🎉 预测完成！")
print(f"结果已成功保存至: {output_path}")
print("你可以打开该 Excel 文件，查看 'Predicted_Class' 和 'Predicted_Probability' 列。")