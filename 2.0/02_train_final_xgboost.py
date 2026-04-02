import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os


def train_xgboost_model(data_path, model_output_path):
    print("🔥 [阶段二] 开始启动 XGBoost 炼丹炉...\n")

    # ==========================================
    # 1. 拿取完美试卷
    # ==========================================
    if not os.path.exists(data_path):
        print("❌ 找不到训练数据！请确认 01_data_preprocessing.py 是否已成功生成 CSV。")
        return

    df = pd.read_csv(data_path)
    print(f"[1/5] 成功读取终极训练集，共 {len(df)} 条数据。")

    # ==========================================
    # 2. 扫雷行动：精准提取纯数字特征
    # ==========================================
    print("[2/5] 正在剥离字符串，锁定纯数学特征...")

    # 🚨 我们在这里硬编码指定：只允许纯数字的微观/宏观数学特征进入模型！
    # 坚决把 cut_name, sample_vars 等字符串，以及 node_depth 等环境特征拦在门外！
    pure_math_features = [
        'cut_efficacy',  # 效能 (切得深不深)
        'parallelism',  # 平行度 (正交性)
        'var_count',  # 变量总数
        'non_zero_coefs',  # 非零变量数 (稠密度)
        'row_norm',  # 行范数 (数值稳定性)
        'max_coef',  # 最大系数
        'min_coef',  # 最小系数
        'avg_coef',  # 平均系数
        'is_local'  # 是否局部割 (通常是 0 或 1)
    ]

    # 防止有些列你没提取出来导致报错，做个交集过滤
    features_to_use = [col for col in pure_math_features if col in df.columns]
    print(f"  ✓ 最终喂给 AI 的纯数学显微镜 ({len(features_to_use)} 个特征): {features_to_use}")

    # 考卷 (X) 和 答案 (y)
    X = df[features_to_use]
    y = df['Label']

    # ==========================================
    # 3. 划分考场
    # ==========================================
    print("\n[3/5] 正在划分考场 (80% 日常练习, 20% 期末考试)...")
    # random_state=42 保证每次划分的结果一样，方便我们复现
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # 4. 初始化神仙大模型并疯狂炼丹
    # ==========================================
    # 💡 极其重要：根据上一步测算出来的 0.52 填入这里！
    WEIGHT_RATIO = 0.52

    print(f"\n[4/5] ⚙️ 正在训练 XGBoost (已开启极端不平衡矫正: scale_pos_weight={WEIGHT_RATIO})...")
    model = XGBClassifier(
        scale_pos_weight=WEIGHT_RATIO,
        n_estimators=100,  # 种 100 棵决策树
        max_depth=6,  # 每棵树最高 6 层（防止学成死记硬背的书呆子）
        learning_rate=0.1,  # 学习率
        random_state=42,
        n_jobs=-1  # 火力全开！调用你电脑所有的 CPU 核心一起算
    )

    # 开始拟合（也就是学习的过程）
    model.fit(X_train, y_train)

    # ==========================================
    # 5. 阅卷评分与保存
    # ==========================================
    print("\n[5/5] 📊 模型期末考试成绩单：")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  🏆 综合准确率: {accuracy * 100:.2f}%\n")

    # 打印详细体检报告
    print(classification_report(y_test, y_pred, target_names=['废割(0)', '极品好割(1)']))

    # 💡 AI 揭秘：它到底觉得哪个特征最重要？
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features_to_use, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print("\n💡 AI 摸索出的【运筹学特征重要性排行榜】:")
    print(feature_importance_df.to_string(index=False))

    # 保存这颗浓缩了顶级运筹智慧的仙丹
    model.save_model(model_output_path)
    print(f"\n🎉 大功告成！模型已保存至: {model_output_path}")


if __name__ == "__main__":
    # 获取 2.0 文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 刚才生成的完美 CSV 路径
    DATA_PATH = os.path.join(current_dir, 'data', 'train_data_final.csv')

    # 💡 你的提速版代码里读取的就是这个名字，我们直接保存成它！
    MODEL_OUTPUT_PATH = os.path.join(current_dir, 'xgboost_gap_elite_model.json')

    # 启动！
    train_xgboost_model(DATA_PATH, MODEL_OUTPUT_PATH)