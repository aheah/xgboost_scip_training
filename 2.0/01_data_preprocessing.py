import pandas as pd
import numpy as np
import os


def prepare_xgboost_training_data(cuts_file_path, gap_file_path, output_file_path):
    print("🚀 [阶段一] 开始构建 XGBoost 终极训练集...\n")

    # ==========================================
    # 第一步：跨目录读取原始数据
    # ==========================================
    print("[1/4] 正在跨目录读取 1.0 的原始 CSV 文件...")
    if not os.path.exists(cuts_file_path) or not os.path.exists(gap_file_path):
        print("❌ 错误：找不到原始数据文件！请检查路径和文件名是否正确。")
        return

    df_cuts = pd.read_csv(cuts_file_path)
    df_gap = pd.read_csv(gap_file_path)

    # 🚨 修复点 1：各回各家，各找各妈
    # 割平面表按 timestamp 排序，Gap表按 time 排序
    df_cuts = df_cuts.sort_values('timestamp').reset_index(drop=True)
    df_gap = df_gap.sort_values('time').reset_index(drop=True)

    print(f"  ✓ 成功读取候选割平面: {len(df_cuts)} 条")
    print(f"  ✓ 成功读取 Gap 流水记录: {len(df_gap)} 条")

    # ==========================================
    # 第二步：时间线对齐与标签判定 (终极修复版)
    # ==========================================
    print("\n[2/4] 正在进行时间线对齐与标签(0/1)判定...")

    df_merged = pd.merge_asof(
        df_cuts,
        df_gap[['time', 'dual_bound']],  # 👈 放弃 gap，改用最严谨的 dual_bound（地板）
        left_on='timestamp',
        right_on='time',
        direction='forward',
        suffixes=('_current', '_next'),
        allow_exact_matches=False  # 👈 救命参数：绝不允许原地踏步，必须找严格发生在此之后的记录！
    )

    epsilon = 1e-6

    # ⚖️ 全新判卷逻辑：如果未来的地板（dual_bound_next）严格大于现在的地板，说明下界被抬升了！极品好割！
    df_merged['Label'] = np.where(
        df_merged['dual_bound_next'] > df_merged['dual_bound_current'] + epsilon,
        1,
        0
    )

    count_0 = len(df_merged[df_merged['Label'] == 0])
    count_1 = len(df_merged[df_merged['Label'] == 1])
    print(f"  ✓ 标签计算完成！")
    print(f"  ⚠️ 数据分布极其不平衡 -> 极品好割(1): {count_1} 条 | 冗余废割(0): {count_0} 条")

    if count_1 > 0:
        weight_ratio = count_0 / count_1
        print(f"  💡 【重要记录】建议在下一步训练 XGBoost 时，设置 scale_pos_weight = {weight_ratio:.2f}")

    # ==========================================
    # 第三步：过河拆桥，大清洗防作弊
    # ==========================================
    print("\n[3/4] 正在清洗非数学特征，防止 AI 数据穿越...")

    # 把跟对偶界相关的也全部加进黑名单撕掉
    columns_to_drop = [
        'timestamp', 'time', 'gap', 'gap_percent',
        'dual_bound_current', 'dual_bound_next',  # 👈 注意这里名字变了
        'dual_bound', 'primal_bound', 'node_id', 'round_id'
    ]

    actual_cols_to_drop = [col for col in columns_to_drop if col in df_merged.columns]
    df_final = df_merged.drop(columns=actual_cols_to_drop)

    print(f"  ✓ 已销毁作弊特征: {actual_cols_to_drop}")
    print(f"  ✓ 成功保留特征: {[col for col in df_final.columns if col != 'Label']}")

    # ==========================================
    # 第四步：复印装订，保存数据
    # ==========================================
    print(f"\n[4/4] 正在将终极训练集保存至: {output_file_path}")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df_final.to_csv(output_file_path, index=False)
    print("🎉 恭喜！完美数据集已生成，随时可以开启炼丹炉！")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dir_1_0_data = os.path.join(project_root, '1.0', 'data')

    # ⚠️ 请确保这里是你目前最新的文件名！
    actual_cuts_filename = "aflow30a\\20260319_181443\\separator_data_20260319_181443.csv"
    actual_gap_filename = "aflow30a\\20260319_181443\\gap_20260319_181443.csv"

    CUTS_FILE = os.path.join(dir_1_0_data, actual_cuts_filename)
    GAP_FILE = os.path.join(dir_1_0_data, actual_gap_filename)

    dir_2_0_data = os.path.join(current_dir, 'data')
    OUTPUT_FILE = os.path.join(dir_2_0_data, "train_data_final.csv")

    prepare_xgboost_training_data(CUTS_FILE, GAP_FILE, OUTPUT_FILE)