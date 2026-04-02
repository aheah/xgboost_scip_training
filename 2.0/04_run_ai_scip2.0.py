import os
import time
import numpy as np
import pandas as pd
import traceback
import xgboost as xgb
from pyscipopt import Model, SCIP_RESULT
from pyscipopt.scip import Cutsel


# ==========================================
# 🧠 核心提取器 (保持不变)
# ==========================================
def extract_9_pure_math_features(model, cut):
    try:
        efficacy = model.getCutEfficacy(cut)
    except:
        efficacy = 0.0

    try:
        norm = cut.getNorm()
        parallelism = 1.0 / norm if norm > 1e-6 else 0.0
    except:
        parallelism = 0.0

    var_count, non_zero_coefs, row_norm, max_coef, min_coef, avg_coef, is_local = 0, 0, 0.0, 0.0, 0.0, 0.0, 0
    try:
        cols = cut.getCols()
        coefs = cut.getVals()
        if cols and coefs:
            var_count = len(cols)
            valid_coefs = [abs(c) for c in coefs if abs(c) > 1e-6]
            non_zero_coefs = len(valid_coefs)
            if valid_coefs:
                max_coef, min_coef, avg_coef = max(valid_coefs), min(valid_coefs), sum(valid_coefs) / len(valid_coefs)
        row_norm = cut.getNorm()
        is_local = int(cut.isLocal())
    except:
        pass

    return [efficacy, parallelism, var_count, non_zero_coefs, row_norm, max_coef, min_coef, avg_coef, is_local]


# ==========================================
# 🛡️ 究极提速版 AI 安检员 (批量判卷大法)
# ==========================================
class XGBoostCutSelector(Cutsel):
    def __init__(self, model_path):
        super().__init__()
        self.ai_model = xgb.XGBClassifier()
        self.ai_model.load_model(model_path)
        self.total_inspected = 0
        self.total_rejected = 0
        self.total_accepted = 0

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        try:
            accepted_cuts = []
            rejected_cuts = []
            feature_names = ['cut_efficacy', 'parallelism', 'var_count', 'non_zero_coefs', 'row_norm', 'max_coef',
                             'min_coef', 'avg_coef', 'is_local']

            # 🚀 提速核心：不要一个一个算！先把这批次的考卷全部收集起来！
            batch_features = []
            cut_indices_to_predict = []  # 记录哪些是真正的生成割

            for i, cut in enumerate(cuts):
                if cut.getOrigintype() == 3:
                    self.total_inspected += 1
                    batch_features.append(extract_9_pure_math_features(self.model, cut))
                    cut_indices_to_predict.append(i)
                else:
                    # 不是生成割，直接放进通过名单
                    accepted_cuts.append(cut)

            # 🚀 提速核心：一次性发给 Pandas 和 XGBoost！（几十倍提速）
            if batch_features:
                X_batch = pd.DataFrame(batch_features, columns=feature_names)

                # 💡 优化 1：不要死板的 0/1，我们要看 AI 给出是好割的“概率”
                probabilities = self.ai_model.predict_proba(X_batch)[:, 1]

                # 发放结果
                for idx, prob in enumerate(probabilities):
                    real_cut = cuts[cut_indices_to_predict[idx]]
                    # 获取这个割平面的真实效能 (特征列表里的第 0 个就是 cut_efficacy)
                    efficacy = batch_features[idx][0]

                    # 💡 优化 2：【终极双保险逻辑】
                    # 条件 A: AI 觉得是个好割 (概率 >= 0.4，稍微放宽一点 AI 的严苛度)
                    # 条件 B: 专家兜底！只要切得极其深 (效能 > 0.05)，哪怕 AI 说是垃圾也强制放行！
                    if prob >= 0.4 or efficacy > 0.05:
                        accepted_cuts.append(real_cut)
                        self.total_accepted += 1
                    else:
                        rejected_cuts.append(real_cut)
                        self.total_rejected += 1

            # 拼合返回
            reordered_cuts = accepted_cuts + rejected_cuts
            final_nselected = min(len(accepted_cuts), maxnselectedcuts)

            return {'cuts': reordered_cuts, 'nselectedcuts': final_nselected, 'result': SCIP_RESULT.SUCCESS}

        except Exception as e:
            traceback.print_exc()
            return {'cuts': cuts, 'nselectedcuts': min(maxnselectedcuts, len(cuts)), 'result': SCIP_RESULT.SUCCESS}


# ==========================================
# 🚀 启动器 (复刻你 1.0 的极限规则，并完美打印状态)
# ==========================================
def run_ai_scip(mps_file_path, model_path):
    print(f"\n🚀 启动终极测试：正在求解 {os.path.basename(mps_file_path)}")
    scip = Model("AI_Accelerated_SCIP")
    scip.readProblem(mps_file_path)

    # 完全对齐 1.0 的 Configuration.py
    scip.setParam("limits/time", 180)
    scip.setParam("limits/nodes", 10000)
    scip.setParam("separating/maxrounds", 5)
    scip.setParam("separating/maxroundsroot", 10)
    scip.setParam("separating/maxstallrounds", 3)
    scip.setParam("separating/maxcuts", 10)
    scip.setParam("separating/maxcutsroot", 5)

    heuristic_freq_params = [
        'heuristics/feaspump/freq', 'heuristics/alns/freq', 'heuristics/rens/freq',
        'heuristics/multistart/freq', 'heuristics/crossover/freq', 'heuristics/mutation/freq',
        'heuristics/rins/freq', 'heuristics/rounding/freq', 'heuristics/shifting/freq'
    ]
    for param in heuristic_freq_params:
        try:
            scip.setParam(param, -1)
        except:
            pass

    scip.hideOutput(True)

    # 挂载 AI 安检员
    ai_cutsel = XGBoostCutSelector(model_path)
    scip.includeCutsel(ai_cutsel, "AI_Cut_Selector", "XGBoost ML4CO Filter", priority=999999)

    start_time = time.time()
    scip.optimize()
    end_time = time.time()
    solving_time = end_time - start_time

    # 💡 完美复刻你的 1.0 打印格式，让你一目了然！
    print("\n" + "=" * 70)
    print("最终状态:")
    print(f"求解状态: {scip.getStatus()}")

    # 获取上下界时加上异常处理，防止未找到解时报错
    try:
        primal = scip.getPrimalbound()
    except:
        primal = "N/A"
    try:
        dual = scip.getDualbound()
    except:
        dual = "N/A"

    print(f"原始界: {primal}")
    print(f"对偶界: {dual}")
    print(f"对偶间隙: {scip.getGap() * 100:.1f}%")
    print(f"总节点数: {scip.getNNodes()}")
    print("=" * 70)
    print("🏆 【AI 辅助求解究极战报】")
    print("=" * 70)
    print(f"⏱️ 求解总耗时: {solving_time:.2f} 秒")
    print(f"🔍 AI 总计查验生成割: {ai_cutsel.total_inspected} 个")
    print(f"✅ AI 批准进入矩阵: {ai_cutsel.total_accepted} 个")
    if ai_cutsel.total_inspected > 0:
        reject_rate = (ai_cutsel.total_rejected / ai_cutsel.total_inspected) * 100
        print(f"🗑️ AI 冷酷拦截废弃: {ai_cutsel.total_rejected} 个 (垃圾拦截率: {reject_rate:.1f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_FILE = os.path.join(current_dir, 'xgboost_gap_elite_model.json')
    MPS_FILE = r'C:\Users\Lyouth\Desktop\code\aflow30a.mps'

    if os.path.exists(MPS_FILE) and os.path.exists(MODEL_FILE):
        run_ai_scip(MPS_FILE, MODEL_FILE)