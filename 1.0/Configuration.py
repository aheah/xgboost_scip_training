from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE
import time
import pandas as pd
import os
import datetime
from CutSelectors import MaxEfficacyCutSelector
from EventHandler import EventHandler

# 全局设置切割轮次
cutting_params = {
    "separating/maxrounds": 5,  # 最大切割轮次
    "separating/maxroundsroot": 10,  # 根节点切割轮次
    "separating/maxstallrounds": 3,  # 停滞轮次
    "separating/maxcuts": 10,  # 每轮最大切割数
    "separating/maxcutsroot": 5,  # 根节点最大切割数
}

# 启用各种切割生成器
cutting_plugins = {
    "separating/cgmip/freq": 10,
    "separating/gomory/freq": 1,
    "separating/clique/freq": 1,
    "separating/knapsackcover/freq": 1,
    "separating/flowcover/freq": 1,
    "separating/zerohalf/freq": 1,
    "separating/strongcg/freq": 1
}

# 列出所有已知的启发式频率参数
heuristic_freq_params = [
    'heuristics/feaspump/freq',  # 可行性泵
    'heuristics/alns/freq',  # 自适应大邻域搜索
    'heuristics/rens/freq',  # RENS
    'heuristics/locks/freq',  # Locks
    'heuristics/multistart/freq',  # 多起点
    'heuristics/undercover/freq',  # Undercover
    'heuristics/vbounds/freq',  # 虚拟界
    'heuristics/fracdiving/freq',  # 分数潜水
    'heuristics/guideddiving/freq',  # 引导潜水
    'heuristics/linesearchdiving/freq',  # 线搜索潜水
    'heuristics/coefdiving/freq',  # 系数潜水
    'heuristics/crossover/freq',  # 交叉
    'heuristics/dins/freq',  # DINS
    'heuristics/distributiondiving/freq',  # 分布潜水
    'heuristics/dualval/freq',  # 对偶值
    'heuristics/fixandinfer/freq',  # 固定与推理
    'heuristics/indicator/freq',  # 指示器
    'heuristics/intshifting/freq',  # 整数平移
    'heuristics/localbranching/freq',  # 局部分支
    'heuristics/mutation/freq',  # 变异
    'heuristics/nlpdiving/freq',  # NLP潜水
    'heuristics/octane/freq',  # 辛烷值
    'heuristics/ofins/freq',  # OFINS
    'heuristics/proximity/freq',  # 邻近
    'heuristics/pscostdiving/freq',  # 伪代价潜水
    'heuristics/randrounding/freq',  # 随机舍入
    'heuristics/rins/freq',  # RINS
    'heuristics/rootsoldiving/freq',  # 根节点解潜水
    'heuristics/rounding/freq',  # 舍入
    'heuristics/shifting/freq',  # 平移
    'heuristics/subnlp/freq',  # 子NLP
    'heuristics/trivial/freq',  # 平凡启发式
    'heuristics/trysol/freq',  # 尝试解
    'heuristics/twoopt/freq',  # 2-opt
    'heuristics/veclendiving/freq',  # 向量长度潜水
    'heuristics/zeroobj/freq',  # 零目标
    'heuristics/zirounding/freq',  # ZI舍入
]


def configure_for_p0201(model,log_path):
    print("\n配置求解参数...")
    # 基础参数
    model.setParam("limits/time", 180)
    model.setParam("limits/nodes", 10000)
    # 启用激进切割
    print("配置激进切割生成策略...")

    # 设置轮次切割参数
    for param, value in cutting_params.items():
        model.setParam(param, value)
        print(f"✓ {param} = {value}")
    # 设置启用切割生成器
    # for plugin, enabled in cutting_plugins.items():
    #     if enabled:
    #         print(f"✓ 启用 {plugin}")
    # 设置系统日志输出（高详细级别）
    model.setParam("display/verblevel", 5)
    model.setParam("display/freq", 10)
    model.setLogfile(f"{log_path}")

    # 设置关闭所有启发式算法
    print("=== 关闭所有启发式算法 ===")

    success_count = 0
    for param in heuristic_freq_params:
        try:
            model.setParam(param, -1)  # -1表示永不调用
            success_count += 1
        except:
            pass

    print(f"成功关闭了 {success_count} 个启发式算法")
    # print("设置频率为-1意味着这些启发式将永远不会被调用")

    # 分离器统计表
    model.setParam("table/separator/active", True)
