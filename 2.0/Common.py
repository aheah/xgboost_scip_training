from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE

# 创建事件类型映射
event_type_names = {
    SCIP_EVENTTYPE.ROWADDEDSEPA: "ROWADDEDSEPA",
    SCIP_EVENTTYPE.ROWADDEDLP: "ROWADDEDLP",
    SCIP_EVENTTYPE.BESTSOLFOUND: "BESTSOLFOUND",
    SCIP_EVENTTYPE.LPSOLVED: "LPSOLVED",
    SCIP_EVENTTYPE.NODEFOCUSED: "NODEFOCUSED",
    SCIP_EVENTTYPE.PRESOLVEROUND: "PRESOLVEROUND"
}

# 完整的行起源类型映射
COMPLETE_ROWORIGIN_TYPES = {
    0: "SCIP_ROWORIGINTYPE_UNSPEC",  # unspecified origin of row
    # 1: "SCIP_ROWORIGINTYPE_CONSHDLR",       # row created by a constraint handler
    2: "SCIP_ROWORIGINTYPE_CONS",  # row created by a constraint
    3: "SCIP_ROWORIGINTYPE_SEPA",  # row created by separator
    4: "SCIP_ROWORIGINTYPE_REOPT",  # row created by reoptimization
}

CUT_TYPE_MAPPING = {
    # 基本缩写
    "cmir": "Complemented MIR Cut",
    "objcmir": "Objective c-MIR Cut",
    "flowcover": "Flow Cover Cut",
    "objflowcover": "Objective Flow Cover Cut",
    "lci": "Lifted Cover Inequality",
    "cgcut": "Chvátal-Gomory Cut",
    "clique": "Clique Cut",
    "closecuts": "Close Cuts",
    "proj_cut": "Projection Cut",
    "disjunctive": "Disjunctive Cut",
    "ec": "Edge Concatenation Cut",
    "flower": "Flow Cover with Strengthening Cut",
    "gom": "Basic Gomory Cut",
    "scg": "Strong Chvátal-Gomory Cut",
    "implbd": "Implied Bound Cut",
    "interminor": "Intersection Minor Cut",
    "objrow": "Objective Row Cut",
    "lagromory": "Lagrangian Gomory Cut",
    "mcf": "Multi-Commodity Flow Cut",
    "minor": "Minor Cut",
    "mix": "Mixing Cut",
    "oddcycle": "Odd Cycle Cut",
    "rlt": "Reformulation-Linearization Technique Cut",

    # 附加常见缩写
    "cg": "Chvátal-Gomory Cut",
    "mir": "Mixed Integer Rounding Cut",
    "zerohalf": "Zero-half Cut",
    "eccuts": "Edge Concatenation Cuts",
}
# cut_template={
# # 1. 全局设置
# separating/maxrounds = 10          # 每节点最多10轮分离
# separating/maxroundsroot = 20      # 根节点最多20轮
#
# # 2. 启用/禁用特定割
# separating/gomory/enabled = TRUE
# separating/aggregation/enabled = TRUE
# separating/cmir/enabled = TRUE
# separating/disjunctive/enabled = FALSE  # 太贵，禁用
#
# # 3. 设置频率
# separating/gomory/freq = 3              # 经常调用
# separating/aggregation/freq = 10        # 适度调用
# separating/cmir/freq = 15               # 较少调用
# separating/strongcg/freq = 10
#
# # 4. 条件限制
# separating/gomory/maxdepth = 200        # 只在浅层使用
# separating/aggregation/mingain = 0.005  # 至少改进0.5%
# }

####################################### CutSelectors.py ###############################################################
def get_cut_full_name(cut_name):
    """根据割平面名称获取全称"""
    # 确保 cut_name 是字符串类型
    if not isinstance(cut_name, str):
        cut_name = str(cut_name)

    # 转为小写
    cut_name_lower = cut_name.lower()

    # 检查是否匹配映射表中的缩写
    for abbrev, full_name in CUT_TYPE_MAPPING.items():
        if abbrev in cut_name_lower:
            return full_name

    # 如果没有匹配，返回原始名称
    return f"Unknown Cut ({cut_name})"


def calculate_parallelism(cut):
    """计算割平面与最优解的平行度（近似）"""

    # 这里可以添加更复杂的平行度计算
    # 目前使用行范数作为简单代理
    norm = cut.getNorm()
    if norm > 1e-6:
        return 1.0 / norm  # 范数越小，平行度越高
    return 0.0


def analyze_cut_name(cut_name):
    """分析切割名称的模式特征"""
    features = {
        "prefix": "",
        "suffix": "",
        "has_numbers": False,
        "length": len(cut_name)
    }
    try:
        # 分析名称模式
        if cut_name:
            # 检查是否包含数字
            features["has_numbers"] = any(char.isdigit() for char in cut_name)

            # 分割前缀和后缀（基于下划线）
            parts = cut_name.split('_')
            if len(parts) >= 2:
                features["prefix"] = parts[0]
                features["suffix"] = parts[-1]
            elif len(parts) == 1:
                features["prefix"] = parts[0]

    except Exception as e:
        print(f"⚠️ 名称分析失败: {e}")

    return features


def assess_confidence(efficacy):
    """基于切割效能评估置信度"""
    if efficacy > 0.1:
        return 'high'
    elif efficacy > 0.01:
        return 'medium'
    elif efficacy > 0.001:
        return 'low'
    else:
        return 'very_low'


def extract_numerical_features(cut):
    """提取切割的数值特征"""
    features = {
        "var_count": 0,
        "non_zero_count": 0,
        "norm": 0,
        "is_local": 0,
        "parallelism": 0,
        "max_coef": 0,
        "min_coef": 0,
        "avg_coef": 0
    }

    try:
        # 获取系数信息
        cols = cut.getCols()
        coefs = cut.getVals()

        if cols and coefs:
            features["var_count"] = len(cols)

            # 过滤非零系数
            non_zero_coefs = [c for c in coefs if abs(c) > 1e-6]
            features["non_zero_count"] = len(
                non_zero_coefs)  # 非零系数数量：割平面约束中非零系数的个数，数值越小，割平面越稀疏，数值稳定性越好，但过小质量就越低，反之约束过于复杂，数值稳定性差

            if non_zero_coefs:
                features["max_coef"] = max(abs(c) for c in non_zero_coefs)
                features["min_coef"] = min(abs(c) for c in non_zero_coefs)
                features["avg_coef"] = sum(abs(c) for c in non_zero_coefs) / len(non_zero_coefs)

        # 其他数值特征
        features["row_nnz"] = cut.getNNonz(),  # 非零系数数量：割平面约束中非零系数的个数，数值越小，割平面越稀疏，数值稳定性越好，但过小质量就越低，反之约束过于复杂，数值稳定性差
        features[
            "norm"] = cut.getNorm()  # 欧几里得范数：sqrt(a₁² + a₂² + ... + aₙ²)，搭配非零系数数量nnz,高norm + 高nnz：可能数值不稳定,低norm + 低nnz：可能是无效的弱割平面
        features["is_local"] = int(cut.isLocal())  # 是否局部割平面：根节点是全局割对整个问题有效，非根节点是局部割仅限当前子树节点
        features["parallelism"] = calculate_parallelism(cut)  # 平行度：割平面与目标函数的"对齐程度"，割平面法向量与目标函数梯度方向平行 → 效果最好，如果垂直 → 可能无效

    except Exception as e:
        print(f"⚠️ 数值特征提取失败: {e}")

    return features


def record_cut_features(model, cut):
    """记录割平面的完整特征"""
    node = model.getCurrentNode()

    # 计算置信度
    efficacy = model.getCutEfficacy(cut)
    # 分析切割名称模式
    name_features = analyze_cut_name(cut.name)
    # 获取数值特征
    numerical_features = extract_numerical_features(cut)
    # 评估效能
    confidence = assess_confidence(efficacy)
    # 完整的特征记录
    cut_record = {
        # 基本信息
        "key": str(node.getNumber()) + "_" + str(node.getDepth()) + "_" + str(model.getNNodes()) + "_" + str(
            model.getNSepaRounds()) + "_" + cut.name + "_" + str(efficacy),
        "node_id": node.getNumber(),
        "node_depth": node.getDepth(),
        "node_count": model.getNNodes(),  # 节点总数
        "round_id": model.getNSepaRounds(),
        "cut_name": cut.name,
        "cut_type": get_cut_full_name(cut.name),
        "origin_type": cut.getOrigintype(),
        "origin_name": COMPLETE_ROWORIGIN_TYPES.get(cut.getOrigintype(), f"UNKNOWN_{cut.getOrigintype()}"),
        # "event_type": event_type_names.get(event_type, "UNKNOWN"),
        # 名称特征
        "name_prefix": name_features["prefix"],
        "name_suffix": name_features["suffix"],
        "name_has_numbers": name_features["has_numbers"],
        "name_length": name_features["length"],
        # 数值特征
        "cut_efficacy": efficacy,
        "cut_confidence": confidence,
        "var_count": numerical_features["var_count"],
        "non_zero_coefs": numerical_features["non_zero_count"],
        "row_norm": numerical_features["norm"],
        "is_local": numerical_features["is_local"],
        "parallelism": numerical_features["parallelism"],
        "max_coef": numerical_features["max_coef"],
        "min_coef": numerical_features["min_coef"],
        "avg_coef": numerical_features["avg_coef"],
        # 求解状态
        "timestamp": model.getSolvingTime(),
        "primal_bound": model.getPrimalbound(),
        "dual_bound": model.getDualbound(),
        "gap": model.getGap()

    }
    # 记录变量和系数样本
    cols_list = cut.getCols()
    coefs_list = cut.getVals()
    vars_list = [col.getVar() for col in cols_list] if cols_list else []
    try:
        if vars_list:
            var_names = [var.name if var.name else f"var_{var.getIndex()}" for var in vars_list[:5]]    #取变量名称前五个字母
            coef_samples = [repr(c) for c in coefs_list[:5]] # 保留原精度
            cut_record["sample_vars"] = ",".join(var_names)
            cut_record["sample_coefs"] = ",".join(coef_samples)
        else:
            cut_record["sample_vars"] = "no_vars"
            cut_record["sample_coefs"] = "no_coefs"
    except Exception as e:
        cut_record["sample_vars"] = f"error: {e}"
        cut_record["sample_coefs"] = f"error: {e}"
    return cut_record


def _calculate_mathematical_importance(self, cut):
    """自己实现：评估割平面的数学重要性"""
    # 基于可用特征计算重要性
    importance_score = 0.0

    # 1. 效能权重 (40%)
    efficacy = self.model.getCutEfficacy(cut)
    importance_score += efficacy * 0.4

    # 2. 类型权重 (30%)
    type_weight = self._get_cut_type_weight(cut.getOrigintype())
    importance_score += type_weight * 0.3

    # 3. 稀疏性权重 (20%)
    sparsity_score = self._calculate_sparsity_score(cut)
    importance_score += sparsity_score * 0.2

    # 4. 平行度权重 (10%)
    parallelism = self._calculate_parallelism(cut)
    importance_score += parallelism * 0.1

    return importance_score


def _get_cut_type_weight(self, origin_type):
    """自己实现：不同类型的重要性权重"""
    weights = {
        3: 1.5,  # SCIP_ROWORIGIN_SEPA_GOMORY
        4: 1.3,  # SCIP_ROWORIGIN_SEPA_CMIR
        7: 1.3,  # SCIP_ROWORIGIN_SEPA_MIR
        6: 1.2,  # SCIP_ROWORIGIN_SEPA_ZEROHALF
        8: 1.1,  # SCIP_ROWORIGIN_SEPA_CLIQUE
        # 默认权重
    }
    return weights.get(origin_type, 1.0)


def update_separator_statistics(cut_type, cut_info, separator_statistics):
    """更新分离器统计"""
    if cut_type not in separator_statistics:
        separator_statistics[cut_type] = {
            'count': 0,
            'total_efficacy': 0,
            'confidence_high': 0,
            'confidence_medium': 0,
            'confidence_low': 0,
            'confidence_very_low': 0
        }

    stats = separator_statistics[cut_type]
    stats['count'] += 1
    stats['total_efficacy'] += cut_info.get('cut_efficacy', 0)

    # 更新置信度统计
    confidence = cut_info.get('cut_confidence', 'very_low')
    if confidence == 'high':
        stats['confidence_high'] += 1
    elif confidence == 'medium':
        stats['confidence_medium'] += 1
    elif confidence == 'low':
        stats['confidence_low'] += 1
    else:
        stats['confidence_very_low'] += 1
    return separator_statistics


###############################################################################


def _calculate_sparsity_score(self, cut):
    """自己实现：稀疏性评分"""
    nnz = cut.getNNonz()
    total_vars = len(cut.getCols()) if cut.getCols() else 1
    sparsity_ratio = nnz / total_vars

    # 适中的稀疏性最好（既不太稠密也不太稀疏）
    if 0.1 <= sparsity_ratio <= 0.7:
        return 1.0
    else:
        return 0.5


def _select_cuts_by_importance(self, cuts, maxnselectedcuts):
    """基于综合重要性选择割平面"""
    scored_cuts = []
    for cut in cuts:
        # 基础特征
        efficacy = self.model.getCutEfficacy(cut)
        origin_type = cut.getOrigintype()
        norm = cut.getNorm()

        # 综合评分（需要强化学习实现的逻辑）
        score = self._compute_comprehensive_score(cut, efficacy, origin_type, norm)

        scored_cuts.append((cut, score))

    # 按评分排序
    scored_cuts.sort(key=lambda x: x[1], reverse=True)
    sorted_cuts = [cut for cut, score in scored_cuts]

    return sorted_cuts, min(maxnselectedcuts, len(sorted_cuts))


def _compute_comprehensive_score(self, cut, efficacy, origin_type, norm):
    """自己实现：计算综合评分"""
    score = efficacy  # 以效能为基础

    # 类型加成
    if origin_type in [3, 4, 7]:  # Gomory, CMIR, MIR
        score *= 1.2

    # 范数调整（避免过度惩罚小范数）
    if 1e-10 < norm < 1e-6:  # 小范数但合理的范围
        score *= 1.1  # 适度奖励

    # 平行度加成
    try:
        parallelism = self._calculate_parallelism(cut)
        score *= (1.0 + parallelism * 0.1)
    except:
        pass

    return score
