import math
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
    0: "SCIP_ROWORIGINTYPE_UNSPEC",
    2: "SCIP_ROWORIGINTYPE_CONS",
    3: "SCIP_ROWORIGINTYPE_SEPA",
    4: "SCIP_ROWORIGINTYPE_REOPT",
}

CUT_TYPE_MAPPING = {
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
    "cg": "Chvátal-Gomory Cut",
    "mir": "Mixed Integer Rounding Cut",
    "zerohalf": "Zero-half Cut",
    "eccuts": "Edge Concatenation Cuts",
}


def get_cut_full_name(cut_name):
    if not isinstance(cut_name, str):
        cut_name = str(cut_name)
    cut_name_lower = cut_name.lower()
    for abbrev, full_name in CUT_TYPE_MAPPING.items():
        if abbrev in cut_name_lower:
            return full_name
    return f"Unknown Cut ({cut_name})"


def calculate_parallelism(cut):
    norm = cut.getNorm()
    if norm > 1e-6:
        return 1.0 / norm
    return 0.0


def analyze_cut_name(cut_name):
    features = {
        "prefix": "",
        "suffix": "",
        "has_numbers": False,
        "length": len(cut_name)
    }
    try:
        if cut_name:
            features["has_numbers"] = any(char.isdigit() for char in cut_name)
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
    if efficacy > 0.1:
        return 'high'
    elif efficacy > 0.01:
        return 'medium'
    elif efficacy > 0.001:
        return 'low'
    else:
        return 'very_low'


def extract_numerical_features(cut):
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
        cols = cut.getCols()
        coefs = cut.getVals()
        if cols and coefs:
            features["var_count"] = len(cols)
            non_zero_coefs = [c for c in coefs if abs(c) > 1e-6]
            features["non_zero_count"] = len(non_zero_coefs)
            if non_zero_coefs:
                features["max_coef"] = max(abs(c) for c in non_zero_coefs)
                features["min_coef"] = min(abs(c) for c in non_zero_coefs)
                features["avg_coef"] = sum(abs(c) for c in non_zero_coefs) / len(non_zero_coefs)

        features["row_nnz"] = cut.getNNonz(),
        features["norm"] = cut.getNorm()
        features["is_local"] = int(cut.isLocal())
        features["parallelism"] = calculate_parallelism(cut)
    except Exception as e:
        print(f"⚠️ 数值特征提取失败: {e}")
    return features


# 👉 在函数外面加一个字典，用来当“记忆缓存”
_obj_norm_cache = {}


def extract_advanced_features(model, cut, efficacy):
    """提取高阶数学特征：整数支持度 (isp)、目标平行度 (obp)、定向截断距离 (dcd)"""
    features = {"isp": 0.0, "obp": 0.0, "dcd": 0.0}
    global _obj_norm_cache  # 声明使用全局缓存

    try:
        cols = cut.getCols()
        vals = cut.getVals()
        if not cols:
            return features

        int_vars_count = 0
        dot_product = 0.0

        for col, val in zip(cols, vals):
            var = col.getVar()
            # 🐛 修复核心 Bug：使用 PySCIPOpt 真实返回的大写完整单词！
            if var.vtype() in ['INTEGER', 'BINARY', 'IMPLINT']:
                int_vars_count += 1
            dot_product += val * var.getObj()

        features["isp"] = int_vars_count / len(cols)

        # 🚀 提速核心：缓存机制！
        prob_name = model.getProbName()  # 获取当前题目的名字
        if prob_name not in _obj_norm_cache:
            _obj_norm_cache[prob_name] = sum(v.getObj() ** 2 for v in model.getVars())

        obj_norm_sq = _obj_norm_cache[prob_name]
        cut_norm = cut.getNorm()

        if obj_norm_sq > 1e-9 and cut_norm > 1e-9:
            obp = abs(dot_product) / (cut_norm * math.sqrt(obj_norm_sq))
            features["obp"] = obp
            if obp > 1e-6:
                features["dcd"] = efficacy / obp

    except Exception as e:
        print(f"⚠️ 高阶特征提取失败: {e}")

    return features


def record_cut_features(model, cut):
    """记录割平面的完整特征"""
    node = model.getCurrentNode()
    efficacy = model.getCutEfficacy(cut)
    name_features = analyze_cut_name(cut.name)
    numerical_features = extract_numerical_features(cut)
    confidence = assess_confidence(efficacy)
    # ⚡️ 核心：调用高阶特征提取引擎
    advanced_features = extract_advanced_features(model, cut, efficacy)
    cut_record = {
        "key": str(node.getNumber()) + "_" + str(node.getDepth()) + "_" + str(model.getNNodes()) + "_" + str(
            model.getNSepaRounds()) + "_" + cut.name + "_" + str(efficacy),
        "node_id": node.getNumber(),
        "node_depth": node.getDepth(),
        "node_count": model.getNNodes(),
        "round_id": model.getNSepaRounds(),
        "cut_name": cut.name,
        "cut_type": get_cut_full_name(cut.name),
        "origin_type": cut.getOrigintype(),
        "origin_name": COMPLETE_ROWORIGIN_TYPES.get(cut.getOrigintype(), f"UNKNOWN_{cut.getOrigintype()}"),
        "name_prefix": name_features["prefix"],
        "name_suffix": name_features["suffix"],
        "name_has_numbers": name_features["has_numbers"],
        "name_length": name_features["length"],
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

        # ⚡️ 核心：把新特征写入即将输出的字典
        "isp": advanced_features["isp"],
        "obp": advanced_features["obp"],
        "dcd": advanced_features["dcd"],

        "timestamp": model.getSolvingTime(),
        "primal_bound": model.getPrimalbound(),
        "dual_bound": model.getDualbound(),
        "gap": model.getGap()
    }

    cols_list = cut.getCols()
    coefs_list = cut.getVals()
    vars_list = [col.getVar() for col in cols_list] if cols_list else []
    try:
        if vars_list:
            var_names = [var.name if var.name else f"var_{var.getIndex()}" for var in vars_list[:5]]
            coef_samples = [repr(c) for c in coefs_list[:5]]
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
    importance_score = 0.0
    efficacy = self.model.getCutEfficacy(cut)
    importance_score += efficacy * 0.4
    type_weight = self._get_cut_type_weight(cut.getOrigintype())
    importance_score += type_weight * 0.3
    sparsity_score = self._calculate_sparsity_score(cut)
    importance_score += sparsity_score * 0.2
    parallelism = self._calculate_parallelism(cut)
    importance_score += parallelism * 0.1
    return importance_score


def _get_cut_type_weight(self, origin_type):
    weights = {
        3: 1.5,
        4: 1.3,
        7: 1.3,
        6: 1.2,
        8: 1.1,
    }
    return weights.get(origin_type, 1.0)


def update_separator_statistics(cut_type, cut_info, separator_statistics):
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


def _calculate_sparsity_score(self, cut):
    nnz = cut.getNNonz()
    total_vars = len(cut.getCols()) if cut.getCols() else 1
    sparsity_ratio = nnz / total_vars
    if 0.1 <= sparsity_ratio <= 0.7:
        return 1.0
    else:
        return 0.5


def _select_cuts_by_importance(self, cuts, maxnselectedcuts):
    scored_cuts = []
    for cut in cuts:
        efficacy = self.model.getCutEfficacy(cut)
        origin_type = cut.getOrigintype()
        norm = cut.getNorm()
        score = self._compute_comprehensive_score(cut, efficacy, origin_type, norm)
        scored_cuts.append((cut, score))

    scored_cuts.sort(key=lambda x: x[1], reverse=True)
    sorted_cuts = [cut for cut, score in scored_cuts]
    return sorted_cuts, min(maxnselectedcuts, len(sorted_cuts))


def _compute_comprehensive_score(self, cut, efficacy, origin_type, norm):
    score = efficacy
    if origin_type in [3, 4, 7]:
        score *= 1.2
    if 1e-10 < norm < 1e-6:
        score *= 1.1
    try:
        parallelism = self._calculate_parallelism(cut)
        score *= (1.0 + parallelism * 0.1)
    except:
        pass
    return score