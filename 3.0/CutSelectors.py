from pyscipopt import Model, SCIP_RESULT
from pyscipopt.scip import Cutsel
import csv
from Common import *


class MaxEfficacyCutSelector(Cutsel):
    """增强版高效能割选择器 - 完整特征记录"""

    def __init__(self, basePath=None):
        super().__init__()
        self.round_id = 0
        self.selected_data = []
        self.basePath = basePath

    # forcedcuts会占用maxnselectedcuts名额，通常质量较高，数量通常很少，forcedcuts为已经确定入选割
    # SCIP内部机制 1.数值稳定性保护：范数小但是可能会有数值问题，为了不被筛选设置被强制接受
    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        """选择效能最大的割平面，并记录完整特征"""
        model = self.model
        # self.round_id += 1
        # print(f"🎯 割选择轮次 #{self.round_id}: {len(cuts)}个候选割平面")
        if len(forcedcuts) > 0:
            print(f"出现强制割平面，数量为: {len(forcedcuts)}")
        # 1. 记录所有割平面的完整特征
        j = 0
        for i, cut in enumerate(cuts):
            j = j + 1

            # 判断是否为真正的割平面
            is_cut = (cut.getOrigintype() == 3)
            if is_cut:
                print(f"选择器：第{model.getNSepaRounds()}轮割，第{model.getCurrentNode().getNumber()}个节点，发现第{j}个割平面：{cut.name}！")
                cut_record = record_cut_features(model, cut)
                self.selected_data.append(cut_record)
        # 2. 效能排序和选择
        # sorted_cuts, nselected = self._select_cuts_by_efficacy(cuts, maxnselectedcuts)

        # return {
        #     'cuts': sorted_cuts,
        #     'nselectedcuts': nselected,
        #     'result': SCIP_RESULT.SUCCESS
        # }
        return {
            'cuts': cuts,
            'nselectedcuts': maxnselectedcuts,
            'result': SCIP_RESULT.SUCCESS
        }

    def _select_cuts_by_efficacy(self, cuts, maxnselectedcuts):
        """基于效能选择割平面"""
        try:
            model = self.model
            scored_cuts = []

            for cut in cuts:
                try:
                    efficacy = model.getCutEfficacy(cut)
                    scored_cuts.append((cut, efficacy))
                except:
                    scored_cuts.append((cut, 0.0))

            # 按效能降序排序
            scored_cuts.sort(key=lambda x: x[1], reverse=True)
            sorted_cuts = [cut for cut, efficacy in scored_cuts]

            # 动态选择数量：基于效能阈值
            nselected = self._calculate_dynamic_selection(scored_cuts, maxnselectedcuts)

            print(f"   ✅ 选择 {nselected}/{len(cuts)} 个割平面, 最高效能: {scored_cuts[0][1]:.4f}")
            return sorted_cuts, nselected

        except Exception as e:
            print(f"❌ 选择失败: {e}")
            return cuts, min(maxnselectedcuts, len(cuts))

    def _calculate_dynamic_selection(self, scored_cuts, maxnselectedcuts):
        """动态计算选择数量"""
        if not scored_cuts:
            return 0

        # 策略：选择效能超过阈值的割平面
        max_efficacy = scored_cuts[0][1]
        threshold = max(0.001, max_efficacy * 0.1)  # 至少选择效能>0.001的

        nselected = 0
        for cut, efficacy in scored_cuts:
            if efficacy >= threshold and nselected < maxnselectedcuts:
                nselected += 1
            else:
                break

        # 确保至少选择一个（如果有的话）
        if nselected == 0 and scored_cuts:
            nselected = min(1, maxnselectedcuts)

        return nselected
