# 这是PyCharm的静态类型检查问题，不是代码错误。PySCIPOpt使用运行时动态导入，所以PyCharm无法识别这些引用
import time
from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE
from Common import *


class EventHandler(Eventhdlr):
    """增强版切割追踪器 - 基于完整起源类型的割平面识别"""

    def __init__(self):
        self.separator_data = []
        self.all_row_data = []
        self.separator_lp_data = []
        self.gap_data = []
        # self.lp_solved_data = []
        # self.best_solved_found = []
        self.node_data = []
        self.start_time = None
        # 分离器统计
        self.separator_statistics = {}
        # 轮次统计
        self.turn_count = 0

    def eventinit(self):
        """注册所有可能相关的事件类型"""
        print("注册切割追踪事件...")

        # 尝试注册所有可能的切割相关事件
        event_candidates = [
            'ROWADDEDSEPA',  # 分割器添加的行：分离器生成并提议了一个切割，切割进入割池（Cut Pool），但还未加入LP（可能有用）
            'ROWADDEDLP',  # LP中添加的行：当切割被验证有效后，正式加入到线性规划松弛中（验证：1. 有效性验证 2. 数值稳定性验证 3. 深度验证）
            'ROWDELETEDSEPA',  # 分离器删除的行
            'ROWDELETEDLP',  # LP中删除的行
            'BESTSOLFOUND',  # 找到了一个更好的整数可行解
            'LPSOLVED',  # LP求解的边界改进完成
            'NODEFOCUSED',  # 节点聚焦：开始处理某个节点时触发（聚焦）
            'PRESOLVEROUND',  # 预处理轮次：SCIP在每次预处理轮次完成后触发（简化）
        ]
        # 单轮工作：LPSOLVED LP松弛解目标函数值更新->分离器工作（生成切割） → ROWADDEDSEPA（事件通知） → 割池管理（评估/存储）→ 切割选择器 （筛选最佳）→ ROWADDEDLP（正式添加）→ LP求解（影响边界）回到LPSOLVED

        for event_name in event_candidates:
            event_type = getattr(SCIP_EVENTTYPE, event_name)
            self.model.catchEvent(event_type, self)
        self.start_time = time.time()

    def eventexit(self):
        """清理事件注册"""
        events_to_drop = ['ROWADDEDSEPA', 'ROWADDEDLP', 'ROWDELETEDSEPA',
                          'ROWDELETEDLP', 'BESTSOLFOUND', 'LPSOLVED',
                          'NODEFOCUSED', 'PRESOLVEROUND']

        for event_name in events_to_drop:
            try:
                event_type = getattr(SCIP_EVENTTYPE, event_name)
                self.model.dropEvent(event_type, self)
            except:
                pass

    def eventexec(self, event):
        """处理所有事件 - 增强调试输出"""
        if self.start_time is None:
            return

        current_time = round(time.time() - self.start_time, 2)
        event_type = event.getType()

        # 处理割平面事件
        if event_type in [SCIP_EVENTTYPE.ROWADDEDSEPA]:
            self.process_separator_cut_event(event, current_time, event_type)
        elif event_type in [SCIP_EVENTTYPE.ROWADDEDLP]:
            self.process_separator_lp_cut_event(event, current_time, event_type)
        # 处理间隙相关事件
        elif event_type in [SCIP_EVENTTYPE.LPSOLVED, SCIP_EVENTTYPE.BESTSOLFOUND]:
            self.process_gap_event(current_time, event_type_names.get(event_type, f"UNKNOWN_{event_type}"))
        # 处理节点事件
        elif event_type == SCIP_EVENTTYPE.NODEFOCUSED:
            self.process_node_event(current_time)

    def process_separator_cut_event(self, event, current_time, event_type):
        """处理割平面事件 - 基于完整起源类型识别"""
        model = self.model
        row = event.getRow()
        if row is None:
            return
        else:
            cut_record = record_cut_features(model, row)
            self.all_row_data.append(cut_record)    #验证是不是启发式割导致了割生成器的割重复加入
            is_cut = (row.getOrigintype() == 3)
            if cut_record != {} and is_cut:
                print(f"ROWADDSEPA事件：第{model.getNSepaRounds()}轮割，第{model.getCurrentNode().getNumber()}个节点，发现割平面：{row.name}！")
                self.separator_data.append(cut_record)
                self.separator_statistics = update_separator_statistics(cut_record["cut_type"], cut_record, self.separator_statistics)

    def process_separator_lp_cut_event(self, event, current_time, event_type):
        """处理割平面事件 - 基于完整起源类型识别"""
        model = self.model
        row = event.getRow()
        if row is None:
            return
        else:
            cut_record = record_cut_features(model, row)
            is_cut = (row.getOrigintype() == 3)
            if cut_record != {} and is_cut:
                print(f"ROWADDLP事件：第{model.getNSepaRounds()}轮割，第{model.getCurrentNode().getNumber()}个节点，发现割平面：{row.name}！")
                self.separator_lp_data.append(cut_record)
                self.separator_statistics = update_separator_statistics(cut_record["cut_type"], cut_record, self.separator_statistics)

    def process_gap_event(self, current_time, event_name):
        """记录间隙变化"""
        primal = self.model.getPrimalbound()
        dual = self.model.getDualbound()
        gap = self.model.getGap() * 100
        gap_record = {
            "time": current_time,
            "primal_bound": round(primal, 2) if primal != float('inf') else None,
            "dual_bound": round(dual, 2) if dual != -float('inf') else None,
            "gap_percent": round(gap, 2) if gap != float('inf') else None,
            "event_type": event_name,
            "node_count": self.model.getNNodes()
        }

        self.gap_data.append(gap_record)
        # 定期输出进度
        if len(self.gap_data) % 20 == 0:
            print(f"进度: {current_time}s, 间隙: {gap:.1f}%, 节点: {self.model.getNNodes()}")

    def process_node_event(self, current_time):
        """记录节点信息 - 简洁版本"""
        try:
            node_record = {
                "time": current_time,
                "total_nodes": self.model.getNNodes(),
            }

            # 只添加确实可用的信息
            available_methods = [
                ("depth", lambda: self.model.getCurrentNode().getDepth() if self.model.getCurrentNode() else 0),
                ("lp_iterations", self.model.getNLPIterations),  # LP迭代次数，即单纯形法在求解线性规划问题时执行的底层计算迭代数量
                ("dual_bound", self.model.getDualbound),
                ("primal_bound", self.model.getPrimalbound),
            ]

            for field, method in available_methods:
                try:
                    node_record[field] = method()
                except:
                    node_record[field] = None

            self.node_data.append(node_record)

        except Exception as e:
            print(f"⚠️ 节点事件记录失败: {e}")
            # 记录最小信息
            try:
                self.node_data.append({
                    "time": current_time,
                    "total_nodes": 0,
                    "error": "record_failed"
                })
            except:
                pass

    def calculate_parallelism(self, cut):
        """计算割平面与最优解的平行度（近似）"""
        try:
            # 这里可以添加更复杂的平行度计算
            # 目前使用行范数作为简单代理
            norm = cut.getNorm()
            if norm > 1e-6:
                return 1.0 / norm  # 范数越小，平行度越高
            return 0.0
        except:
            return 0.0

    def print_separator_statistics(self):
        """打印分离器统计信息"""
        if not self.separator_statistics:
            print("❌ 没有分离器统计信息")
            return


        print("割平面分离器统计:")


        total_cuts = sum(stats['count'] for stats in self.separator_statistics.values())

        for sep_type, stats in sorted(self.separator_statistics.items(),
                                      key=lambda x: x[1]['count'], reverse=True):
            percentage = (stats['count'] / total_cuts) * 100
            avg_efficacy = stats['total_efficacy'] / stats['count'] if stats['count'] > 0 else 0

            print(f"\n{sep_type}:")
            print(f"  数量: {stats['count']} ({percentage:.1f}%)")
            print(f"  平均效能: {avg_efficacy:.4f}")
            print(f"  置信度分布 - 高: {stats['confidence_high']}, 中: {stats['confidence_medium']}, "
                  f"低: {stats['confidence_low']}, 极低: {stats['confidence_very_low']}")
