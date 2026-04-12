from Configuration import *


def cuts_record(mps_file):
    # 1. 初始化模型
    model = Model("cut_analyzer")
    basePath = r'C:\\Users\\Lyouth\\Desktop\\code\\'
    mps_path = os.path.join(basePath, mps_file)
    # 创建输出目录
    # 运行时间
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mps_name = os.path.splitext(mps_file)[0]
    data_dir = os.path.join(basePath, "data", mps_name, timestamp)
    log_dir = os.path.join(basePath, "log", mps_name, timestamp)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(mps_path):
        print(f"错误: 文件不存在 {mps_path}")
        return

    try:
        model.readProblem(mps_path)
        print(f"✓ 加载模型: {mps_file}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return

    # 2. 注册事件处理器
    eventHandler = EventHandler()
    model.includeEventhdlr(eventHandler, "EventHandler", "事件处理记录")

    # 3. 注册割选择器474
    cutSelector = MaxEfficacyCutSelector(basePath)
    model.includeCutsel(cutSelector, "MaxEfficacyCutSelector", "选择效能最大割平面并记录信息", 5000000)
    print("✓ 注册自定义割选择器")

    # 4. 配置切割参数和默认日志路径
    log_path = os.path.join(log_dir, f"{mps_file}.log")
    # 自定义参数配置（需要强化学习实现的逻辑）
    configure_for_p0201(model, log_path)
    # 5. 执行求解
    print(f"\n开始求解...")
    print("注意: 观察控制台输出，看是否有'检测到切割'的消息")
    start_time = time.time()

    try:
        model.optimize()
    except Exception as e:
        print(f"求解异常: {e}")

    total_time = time.time() - start_time

    # 6. 保存个性化结果数据分析
    print(f"\n求解完成，耗时: {total_time:.1f}s")
    # 打印分离器统计
    eventHandler.print_separator_statistics()


    if eventHandler.all_row_data:
        cut_file = os.path.join(data_dir, f"separator_all_row_data_{timestamp}.csv")
        pd.DataFrame(eventHandler.all_row_data).to_excel(cut_file, index=False)
        print(f"✓ 保存全部的约束数据: {len(eventHandler.all_row_data)} 条")
    if eventHandler.separator_data:
        cut_file = os.path.join(data_dir, f"separator_data_{timestamp}.csv")
        pd.DataFrame(eventHandler.separator_data).to_excel(cut_file, index=False)
        print(f"✓ 保存割分离器的切割数据: {len(eventHandler.separator_data)} 条")

    if cutSelector.selected_data:
        select_data_file = os.path.join(data_dir, f"separator_selector_data_{timestamp}.csv")
        pd.DataFrame(cutSelector.selected_data).to_excel(select_data_file, index=False)
        print(f"✓ 保存割选择器的切割数据: {len(eventHandler.node_data)} 条")
    if eventHandler.separator_lp_data:
        cut_file = os.path.join(data_dir, f"separator_lp_data_{timestamp}.csv")
        pd.DataFrame(eventHandler.separator_lp_data).to_excel(cut_file, index=False)
        print(f"✓ 保存加入LP的切割数据: {len(eventHandler.separator_lp_data)} 条")
    if eventHandler.separator_statistics:
        # 保存分离器统计
        stats_file = os.path.join(data_dir, f"separator_statistics_{timestamp}.csv")
        stats_data = []
        for sep_type, stats in eventHandler.separator_statistics.items():
            row = {'separator_type': sep_type}
            row.update(stats)
            stats_data.append(row)
        pd.DataFrame(stats_data).to_excel(stats_file, index=False)
        print(f"✓ 保存分离器统计: {len(stats_data)} 种分离器")

    if eventHandler.gap_data:
        gap_file = os.path.join(data_dir, f"gap_{timestamp}.csv")
        pd.DataFrame(eventHandler.gap_data).to_excel(gap_file, index=False)
        print(f"✓ 保存间隙数据: {len(eventHandler.gap_data)} 条")

    if eventHandler.node_data:
        node_file = os.path.join(data_dir, f"node_{timestamp}.csv")
        pd.DataFrame(eventHandler.node_data).to_excel(node_file, index=False)
        print(f"✓ 保存节点数据: {len(eventHandler.node_data)} 条")

    # 7. 输出求解状态
    print("最终状态:")
    try:
        status = model.getStatus()
        print(f"求解状态: {status}")
        print(f"原始界: {model.getPrimalbound()}")
        print(f"对偶界: {model.getDualbound()}")
        print(f"对偶间隙: {model.getGap() * 100:.1f}%")
        print(f"总节点数: {model.getNNodes()}")
    except:
        print("状态获取失败")



if __name__ == "__main__":
    difficult_problems = [
        # "stein9inf.mps",    #inf不可行
        # "10teams.mps",        # 秒级出结果，纯整数规划
        # "binkar10_1.mps",  #求解<10秒，背包割
        # "flugpl.mps",  # 简单航空问题
        # "p0201.mps",  # MIPLIB经典小型问题
        # "30n20b8.mps",  # 30行，20列，其中8个二进制变量，比较复杂
        "aflow30a.mps",
    ]

    for problem in difficult_problems:

        if os.path.exists(os.path.join(r'C:\\Users\\Lyouth\\Desktop\\code\\', problem)):
            print("=" * 100)
            print(f"问题模型 {problem} 开始切割统计分析")
            print("=" * 100)
            cuts_record(problem)
            print("=" * 100)
            print(f"问题模型 {problem} 结束切割统计分析")
            print("=" * 100)

# 创建model模型{读取mps问题文件}->model参数注册{启用哪些分离器、开启哪些割、各种割的节点生成频率}->model插件注册{
# 1.注册监听器（监听）
# 2.注册割分离器（按照一定算法返回一个约束方程）
# 3.注册割选择器（按照一定算法对割集合进行排序）
# 4.注册节点选择器（按照一定算法返回要访问的下一个节点）
# }->optimize()优化=循环执行{
# 1.eventinit()事件类型注册
# 2.eventexec(){事件类型处理:
# process_row_event()->     cut_record 割平面记录(分析重点是“割平面质量”，触发事件为ROWADDEDSEPA/ROWADDEDLP，是估计值，包括各种来源类型：分离器割、约束割、传播约束割、预处理割、固定变量优化化简的割等) + 效能评估 + 割统计
# process_gap_event()->     gap_data 边界更新记录/对偶间隙变化记录（分析重点是“搜索策略”的收敛速度，触发事件为LPSOLVED/BESTSOLFOUND，是实测值，包括多种多次割的求解结果）
# process_node_event()}     node_data 节点统计记录（分析重点是“搜索策略”）
# 3.eventexit()事件销退
# }
# 5.->model.separator_data,gap_data 保存输出
