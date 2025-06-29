import numpy as np

def calculate_distribution(OutE_I, OutE_oppo, isMoved, MPri_i, flagnum):
    # 类型初始数量配置（序号0-11对应12种棋子）
    type_counts = np.array([
        3,  # 地雷
        1,  # 军旗
        1,  # 司令
        1,  # 军长
        2,  # 师长
        2,  # 旅长
        2,  # 团长
        2,  # 营长
        2,  # 炸弹
        3,  # 连长
        3,  # 排长
        3   # 工兵
    ])
    if flagnum != -1:
        type_counts -= 1
    remaining_counts = type_counts.copy()
    
    # 预处理所有攻击者类型（性能优化）
    k_types = [np.argmax(row) for row in MPri_i]
    
    # 第一轮：确定所有唯一类型并更新剩余数量
    determined = {}
    possible_list = []

    for j in range(25):
        possible = set(range(12))
        
        # CASE 2: 地雷只能在0-9号位置
        if j >= 10:
            possible.discard(0)
        if j == flagnum:
            possible_list.append(possible)
            continue
        # CASE 3: 炸弹不能在20-24号位置
        if 20 <= j <= 24:
            possible.discard(8)
        
        # CASE 4: 移动状态约束
        if isMoved[j]:
            possible.discard(0)
            possible.discard(1)
        
        # 处理攻击事件（优化后的逻辑）
        attackers = np.where(OutE_I[:, j] == 1)[0]
        for i in attackers:
            # CASE 5: 处理互吃事件
            if OutE_oppo[j, i] == 1:
                kt = k_types[i]
                allowed = {kt}
                if kt not in {0,1,11}:
                    allowed.add(8)
                possible &= allowed
            else:
                kt = k_types[i]
                if kt == 11:
                    possible &= {0}
                else:
                    possible -= {t for t in possible if t < kt and t not in {0,1,8}}
        
        # 处理敌方攻击事件
        victims = np.where(OutE_oppo[j, :] == 1)[0]
        for m in victims:
            mt = k_types[m]
            if mt == 0:
                possible &= {11}
            elif mt != 1:
                possible -= {t for t in possible if t > mt and t != 8}
        
        # 筛选可能类型并记录
        possible = {t for t in possible if remaining_counts[t] > 0}
        possible_list.append(possible)
        
        # 确定唯一类型
        if len(possible) == 1:
            t = possible.pop()
            determined[j] = t
            remaining_counts[t] -= 1
    
    # 第二轮：计算最终概率分布
    MPub = np.zeros((25, 12))
    for j in range(25):
        if j == flagnum:
            continue
        possible = possible_list[j]
        
        if j in determined:  # CASE 1: 已确定类型
            MPub[j][determined[j]] = 1.0
        else:
            # 根据剩余数量计算概率
            valid_counts = np.array([remaining_counts[t] if t in possible else 0 for t in range(12)])
            total = valid_counts.sum()
            if total > 0:
                MPub[j] = valid_counts / total
            else:  # 异常处理
                MPub[j] = np.ones(12) / 12
    if flagnum != -1:
        MPub[flagnum][1] = 1
    return MPub