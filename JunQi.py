import numpy as np
import torch
from enum import Enum
from collections import deque
import numpy as np
from enum import IntEnum
from collections import defaultdict

import random
class PieceType(IntEnum):
    MINE = 0      # 地雷
    FLAG = 1      # 军旗
    COMMANDER = 2 # 司令
    GENERAL = 3   # 军长
    MAJOR_GEN = 4 # 师长
    BRIGADIER = 5 # 旅长
    COLONEL = 6   # 团长
    MAJOR = 7     # 营长
    BOMB = 8      # 炸弹
    CAPTAIN = 9   # 连长
    LIEUTENANT = 10 # 排长
    ENGINEER = 11  # 工兵

class JunqiEnv:
        
    def __init__(self):
        # 棋盘参数
        self.rtype = {
            0:'雷',
            1:'旗',
            2:'司',
            3:'军',
            4:'师',
            5:'旅',
            6:'团',
            7:'营',
            8:'炸',
            9:'连',
            10:'排',
            11:'工'
        }
        self.maxhistory = 25
        self.rows = 12  # 总行数
        self.cols = 5   # 总列数
        
        # 特殊位置定义
        self.flag_positions = {  # 军旗允许位置
            0: [(0,1), (0,3)],   # 玩家0最后一行（行索引5）的中间两列
            1: [(11,1), (11,3)] # 玩家1最后一行（行索引11）
        }
        self.mine_zones = {     # 地雷允许区域
            0: [(i,j) for i in [0,1] for j in range(5)],  # 玩家0最后两行
            1: [(i,j) for i in [10,11] for j in range(5)] # 玩家1最后两行
        }
        self.bomb_restrictions = {  # 炸弹限制区域
            0: [(5,j) for j in range(5)],  # 玩家0第一行
            1: [(6,j) for j in range(5)]   # 玩家1第一行
        }
        self.base_camps = [  # 行营坐标（不可放置棋子）
            (2,1), (2,3), (4,1), (4,3), (3,2),
            (7,1), (7,3), (9,1), (9,3), (8,2)
        ]
        
        self.initial_id = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 0, 11, 0, 12],
            [13, 14, 0, 15, 16],
            [17, 0, 18, 0, 19],
            [20, 21, 22, 23, 24]
        ]
        # 棋子配置（类型: 数量）
        self.piece_config = {
            PieceType.MINE: 3,
            PieceType.FLAG: 1,
            PieceType.COMMANDER: 1,
            PieceType.GENERAL: 1,
            PieceType.MAJOR_GEN: 2,
            PieceType.BRIGADIER: 2,
            PieceType.COLONEL: 2,
            PieceType.MAJOR: 2,
            PieceType.BOMB: 2,
            PieceType.CAPTAIN: 3,
            PieceType.LIEUTENANT: 3,
            PieceType.ENGINEER: 3
        }
        self.move_history = deque(maxlen=self.maxhistory)  # 存储字典格式的移动记录
        self.prob_cache = {
            'pub': [None, None],    # 玩家0和1的缓存
        }
        self.state_cache = {}  # 状态缓存

        # 初始化状态
        self.board = np.zeros((self.rows, self.cols), dtype=int)  # 棋盘状态
        self.piece_map = {}  # 棋子映射 {位置: (玩家, 类型, 编号)}
        self.piece_counter = {0:0, 1:0}  # 棋子编号计数器
        self.step_count = 0
        self.last_attack_step = -1
        self.current_player = 0
        self.out_history = {
            0: np.zeros((25,25), dtype=np.int16),  # 玩家0的OutE_I
            1: np.zeros((25,25), dtype=np.int16)   # 玩家1的OutE_I
        }
        self.is_moved = {
            0: np.zeros(25, dtype=np.int16),
            1: np.zeros(25, dtype=np.int16)
        }
        self.move_history = deque(maxlen=self.maxhistory)
        self.rphase = {
            0: False,
            1: False
        }
        self.flagnum = {
            0: 0,
            1: 0
        }
        self.id_to_position = {0: {}, 1: {}}
        self.railway_mask = self._create_railway_mask()
        self.id_to_type = {0: {}, 1: {}}
        self.curmove = {}
        self._init_game()
    def _init_game(self):
        self.piece_map.clear()
        for player in [0, 1]:
            # 生成棋子池
            pieces = []
            for pt, count in self.piece_config.items():
                pieces += [pt] * count
            np.random.shuffle(pieces)
            
            # 分配特殊棋子
            flag = pieces.pop(pieces.index(PieceType.FLAG))
            mines = [pieces.pop(pieces.index(PieceType.MINE)) for _ in range(3)]
            bombs = [pieces.pop(pieces.index(PieceType.BOMB)) for _ in range(2)]
            
            # 获取部署区域
            deploy_area = self._get_deploy_zone(player)
            np.random.shuffle(deploy_area)
            
            # === 部署阶段 ===
            # 1. 放置军旗
            flag_pos = self._place_flag(player, deploy_area)
            
            # 2. 放置地雷
            mine_positions = self._place_mines(player, flag_pos, deploy_area)
            
            # 3. 放置炸弹
            bomb_positions = self._place_bombs(player, deploy_area, mine_positions)
            
            # 4. 放置剩余棋子
            self._place_others(player, pieces, deploy_area)


    def _create_railway_mask(self):
        """生成铁路线布尔掩码"""
        mask = np.zeros((self.rows, self.cols), dtype=bool)
        # 纵向铁路（指定列）
        mask[:, 0] = True
        mask[:, 4] = True
        # 横向铁路（指定行）
        mask[1, :] = True
        mask[5, :] = True
        mask[6, :] = True
        mask[10, :] = True
        mask[0, 0] = mask[11, 0] = mask[0, 4] = mask[11, 4] = False
        return mask

    def get_onehot_available_actions(self, crphase, player, selected_pos=None):
        """
        生成一维合法动作掩码（长度60）
        Args:
            crphase: 0-选择阶段，1-移动阶段
            player: 当前玩家 (0/1)
            selected_pos: 移动阶段的选中位置 (row,col)
        Returns:
            available_actions: (60,) 0-1数组
        """
        self.current_player = player
        if crphase == 0:
            return self._get_selection_mask(player).flatten()
        elif crphase == 1:
            self.curmove = self._get_movement_mask(player, self.index_to_pos(selected_pos)).flatten()
            return self.curmove
        else:
            return np.zeros(60, dtype=np.uint8)

    def _get_selection_mask(self, player):
        """选择阶段掩码生成"""
        mask = np.zeros((self.rows, self.cols), dtype=np.uint8)
        for pos, (p, typ, uid) in self.piece_map.items():
            if p == player and (typ not in [PieceType.MINE, PieceType.FLAG]) and (self.id_to_position[player][uid] not in [(0, 1), (0, 3), (11, 1), (11, 3)]):
                if player == 0:
                    mask[pos] = 1
                else:
                    mask[(self.rows - pos[0] - 1, self.cols - pos[1] - 1)] = 1
        return mask

    def _get_movement_mask(self, player, from_pos):
        """移动阶段掩码生成"""
        if player == 1:
            from_pos = (self.rows - from_pos[0] - 1, self.cols - from_pos[1] - 1)
        if from_pos is None or from_pos not in self.piece_map:
            return np.zeros((self.rows, self.cols), dtype=np.uint8)
        
        piece_type = self.piece_map[from_pos][1]
        mask = np.zeros((self.rows, self.cols), dtype=np.uint8)
        
        # 检查是否在大本营
        if from_pos in self.flag_positions:
            return mask
        
        # 根据棋子类型选择移动方式
        reachable = {}
        if piece_type == PieceType.ENGINEER:
            reachable = self._get_engineer_reachable(from_pos)
        elif self.railway_mask[from_pos]:
            reachable = self._get_railway_reachable(from_pos, piece_type)
        Sreachable = self._get_road_reachable(from_pos)
        for i, j in self.base_camps:
            if abs(i - from_pos[0]) < 2 and abs(j - from_pos[1]) < 2:
                Sreachable.append((i, j))
        if from_pos in self.base_camps:
            Sreachable.append((i + 1, j + 1))
            Sreachable.append((i + 1, j - 1))
            Sreachable.append((i - 1, j + 1))
            Sreachable.append((i - 1, j - 1))
        # 过滤非法目标
        for to_pos in Sreachable:
            # 行营保护规则
            if to_pos in self.base_camps and self.board[to_pos] != 0:
                continue
            if (from_pos in [(5, 1), (5, 3)] and to_pos in [(6, 1), (6, 3)]) or (to_pos in [(5, 1), (5, 3)] and from_pos in [(6, 1), (6, 3)]):
                continue
            if player == 0:
                mask[to_pos] = True
            else: 
                mask[(self.rows - to_pos[0] - 1, self.cols - to_pos[1] - 1)] = True
        for to_pos in reachable:
            # 行营保护规则
            if to_pos in self.base_camps and self.board[to_pos] != 0:
                continue
            if (from_pos in [(5, 1), (5, 3)] and to_pos in [(6, 1), (6, 3)]) or (to_pos in [(5, 1), (5, 3)] and from_pos in [(6, 1), (6, 3)]):
                continue
            if player == 0:
                mask[to_pos] = True
            else: 
                mask[(self.rows - to_pos[0] - 1, self.cols - to_pos[1] - 1)] = True
        return mask

    def _get_road_reachable(self, from_pos):
        """公路移动可达位置"""
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        reachable = []
        for dx, dy in directions:
            x = from_pos[0] + dx
            y = from_pos[1] + dy
            if 0 <= x < self.rows and 0 <= y < self.cols:
                if self.board[(x, y)] == 0:
                    reachable.append((x, y))
                elif self.piece_map[(x, y)][0] != self.current_player:
                    reachable.append((x, y))
        return reachable

    def _get_railway_reachable(self, from_pos, piece_type):
        """铁路移动可达位置（非工兵）"""
        reachable = []
        # 直线移动方向
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            step = 1
            while True:
                x = from_pos[0] + dx * step
                y = from_pos[1] + dy * step
                # 超出棋盘或不在铁路线
                if not (0 <= x < self.rows and 0 <= y < self.cols):
                    break
                if not self.railway_mask[x, y]:
                    break
                # 遇到棋子
                if self.board[x, y] != 0:
                    if self.piece_map[(x,y)][0] != self.current_player:  # 敌方棋子可攻击
                        reachable.append((x, y))
                    break
                reachable.append((x, y))
                step += 1
        return reachable

    def _get_engineer_reachable(self, from_pos):
        """工兵铁路移动（BFS实现）"""
        visited = set()
        queue = deque([from_pos])
        reachable = []
        
        while queue:
            pos = queue.popleft()
            if pos in visited:
                continue
            visited.add(pos)
            
            # 检查当前位置是否可停留
            if pos != from_pos and self.board[pos] == 0:
                reachable.append(pos)
            if self.board[pos] != 0:
                if self.piece_map[pos][0] != self.current_player:  # 敌方棋子可攻击
                    reachable.append(pos)
                continue
            # 探索四个方向
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                x = pos[0] + dx
                y = pos[1] + dy
                if 0 <= x < self.rows and 0 <= y < self.cols:
                    if self.railway_mask[x, y] and (x,y) not in visited:
                        queue.append((x, y))
        return reachable

    # 坐标转换函数
    def pos_to_index(self, pos):
        """(row,col) → 0~59"""
        return pos[0] * self.cols + pos[1]
    
    def index_to_pos(self, index):
        """0~59 → (row,col)"""
        return (index // self.cols, index % self.cols)

    def _get_deploy_zone(self, player):
        if player == 0:
            return [(i,j) for i in range(6) for j in range(5) 
                   if (i,j) not in self.base_camps]
        else:
            return [(i,j) for i in range(6,12) for j in range(5)
                   if (i,j) not in self.base_camps]

    def _place_flag(self, player, deploy_area):
        _pos = random.randint(1,len(self.flag_positions[player]))
        flag_pos = self.flag_positions[player][_pos-1]
        self._set_piece(flag_pos, player, PieceType.FLAG)
        deploy_area.remove(flag_pos)
        return flag_pos

    def _place_mines(self, player, flag_pos, deploy_area):
        valid_mine_spots = [pos for pos in deploy_area 
                           if pos in self.mine_zones[player]]
        # 放置剩余地雷
        for _ in range(3):
            pos = valid_mine_spots[random.randint(1,len(valid_mine_spots)) - 1]
            self._set_piece(pos, player, PieceType.MINE)
            valid_mine_spots.remove(pos)
            deploy_area.remove(pos)
        
        return valid_mine_spots[:3]

    def _place_bombs(self, player, deploy_area, exclude_pos):
        valid_spots = [pos for pos in deploy_area 
                      if pos not in exclude_pos 
                      and pos not in self.bomb_restrictions[player]]
        
        for _ in range(2):
            pos = valid_spots[random.randint(1,len(valid_spots)) - 1]
            self._set_piece(pos, player, PieceType.BOMB)
            valid_spots.remove(pos)
            deploy_area.remove(pos)
        return valid_spots[:2]

    def _place_others(self, player, pieces, deploy_area):
        for piece in pieces:
            while True:
                if not deploy_area:
                    raise ValueError("部署区域不足")
                pos = deploy_area.pop()
                if pos not in self.base_camps:
                    self._set_piece(pos, player, piece)
                    break

    def _set_piece(self, pos, player, piece_type):
        row, col = pos
        piece_id = 0
        if player == 0:
            piece_id = self.initial_id[row][col]
        else:
            piece_id = self.initial_id[self.rows - row - 1][self.cols - col - 1]
        self.piece_map[pos] = (player, piece_type, piece_id)
        if player == 0 :
            self.board[row][col] =  (piece_id + 1)
        else:
            self.board[row][col] = -(piece_id + 1)
        # 我方正数，敌方负数
        
        self.piece_counter[player] += 1
        if piece_type == PieceType.FLAG:
            self.flagnum[player] = piece_id
        self.id_to_position[player][piece_id] = pos
        self.id_to_type[player][piece_id] = piece_type

    def _get_adjacent(self, pos):
        i,j = pos
        return [(i+di,j+dj) for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]
               if 0<=i+di<self.rows and 0<=j+dj<self.cols]

    def validate_layout(self, player):
        # 检查军旗位置
        flags = [pos for pos, info in self.piece_map.items() 
                if info[0]==player and info[1]==PieceType.FLAG]
        if len(flags)!=1 or flags[0] not in self.flag_positions[player]:
            return False
        
        # 检查地雷
        mines = [pos for pos, info in self.piece_map.items() 
                if info[0]==player and info[1]==PieceType.MINE]
        if len(mines)!=3 or any(pos not in self.mine_zones[player] for pos in mines):
            return False
        
        # 检查炸弹
        bombs = [pos for pos, info in self.piece_map.items() 
                if info[0]==player and info[1]==PieceType.BOMB]
        if any(pos in self.bomb_restrictions[player] for pos in bombs):
            return False
            
        # 检查行营
        if any(pos in self.base_camps for pos in self.piece_map):
            return False
            
        return True

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.piece_map = {}
        self.piece_counter = {0:0, 1:0}
        self.step_count = 0
        self.last_attack_step = -1
        self.current_player = 0
        self.out_history = {
            0: np.zeros((25,25), dtype=np.int16),  # 玩家0的OutE_I
            1: np.zeros((25,25), dtype=np.int16)   # 玩家1的OutE_I
        }
        self.is_moved = {
            0: np.zeros(25, dtype=np.int16),
            1: np.zeros(25, dtype=np.int16)
        }
        self.move_history = deque(maxlen=self.maxhistory)  # 存储字典格式的移动记录
        self.prob_cache = {
            'pub': [None, None],    # 玩家0和1的缓存
        }
        self.state_cache = {}  # 状态缓存

        # 初始化状态
        self.board = np.zeros((self.rows, self.cols), dtype=int)  # 棋盘状态
        self.piece_map = {}  # 棋子映射 {位置: (玩家, 类型, 编号)}
        self.piece_counter = {0:0, 1:0}  # 棋子编号计数器
        self.step_count = 0
        self.last_attack_step = -1
        self.current_player = 0
        self.out_history = {
            0: np.zeros((25,25), dtype=np.int16),  # 玩家0的OutE_I
            1: np.zeros((25,25), dtype=np.int16)   # 玩家1的OutE_I
        }
        self.is_moved = {
            0: np.zeros(25, dtype=np.int16),
            1: np.zeros(25, dtype=np.int16)
        }
        self.move_history = deque(maxlen=self.maxhistory)
        self.rphase = {
            0: False,
            1: False
        }
        self.flagnum = {
            0: 0,
            1: 0
        }
        self.id_to_position = {0: {}, 1: {}}
        self.railway_mask = self._create_railway_mask()
        self.id_to_type = {0: {}, 1: {}}
        self._init_game()
        #assert self.validate_layout(0) and self.validate_layout(1), "非法布局"

# Step
    def Tstep(self, player, action0, action1):
        if player == 1:
            action0 = self.cols * self.rows - 1 - action0
            action1 = self.cols * self.rows - 1 - action1
        return self.step(player, (abs(self.board[self.index_to_pos(action0)]) - 1, self.index_to_pos(action1)))
    def step(self, player, action):
        """执行动作并返回(reward, done)"""
        if player != self.current_player:
            return 0.0, False
            
        u, to_pos = action
        i, j = to_pos
        reward = 0.0
        done = False
        self.current_player = player
        # ========== 1. 合法性检查 ==========
        from_pos, piece_type = self._get_piece_info(u, player)
        if not self._validate_move(u, from_pos, to_pos, player):
            return -2.0, False  # 非法移动惩罚
        
        # ========== 2. 执行移动 ==========
        is_attack = self.board[to_pos] != 0
        combat_result = None
        self.is_moved[player][u] = 1
        if is_attack:
            combat_result = self._resolve_combat(from_pos, to_pos, player)
            self.last_attack_step = self.step_count
            reward += self._calculate_combat_reward(combat_result, piece_type)
            
        # 更新棋盘状态
        self._update_board_state(u, from_pos, to_pos, player, piece_type, combat_result)
        
        # ========== 3. 计算战略奖励 ==========
        reward += self._calculate_strategic_reward(to_pos)        
        # ========== 4. 轮次切换 ==========
        self.current_player = 1 - self.current_player
        self.step_count += 1
        
        # ========== 5. 终局判断 ==========
        done = self._check_termination(1 - player)
        if done:
            reward += 5000
        if is_attack:
            self._invalidate_cache(0)
            self._invalidate_cache(1)
        self.move_history.append({
            'type': 'attack' if is_attack else 'move',
            'from': from_pos,
            'to': to_pos,
            'player': self.current_player,
            'step': self.step_count
        })
        #
        if self.step_count - self.last_attack_step > 14:
            reward -= (self.step_count - self.last_attack_step - 2) * (self.step_count - self.last_attack_step - 2)
        return reward, done
    
    def _invalidate_cache(self, opponent):
        """使对手的概率缓存失效"""
        self.prob_cache['pub'][opponent] = None
        self.state_cache = {}

    def _get_piece_info(self, u, player):
        """根据编号获取棋子信息"""
        for pos, (p, typ, uid) in self.piece_map.items():
            if p == player and uid == u:
                return pos, typ
        return None, None

    def _validate_move(self, u, from_pos, to_pos, player):
        """综合验证移动合法性"""
        # 基础存在性检查
        if from_pos is None:
            return False
            
        # 固定棋子检查
        if player != self.piece_map[from_pos][0]:
            return False
        if from_pos == to_pos:
            return False
        if player == 0:
            return self.curmove[self.pos_to_index(to_pos)]
        return self.curmove[self.rows * self.cols - self.pos_to_index(to_pos) - 1]

    def _calculate_strategic_reward(self, to_pos):
        """战略位置奖励"""
        add = to_pos[0] - 6
        if self.current_player == 1:
            add = 5 - to_pos[0]        
        if to_pos in self.base_camps:  # 行营控制
            return 0.8 + add * 0.3
        return 0.0 + add * 0.3
    
    def _resolve_combat(self, from_pos, to_pos, attacker_player):
        """战斗结果解析"""
        attacker_type = self.piece_map[from_pos][1]
        defender_type = self.piece_map[to_pos][1]
        attacker_id = self.piece_map[from_pos][2]
        defender_id = self.piece_map[to_pos][2]
        # 工兵排雷
        if attacker_type == PieceType.ENGINEER and defender_type == PieceType.MINE:
            self.out_history[attacker_player][attacker_id, defender_id] = 1
            return {'winner': 'attacker', 'attacker_survive': True}
            
        # 炸弹同归
        if attacker_type == PieceType.BOMB or defender_type == PieceType.BOMB or attacker_type == defender_type:
            self.out_history[attacker_player][attacker_id, defender_id] = 1
            self.out_history[1 - attacker_player][defender_id, attacker_id] = 1
            if attacker_type == PieceType.COMMANDER:
                self.rphase[attacker_player] = True
            if defender_type == PieceType.COMMANDER:
                self.rphase[1 - attacker_player] = True
            return {'winner': None, 'attacker_survive': False}
            
        # 常规战斗
        if attacker_type.value < defender_type.value:
            self.out_history[attacker_player][attacker_id, defender_id] = 1
            return {'winner': 'attacker', 'attacker_survive': True}
        else:
            if attacker_type == PieceType.COMMANDER:
                self.rphase[attacker_player] = True
            self.out_history[1 - attacker_player][defender_id, attacker_id] = 1
            return {'winner': 'defender', 'attacker_survive': False}

    def _calculate_combat_reward(self, result, attacker_type):
        """战斗奖励计算"""
        base_rewards = {
            'attacker': 2.0,
            'defender': -1.5,
            'draw': 0.5
        }
        reward = base_rewards.get(result['winner'], 0.0)

        # 重要目标加成

        if attacker_type == PieceType.ENGINEER and result.get('defender_type') == PieceType.MINE:
            reward += 3.0
            
        return reward

    def _check_termination(self, player):
        """终局条件判断"""
        # 检查军旗存在
        if self.step_count - self.last_attack_step > 70:
            return True
        flags = False
        for (p, typ, _) in self.piece_map.values():
            if typ == PieceType.FLAG and p == player:
                flags = True
                break
        if not flags:
            return True

        # 检查可移动性
        movable = any(
            typ not in [PieceType.MINE, PieceType.FLAG]
            for (p, typ, _) in self.piece_map.values() if p == player
        )
        if not movable:
            return True                
        return False

    def _update_board_state(self, u, from_pos, to_pos, player, piece_type, combat_result):
        """更新游戏状态"""
        # 移除原位置
        del self.piece_map[from_pos]
        self.board[from_pos] = 0
        self.id_to_position[player][u] = to_pos
        # 处理战斗结果
        if combat_result:
            defender_u = self.piece_map[to_pos][2]
            del self.piece_map[to_pos]
            self.board[to_pos] = 0
            if not combat_result['attacker_survive']:
                return  # 同归于尽
                
        # 设置新位置
        self.piece_map[to_pos] = (player, piece_type, u)
        if player == 0:
            self.board[to_pos] = u + 1
        else:
            self.board[to_pos] = -(u + 1)

# IsLegalMove
    def is_legal_move(self, from_pos, to_pos, player):
        # 基础检查
        if not self._basic_check(from_pos, to_pos, player):
            return False
        
        piece_type = self.piece_map[from_pos][1]
        
        # 特殊棋子检查
        if from_pos in self.flag_positions:
            return False
        
        if piece_type in [PieceType.MINE, PieceType.FLAG]:
            return False
        
        # 行营规则（不可攻击敌方行营）
        if to_pos in self.base_camps and self.board[to_pos] != 0:
            return False
        
        if self._normal_move(from_pos, to_pos):
            return True
        # 工兵移动规则
        if piece_type == PieceType.ENGINEER:
            return self._engineer_move(from_pos, to_pos)
        
        # 普通棋子铁路移动
        if self._is_railway(from_pos) and self._is_railway(to_pos):
            return self._railway_move(from_pos, to_pos)
        
        # 普通陆地移动
        return self._normal_move(from_pos, to_pos)

    def _basic_check(self, from_pos, to_pos, player):
        return (
            0 <= from_pos[0] < self.rows and
            0 <= from_pos[1] < self.cols and
            0 <= to_pos[0] < self.rows and
            0 <= to_pos[1] < self.cols and
            from_pos in self.piece_map and
            self.piece_map[from_pos][0] == player
        )

    def _is_railway(self, pos):
        return pos[0] in [1,5,6,10] or ((pos[1] in [0,4]) and (pos[0] > 0 and pos[0] < self.rows))

    def _engineer_move(self, from_pos, to_pos):
        if not (self._is_railway(from_pos) and self._is_railway(to_pos)):
            return False
        
        visited = set()
        queue = deque([(from_pos, None)])  # (位置, 来向)
        directions = [(-1,0), (1,0), (0,1), (0,-1)]
        
        while queue:
            pos, prev_dir = queue.popleft()
            if pos == to_pos:
                return True
            if pos in visited:
                continue
            visited.add(pos)
            
            for dx, dy in directions:
                # 禁止立即掉头
                if prev_dir and (dx, dy) == (-prev_dir[0], -prev_dir[1]):
                    continue
                    
                nx, ny = pos[0]+dx, pos[1]+dy
                next_pos = (nx, ny)
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    if self._is_railway(next_pos) and self.board[nx, ny] == 0:
                        queue.append((next_pos, (dx, dy)))
        return False

    def _railway_move(self, from_pos, to_pos):
        # 检查直线
        if from_pos[0] != to_pos[0] and from_pos[1] != to_pos[1]:
            return False
        
        # 检查路径畅通
        step = 1 if to_pos > from_pos else -1
        if from_pos[0] == to_pos[0]:  # 水平移动
            for y in range(from_pos[1]+step, to_pos[1], step):
                if self.board[from_pos[0], y] != 0:
                    return False
        else:  # 垂直移动
            for x in range(from_pos[0]+step, to_pos[0], step):
                if self.board[x, from_pos[1]] != 0:
                    return False
        return True

    def _normal_move(self, from_pos, to_pos):
        dx = abs(from_pos[0] - to_pos[0])
        dy = abs(from_pos[1] - to_pos[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)
# Extract data
    def extract_state(self, player, crphase, selection_mask):
        self.current_player = player
        """优化后的状态提取函数"""
        # 生成各组件
        state = {
            'phase': np.array([crphase], dtype = np.int16),
            'Pri_I': self._get_pri_tensor(player),
            'Pub_oppo': self._get_cached_pub_tensor(1 - player),
            'Move': self._get_compressed_move_tensor(),
            'selected': selection_mask,
            'steps_since_attack': np.array([self.step_count - self.last_attack_step], dtype=np.int16),
        }
        
        # 转换为适合PyTorch的格式并缓存
        processed = [
        state['phase'],                 # 假设是二维数组
        state['Pri_I'].flatten(),                 # 展平为1D
        state['Pub_oppo'].flatten(),
        state['Move'].flatten(),
        state['selected'].flatten(),
        state['steps_since_attack'].flatten()      # 标量也会被展平
        ]
        return np.concatenate(processed)
    def _get_pri_tensor(self, player):
        """优化私有张量生成"""
        tensor = np.zeros((12, 5, 12), dtype=np.float16)
        for pos, (p, typ, _) in self.piece_map.items():
            if p == player:
                i, j = pos
                if player == 0:
                    tensor[i, j, typ.value] = 1.0
                else:
                    tensor[self.rows - i - 1, self.cols - j - 1, typ.value] = 1.0
        return tensor

    def _get_cached_pub_tensor(self, opponent):
        """带缓存的公开概率生成"""
        if self.prob_cache['pub'][opponent] is None:
            self.prob_cache['pub'][opponent] = self._calculate_pub_distribution(opponent)
        return self.prob_cache['pub'][opponent]

    def _get_pub_tensor(self, opponent):
        """MPub转Pub的核心逻辑"""
            # 步骤1：获取MPub
        mpub = self._calculate_pub_distribution(opponent)
            
            # 步骤2：转换为Pub
        pub = np.zeros((12, 5, 12), dtype=np.float16)
        for uid, pos in self.id_to_position[opponent].items():
            if uid < 25 and pos is not None:
                if opponent == 1:
                    if board[pos] == -(uid + 1):
                        i, j = pos
                        pub[i,j] = mpub[uid]
                if opponent == 0:
                    if board[pos] == uid + 1:
                        i, j = pos
                        i = self.rows - i - 1
                        j = self.cols - j - 1
                        pub[i,j] = mpub[uid]
            
        return pub
    def _calculate_pub_distribution(self, opponent):
        """计算敌方概率分布"""
        # 调用外部计算模块
        from calcdis import calculate_distribution
        out_e_self = self.out_history[1 - opponent]
        out_e_oppo = self.out_history[opponent]
        
        # 获取MPri输入
        mpri = self._get_mpri_tensor(1 - opponent)
        
        # 计算并处理特殊规则
        posflag = -1
        if self.rphase[opponent] == True:
            posflag = self.flagnum[opponent]
        return calculate_distribution(out_e_self, out_e_oppo, 
                                      self.is_moved[opponent], mpri, posflag)

    def _get_compressed_move_tensor(self):
        """压缩移动历史张量"""
        tensor = np.zeros((12, 5, self.maxhistory), dtype=np.int8)
        for t in range(min(self.maxhistory, len(self.move_history))):
            move = self.move_history[-t-1]  # 从最新开始填充
            i_from, j_from = move['from']
            i_to, j_to = move['to']
            
            if self.current_player == 1:
                i_from = self.rows - i_from - 1
                i_to = self.rows - i_to - 1
                j_from = self.cols - j_from - 1
                j_to = self.cols - j_to - 1

            # 出发位置标记-1
            tensor[i_from, j_from, t] = -1
            
            # 目标位置标记
            tensor[i_to, j_to, t] = 1 if move['type'] == 'attack' else -2
        return tensor

    def _get_mpri_tensor(self, player):
        """编号到类型的映射"""
        tensor = np.zeros((25,12), dtype=np.float16)
        for uid, typ in self.id_to_type[player].items():
            if uid < 25:
                tensor[uid, typ.value] = 1.0
        return tensor
    
    def output(self):
        outputmatrix = [[('0', 0) for _ in range(5)] for _ in range(12)]
        for i in range(12):
            for j in range(5):
                typ_index = abs(self.board[i][j]) - 1
                if self.board[i][j] == 0:
                    outputmatrix[i][j] = ('0', 0)
                elif self.board[i][j] > 0:
                    typ_index = self.id_to_type[0][typ_index]
                    print(typ_index)
                    outputmatrix[i][j] = (self.rtype[typ_index], 1)
                else:
                    typ_index = self.id_to_type[1][typ_index]
                    print(typ_index)
                    outputmatrix[i][j] = (self.rtype[typ_index], -1)
        return outputmatrix

    def close(self):
        pass