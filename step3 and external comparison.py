import numpy as np
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle as MplCircle
import matplotlib.patches as mpatches
# 常量定义
MAP_WIDTH = 650
MAP_HEIGHT = 450
BASE = (600.0, 50.0)
PROTECTED_RADIUS = 100.0

INNER_TOWER_POSITIONS = [
    (500, 0), (500, 50), (500, 100), (500, 150),
    (550, 150), (600, 150), (650, 150)
]
OUTER_TOWER_POSITIONS = [
    (450, 50), (400, 50), (350, 50),
    (600, 200), (600, 250), (600, 300)
]

INNER_MAX_SHELL_SPEED = 20.0
INNER_COST_PER_SHELL = 300.0
OUTER_MAX_SHELL_SPEED = 5.0
OUTER_COST_BASE = 100.0

MAX_AMMO = 10
TOTAL_ROUNDS = 20
MIN_MISSILES_PER_ROUND = 5
MAX_MISSILES_PER_ROUND = 15
MISSILE_SPEED_MEAN = 4.0
MISSILE_SPEED_VARIANCE = 2.0
TARGET_HIT_PROB = 0.99

# 动态概率阈值范围
MIN_TARGET_PROB = 0.60      # 远距离最低要求
MAX_TARGET_PROB = 0.99      # 近距离最高要求
BASE_TARGET_PROB = 0.80     # 默认基准

UAV_MAX_SHELL_SPEED = 3.0
UAV_AMMO_CAPACITY = 5
UAV_COST_PER_SHELL_BASE = 50.0

UAV_INIT_POSITIONS = [
    (300, 100), (300, 200), (300, 300),
    (350, 150), (350, 250)
]

# 内层塔精准激活阈值
INNER_ACTIVATE_SPEED_THRESHOLD = 5
INNER_ACTIVATE_DISTANCE_THRESHOLD = 200

# 交错换弹参数
STAGGER_MIN_READY_RATIO = 0.4

# 弹药预留比例
AMMO_RESERVE_RATIO = 0.0



@dataclass
class Missile:
    missile_id: int
    spawn_point: Tuple[float, float]
    speed: int
    current_position: Tuple[float, float]
    destroyed: bool = False
    hit_base: bool = False
    total_shells_fired_at: int = 0
    resolved: bool = False

    @property
    def direction(self) -> Tuple[float, float]:
        dx = BASE[0] - self.spawn_point[0]
        dy = BASE[1] - self.spawn_point[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-8:
            return (0.0, 0.0)
        return (dx / dist, dy / dist)

    @property
    def dist_to_base(self) -> float:
        return math.hypot(BASE[0] - self.current_position[0],
                          BASE[1] - self.current_position[1])

    def get_threat_level(self) -> float:
        """综合威胁度：0~1，越高越危险"""
        d = self.dist_to_base
        d_norm = min(1.0, d / max(MAP_WIDTH, MAP_HEIGHT))
        speed_norm = min(1.0, self.speed / 10.0)
        return 0.6 * (1.0 - d_norm) + 0.4 * speed_norm

    def get_dynamic_prob_threshold(self) -> float:
        """自适应概率阈值：远→低，近→高"""
        d = self.dist_to_base
        d_norm = min(1.0, d / max(MAP_WIDTH, MAP_HEIGHT))
        proximity = 1.0 - d_norm
        speed_mod = min(1.0, self.speed / 10.0) * 0.1
        threshold = MIN_TARGET_PROB + proximity * (MAX_TARGET_PROB - MIN_TARGET_PROB) + speed_mod
        return min(MAX_TARGET_PROB, max(MIN_TARGET_PROB, threshold))


@dataclass
class Tower:
    tower_id: int
    position: Tuple[float, float]
    is_inner: bool
    ammo: int = MAX_AMMO
    reload_state: Optional[str] = None
    fired_this_cycle: bool = False
    total_shells_fired: int = 0
    total_cost: float = 0.0
    stagger_group: int = 0

    @property
    def max_shell_speed(self) -> float:
        return INNER_MAX_SHELL_SPEED if self.is_inner else OUTER_MAX_SHELL_SPEED

    @property
    def can_fire(self) -> bool:
        return self.reload_state is None and self.ammo > 0

    def get_cost_per_shell(self, shell_speed: float) -> float:
        if self.is_inner:
            return INNER_COST_PER_SHELL
        return OUTER_COST_BASE * shell_speed

    def fire(self, count: int, shell_speed: float) -> Tuple[int, float]:
        """执行发射：返回实际发射数和成本"""
        actual = min(count, self.ammo)
        cost = actual * self.get_cost_per_shell(shell_speed)
        self.ammo -= actual
        self.total_shells_fired += actual
        self.total_cost += cost
        if actual > 0:
            self.fired_this_cycle = True
        return actual, cost


@dataclass
class UAV:
    uav_id: int
    position: Tuple[float, float]
    max_shell_speed: float = UAV_MAX_SHELL_SPEED
    ammo: int = UAV_AMMO_CAPACITY
    total_shells_fired: int = 0
    total_cost: float = 0.0
    reload_state: Optional[str] = None
    fired_this_cycle: bool = False
    stagger_group: int = 0

    @property
    def can_fire(self) -> bool:
        return self.reload_state is None and self.ammo > 0

    def get_cost_per_shell(self, shell_speed: float) -> float:
        return UAV_COST_PER_SHELL_BASE * shell_speed

    def fire(self, count: int, shell_speed: float) -> Tuple[int, float]:
        """执行发射：返回实际发射数和成本"""
        actual = min(count, self.ammo)
        cost = actual * self.get_cost_per_shell(shell_speed)
        self.ammo -= actual
        self.total_shells_fired += actual
        self.total_cost += cost
        if actual > 0:
            self.fired_this_cycle = True
        return actual, cost


@dataclass
class PendingIntercept:
    missile: Missile
    cumulative_prob: float
    cost: float
    intercept_point: Tuple[float, float]


# ==================== 物理计算 ====================

def compute_min_intercept_speed(
        missile_pos: Tuple[float, float],
        missile_speed: float,
        missile_target: Tuple[float, float],
        tower_pos: Tuple[float, float],
        max_shell_speed: float,
        protected_radius: float = PROTECTED_RADIUS,
) -> Tuple[bool, float, Tuple[float, float], float]:
    S = np.array(missile_pos, dtype=float)
    B = np.array(missile_target, dtype=float)
    T = np.array(tower_pos, dtype=float)

    vec_SB = B - S
    dist_SB = float(np.linalg.norm(vec_SB))
    if dist_SB < 1e-8:
        return (False, 0.0, missile_pos, 0.0)

    u = vec_SB / dist_SB
    A = S - T
    A_sq = float(np.dot(A, A))
    A_dot_u = float(np.dot(A, u))

    max_d = dist_SB - protected_radius
    if max_d <= 0:
        return (False, 0.0, missile_pos, 0.0)

    if A_dot_u < -1e-10:
        d_opt = -A_sq / A_dot_u
        if d_opt <= 0:
            d_opt = 0.0
    else:
        d_opt = 0.0

    d_opt = max(0.0, min(d_opt, max_d))

    P = S + d_opt * u
    dist_TP = float(np.linalg.norm(P - T))

    if d_opt < 1e-8:
        v_shell = dist_TP / 0.001
        t_intercept = 0.001
    else:
        t_intercept = d_opt / missile_speed
        v_shell = dist_TP / t_intercept

    if v_shell > max_shell_speed + 1e-8:
        v_ratio = max_shell_speed / missile_speed
        a_coef = 1.0 - v_ratio * v_ratio
        b_coef = 2.0 * A_dot_u
        c_coef = A_sq

        if abs(a_coef) < 1e-10:
            if abs(b_coef) < 1e-10:
                return (False, 0.0, tuple(P.tolist()), 0.0)
            d_sol = -c_coef / b_coef
        else:
            disc = b_coef * b_coef - 4 * a_coef * c_coef
            if disc < 0:
                return (False, 0.0, tuple(P.tolist()), 0.0)
            sqrt_disc = math.sqrt(disc)
            d1 = (-b_coef + sqrt_disc) / (2 * a_coef)
            d2 = (-b_coef - sqrt_disc) / (2 * a_coef)
            sols = [d for d in (d1, d2) if d > 1e-8]
            if not sols:
                return (False, 0.0, tuple(P.tolist()), 0.0)
            d_sol = min(sols)

        if not (0 < d_sol <= max_d):
            return (False, 0.0, tuple(P.tolist()), 0.0)

        d_opt = d_sol
        P = S + d_opt * u
        dist_TP = float(np.linalg.norm(P - T))
        t_intercept = d_opt / missile_speed
        v_shell = dist_TP / t_intercept

    if v_shell > max_shell_speed + 1e-8:
        return (False, 0.0, tuple(P.tolist()), 0.0)

    dist_PB = float(np.linalg.norm(P - B))
    if dist_PB < protected_radius - 1e-8:
        return (False, 0.0, tuple(P.tolist()), 0.0)

    return (True, v_shell, tuple(P.tolist()), t_intercept)


def shells_needed(survival: float, missile_speed: int, start_num: int
                  ) -> Tuple[int, float, float]:
    """计算达到 TARGET_HIT_PROB (0.99) 所需的炮弹数。
    survival 从 1.0 开始，一直打到 survival <= 0.01 为止（即 cum_prob >= 0.99）。
    """
    n = 0
    while survival > 0.01 and n < 1000:
        sn = start_num + n + 1
        p_hit = min(sn / missile_speed, 1.0)
        survival *= (1.0 - p_hit)
        n += 1
        if p_hit >= 1.0:
            break
    return n, survival, 1.0 - survival


# ==================== 增强型 BDI Agent ====================

@dataclass
class BDIBelief:
    my_position: Tuple[float, float]
    my_ammo: int
    my_type: str
    my_max_speed: float
    visible_missiles: List[Missile] = field(default_factory=list)
    total_active_missiles: int = 0
    ally_ready_count: Dict[str, int] = field(default_factory=dict)


class FinalBDIAgent:

    def __init__(self, agent_id: int, agent_type: str,
                 position: Tuple[float, float],
                 max_speed: float, ammo: int,
                 stagger_group: int = 0):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.position = position
        self.max_speed = max_speed
        self.ammo = ammo
        self.stagger_group = stagger_group
        self.beliefs: Optional[BDIBelief] = None
        self.desires: Dict[int, float] = {}
        self.intention: Optional[int] = None
        self.intention_cum_prob: float = 0.0

    def update_beliefs(self, all_missiles: List[Missile],
                       ally_counts: Dict[str, int] = None):
        visible = [m for m in all_missiles
                   if not m.resolved and not m.destroyed and not m.hit_base]
        self.beliefs = BDIBelief(
            my_position=self.position,
            my_ammo=self.ammo,
            my_type=self.agent_type,
            my_max_speed=self.max_speed,
            visible_missiles=visible,
            total_active_missiles=len(visible),
            ally_ready_count=ally_counts or {},
        )

    def is_activated(self) -> bool:
        """
        内层塔精准激活：仅在导弹速度或距离满足阈值时才参与常规拦截，
        但只要有可见导弹就保持激活状态（通过降低期望值而非完全禁止参与）。
        """
        if self.agent_type != 'inner_tower':
            return True
        if self.ammo <= 0:
            return False
        if self.beliefs is None:
            return True  # 无信息时默认激活
        # 只要有任一导弹满足条件就激活
        for m in self.beliefs.visible_missiles:
            if m.speed >= INNER_ACTIVATE_SPEED_THRESHOLD:
                return True
            if m.dist_to_base < INNER_ACTIVATE_DISTANCE_THRESHOLD:
                return True
        # 即使不满足精准激活条件，只要有导弹就保持参与
        return len(self.beliefs.visible_missiles) > 0

    def compute_desire(self, missile: Missile,
                       min_speed: float,
                       can_intercept: bool) -> float:
        """
        分层期望值计算：
        - 综合考虑威胁度、拦截可行性、距离偏好、费效比和弹药意识
        - 不能拦截的导弹直接返回 0，避免无效投标干扰 CNP
        """
        if self.ammo <= 0:
            return 0.0

        if not can_intercept:
            return 0.0

        d_to_base = missile.dist_to_base
        # 威胁度归一化
        threat = max(0.0, 1.0 - d_to_base / max(MAP_WIDTH, MAP_HEIGHT))

        # 拦截可行性
        if self.max_speed > 0:
            feasibility = max(0.0, 1.0 - min_speed / self.max_speed)
        else:
            feasibility = 0.0

        # 动态阈值：根据导弹距离和速度自适应调整
        dyn_threshold = missile.get_dynamic_prob_threshold()
        urgency_bonus = max(0.0, 1.0 - d_to_base / (PROTECTED_RADIUS * 3))

        # 分层偏好：不同 Agent 类型采用差异化策略
        if self.agent_type == 'uav':
            range_pref = max(0.0, d_to_base / MAP_WIDTH)
            cost_factor = 1.0   # UAV 成本最低，优先承担拦截任务
            speed_penalty = max(0.0, 1.0 - (missile.speed - 3) / 10)
        elif self.agent_type == 'outer_tower':
            range_pref = 0.5
            cost_factor = 0.7
            speed_penalty = 1.0
        else:  # inner_tower
            range_pref = max(0.0, 1.0 - d_to_base / MAP_WIDTH)
            cost_factor = 0.10  # 仅在紧急情况参与，避免高成本弹药浪费
            speed_penalty = 1.0
            # 不满足精准激活条件时参与度极低
            if missile.speed < INNER_ACTIVATE_SPEED_THRESHOLD and \
               missile.dist_to_base >= INNER_ACTIVATE_DISTANCE_THRESHOLD:
                cost_factor = 0.01

        # 弹药意识：弹药越少越谨慎
        if self.agent_type == 'uav':
            ammo_cap = UAV_AMMO_CAPACITY
        else:
            ammo_cap = MAX_AMMO
        ammo_factor = max(0.1, self.ammo / ammo_cap)

        # 综合期望值（分层权重：UAV 侧重距离覆盖，内层塔侧重紧急性）
        if self.agent_type == 'uav':
            desire = (0.25 * threat + 0.25 * feasibility +
                      0.25 * range_pref + 0.15 * cost_factor +
                      0.10 * speed_penalty)
        elif self.agent_type == 'outer_tower':
            desire = (0.30 * threat + 0.25 * feasibility +
                      0.15 * range_pref + 0.20 * cost_factor +
                      0.10 * ammo_factor)
        else:  # inner_tower
            desire = (0.40 * threat + 0.20 * feasibility +
                      0.15 * range_pref + 0.10 * cost_factor +
                      0.15 * ammo_factor)

        # Intention 锁定加成：已承诺目标不轻易放弃
        if self.intention == missile.missile_id:
            desire = min(1.0, desire * 2.0)

        return max(0.001, desire)

    def evaluate_desires(self, intercept_results: Dict[int, Tuple[bool, float]]):
        self.desires = {}
        if self.beliefs is None:
            return
        for m in self.beliefs.visible_missiles:
            can_intercept, min_speed = intercept_results.get(
                m.missile_id, (False, float('inf')))
            desire = self.compute_desire(m, min_speed, can_intercept)
            self.desires[m.missile_id] = desire

    def commit_intention(self, missile_id: int, cum_prob: float = 0.0):
        self.intention = missile_id
        self.intention_cum_prob = cum_prob

    def release_intention(self):
        self.intention = None
        self.intention_cum_prob = 0.0
        self.intention_commit_cycles = 0

    def should_release_intention(self, missile: Optional[Missile]) -> bool:
        if self.intention is None:
            return True
        if missile is None or missile.resolved or missile.destroyed or missile.hit_base:
            return True
        if self.intention_cum_prob >= TARGET_HIT_PROB:
            return True
        return False


# ==================== 最终版双层 CNP ====================

@dataclass
class CNPBid:
    agent_type: str
    agent_id: int
    agent_position: Tuple[float, float]
    missile_id: int
    bid_value: float
    desire: float
    cost_per_shell: float
    min_shell_speed: float
    intercept_point: Tuple[float, float]
    can_intercept: bool
    max_shells_can_fire: int
    stagger_group: int = 0


@dataclass
class CNPContract:
    missile_id: int
    winner_type: str
    winner_id: int
    winner_position: Tuple[float, float]
    shells_assigned: int
    min_shell_speed: float
    intercept_point: Tuple[float, float]
    bid_value: float
    use_reserved: bool = False


class FinalDoubleLayerCNP:
    """最终版双层合同网：分层投标 + 接力协调 + 漏防补位"""

    def __init__(self):
        self.layer1_bids: Dict[int, List[CNPBid]] = defaultdict(list)
        self.layer2_contracts: Dict[int, List[CNPContract]] = defaultdict(list)
        self.overlap_events: int = 0
        self.leakage_events: int = 0
        self.overlap_resolved: int = 0
        self.leakage_resolved: int = 0
        self.relay_events: int = 0
        self.inner_tower_activations: int = 0

    def reset_cycle(self):
        self.layer1_bids.clear()
        self.layer2_contracts.clear()

    def layer1_collect_bids(self, bdi_agents: Dict[str, List[FinalBDIAgent]],
                            missiles: List[Missile],
                            intercept_data: Dict[Tuple[str, int, int],
                                                 Tuple[bool, float, Tuple[float, float]]]):
        self.layer1_bids.clear()

        for agent_type, agents in bdi_agents.items():
            for agent in agents:
                if agent.ammo <= 0:
                    continue

                if agent_type == 'inner_tower':
                    cost_fn = lambda spd: INNER_COST_PER_SHELL
                elif agent_type == 'outer_tower':
                    cost_fn = lambda spd: OUTER_COST_BASE * spd
                else:
                    cost_fn = lambda spd: UAV_COST_PER_SHELL_BASE * spd

                for m in missiles:
                    if m.resolved or m.destroyed or m.hit_base:
                        continue

                    key = (agent_type, agent.agent_id, m.missile_id)
                    can_intercept, min_speed, intercept_point = intercept_data.get(
                        key, (False, float('inf'), (0, 0)))

                    # 只有能拦截且有期望值的 Agent 才参与投标
                    if not can_intercept:
                        continue
                    desire = agent.desires.get(m.missile_id, 0.0)
                    if desire <= 0:
                        continue

                    cost_per_shell = cost_fn(min_speed) if can_intercept else cost_fn(agent.max_speed)
                    p_hit_single = min(1.0 / m.speed, 1.0)
                    cost_efficiency = (p_hit_single / cost_per_shell
                                       if cost_per_shell > 0 else float('inf'))

                    if agent_type == 'uav':
                        type_bonus = 0.15  # UAV 投标加成
                    elif agent_type == 'outer_tower':
                        type_bonus = 0.05
                    else:
                        type_bonus = 0.0

                    bid_value = (0.35 * desire + 0.35 * min(cost_efficiency * 100, 1.0) +
                                 0.15 * type_bonus + 0.15 * (1.0 - threat_proxy(m)))

                    bid = CNPBid(
                        agent_type=agent_type,
                        agent_id=agent.agent_id,
                        agent_position=agent.position,
                        missile_id=m.missile_id,
                        bid_value=bid_value,
                        desire=desire,
                        cost_per_shell=cost_per_shell,
                        min_shell_speed=min_speed if can_intercept else agent.max_speed,
                        intercept_point=intercept_point,
                        can_intercept=can_intercept,
                        max_shells_can_fire=agent.ammo,
                        stagger_group=agent.stagger_group,
                    )
                    self.layer1_bids[m.missile_id].append(bid)

    def layer2_relay_coordinate(self, all_missiles: List[Missile],
                                 missile_cum_probs: Dict[int, float],
                                 current_cycle: int = 0
                                 ) -> Dict[int, List[CNPContract]]:
        self.layer2_contracts.clear()

        active_missiles = [m for m in all_missiles
                           if not m.resolved and not m.destroyed and not m.hit_base]

        assigned_agents: Dict[int, Set[Tuple[str, int]]] = defaultdict(set)

        # 按距离排序
        active_missiles.sort(key=lambda m: m.dist_to_base)

        for missile in active_missiles:
            mid = missile.missile_id
            current_cum_prob = missile_cum_probs.get(mid, 0.0)

            if current_cum_prob >= TARGET_HIT_PROB:
                continue

            bids = self.layer1_bids.get(mid, [])
            if not bids:
                self.leakage_events += 1
                continue

            # 分层排序
            type_order = {'uav': 0, 'outer_tower': 1, 'inner_tower': 2}
            bids.sort(key=lambda b: (
                type_order.get(b.agent_type, 99),
                -b.bid_value
            ))

            unique_agents = set((b.agent_type, b.agent_id) for b in bids)
            if len(unique_agents) >= 2:
                self.overlap_events += 1

            survival = 1.0 - current_cum_prob
            assigned_list: List[CNPContract] = []
            shells_total = missile.total_shells_fired_at

            for bid in bids:
                if survival <= 0.01:
                    break

                # 内层塔不参与常规接力拦截
                if bid.agent_type == 'inner_tower':
                    continue

                agent_key = (bid.agent_type, bid.agent_id)
                if agent_key in assigned_agents[mid]:
                    continue

                if bid.max_shells_can_fire <= 0:
                    continue

                needed, new_survival, _ = shells_needed(survival, missile.speed, shells_total)
                if needed == 0:
                    break

                actual_fire = min(needed, bid.max_shells_can_fire)
                if actual_fire == 0:
                    continue

                temp_survival = survival
                for i in range(actual_fire):
                    sn = shells_total + i + 1
                    p_hit = min(sn / missile.speed, 1.0)
                    temp_survival *= (1.0 - p_hit)

                contract = CNPContract(
                    missile_id=mid,
                    winner_type=bid.agent_type,
                    winner_id=bid.agent_id,
                    winner_position=bid.agent_position,
                    shells_assigned=actual_fire,
                    min_shell_speed=bid.min_shell_speed,
                    intercept_point=bid.intercept_point,
                    bid_value=bid.bid_value,
                    use_reserved=False,
                )
                assigned_list.append(contract)
                assigned_agents[mid].add(agent_key)

                shells_total += actual_fire
                survival = temp_survival

                if len(assigned_list) > 1:
                    self.relay_events += 1

                cum_prob_sofar = 1.0 - survival
                if cum_prob_sofar >= TARGET_HIT_PROB:
                    break

            if assigned_list:
                self.layer2_contracts[mid] = assigned_list
                if len(assigned_list) > 1:
                    self.overlap_resolved += 1
                for c in assigned_list:
                    if c.winner_type == 'inner_tower':
                        self.inner_tower_activations += 1
            else:
                self.leakage_events += 1

        # 检查漏防 + 强制分配
        for missile in active_missiles:
            mid = missile.missile_id
            if mid not in self.layer2_contracts:
                bids = self.layer1_bids.get(mid, [])
                if bids:
                    type_order = {'uav': 0, 'outer_tower': 1, 'inner_tower': 2}
                    bids.sort(key=lambda b: (type_order.get(b.agent_type, 99), -b.bid_value))
                    best_bid = bids[0]
                    fire_count = min(best_bid.max_shells_can_fire, missile.speed)
                    if fire_count > 0:
                        contract = CNPContract(
                            missile_id=mid,
                            winner_type=best_bid.agent_type,
                            winner_id=best_bid.agent_id,
                            winner_position=best_bid.agent_position,
                            shells_assigned=fire_count,
                            min_shell_speed=best_bid.min_shell_speed,
                            intercept_point=best_bid.intercept_point,
                            bid_value=best_bid.bid_value,
                            use_reserved=True,
                        )
                        self.layer2_contracts[mid] = [contract]
                        self.leakage_resolved += 1


def threat_proxy(missile: Missile) -> float:
    return max(0.0, missile.dist_to_base / max(MAP_WIDTH, MAP_HEIGHT))


# ====================周期记录 ====================

@dataclass
class FinalCycleRecord:
    cycle_num: int
    missiles: List[Missile] = field(default_factory=list)
    emergency_ids: List[int] = field(default_factory=list)
    towers: List[Tower] = field(default_factory=list)
    uavs: List[UAV] = field(default_factory=list)
    hit_base_this_cycle: int = 0
    missiles_destroyed: int = 0
    total_cost: float = 0.0
    total_hits: int = 0
    destroyed_positions: List[Tuple[float, float]] = field(default_factory=list)
    failed_positions: List[Tuple[float, float]] = field(default_factory=list)
    overlap_events: int = 0
    leakage_events: int = 0
    overlap_resolved: int = 0
    leakage_resolved: int = 0
    relay_events: int = 0
    inner_activations: int = 0
    reserved_used: int = 0
    cnp_assignments: Dict[int, List[Tuple[str, int]]] = field(default_factory=dict)
    overlap_positions: List[Tuple[float, float]] = field(default_factory=list)
    leakage_positions: List[Tuple[float, float]] = field(default_factory=list)


# ====================仿真引擎 ====================

class FinalStep3Engine:
    """最终版完整模型：自适应 BDI + 双层接力 CNP + 交错换弹 + 多层安全兜底"""

    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)
        self.spawn_rng = random.Random(random_seed)  # 独立种子，保证导弹生成一致

        # 防御单元
        self.all_towers: List[Tower] = []
        tid = 0
        for idx, pos in enumerate(INNER_TOWER_POSITIONS):
            t = Tower(tid, pos, True)
            t.stagger_group = idx % 2
            self.all_towers.append(t)
            tid += 1
        for idx, pos in enumerate(OUTER_TOWER_POSITIONS):
            t = Tower(tid, pos, False)
            t.stagger_group = idx % 2
            self.all_towers.append(t)
            tid += 1

        self.all_uavs: List[UAV] = []
        for idx, pos in enumerate(UAV_INIT_POSITIONS):
            u = UAV(uav_id=idx, position=pos)
            u.stagger_group = idx % 2
            self.all_uavs.append(u)

        # BDI Agents
        self.bdi_agents: Dict[str, List[FinalBDIAgent]] = {
            'inner_tower': [], 'outer_tower': [], 'uav': [],
        }
        for t in self.all_towers:
            atype = 'inner_tower' if t.is_inner else 'outer_tower'
            self.bdi_agents[atype].append(
                FinalBDIAgent(t.tower_id, atype, t.position,
                              t.max_shell_speed, t.ammo,
                              t.stagger_group))
        for u in self.all_uavs:
            self.bdi_agents['uav'].append(
                FinalBDIAgent(u.uav_id, 'uav', u.position,
                              u.max_shell_speed, u.ammo,
                              u.stagger_group))

        # CNP
        self.cnp = FinalDoubleLayerCNP()

        # 导弹队列
        self.regular_queue: List[Missile] = []
        self.emergency_queue: List[Missile] = []
        self.pending_intercepts: List[PendingIntercept] = []

        # 统计
        self.total_cost = 0.0
        self.hit_counter = 0
        self.missiles_destroyed = 0
        self.missile_id_counter = 0
        self.current_cycle = 0
        self.current_round = 0
        self.cycle_records: List[FinalCycleRecord] = []

        self.total_overlap_events = 0
        self.total_leakage_events = 0
        self.total_overlap_resolved = 0
        self.total_leakage_resolved = 0
        self.total_relay_events = 0
        self.total_reserved_used = 0

        self.missile_cum_probs: Dict[int, float] = {}

        self.active_stagger_group = 0

    # ==================== 生成 ====================

    def spawn_point(self) -> Tuple[float, float]:
        if self.spawn_rng.random() < 0.5:
            return (float(self.spawn_rng.randint(0, MAP_WIDTH)), 400.0)
        return (50.0, float(self.spawn_rng.randint(0, MAP_HEIGHT)))

    def spawn_speed(self) -> int:
        s = self.spawn_rng.gauss(MISSILE_SPEED_MEAN, math.sqrt(MISSILE_SPEED_VARIANCE))
        return max(1, int(round(s)))

    def spawn_missiles(self):
        n = self.spawn_rng.randint(MIN_MISSILES_PER_ROUND, MAX_MISSILES_PER_ROUND)
        for _ in range(n):
            sp = self.spawn_point()
            m = Missile(
                missile_id=self.missile_id_counter,
                spawn_point=sp,
                speed=self.spawn_speed(),
                current_position=sp,
            )
            self.missile_id_counter += 1
            self.regular_queue.append(m)

    # ==================== 拦截计算 ====================

    def compute_intercept_for_unit(self, unit, missile: Missile,
                                    for_emergency: bool) -> Tuple[bool, float, Tuple[float, float]]:
        pr = 0.0 if for_emergency else PROTECTED_RADIUS
        ok, spd, pt, _ = compute_min_intercept_speed(
            missile.current_position, float(missile.speed), BASE,
            unit.position, unit.max_shell_speed, pr)
        return ok, spd, pt

    def compute_all_intercepts(self, missiles: List[Missile],
                                for_emergency: bool
                                ) -> Dict[Tuple[str, int, int],
                                          Tuple[bool, float, Tuple[float, float]]]:
        """计算所有防御单元对所有导弹的拦截能力。
        所有 can_fire 的 Agent 均参与物理计算，内层塔的激活控制仅在 BDI 层通过期望值实现。
        """
        data = {}
        for t in self.all_towers:
            if not t.can_fire:
                continue
            atype = 'inner_tower' if t.is_inner else 'outer_tower'
            for m in missiles:
                if m.resolved:
                    continue
                ok, spd, pt = self.compute_intercept_for_unit(t, m, for_emergency)
                data[(atype, t.tower_id, m.missile_id)] = (ok, spd, pt)

        for u in self.all_uavs:
            if not u.can_fire:
                continue
            for m in missiles:
                if m.resolved:
                    continue
                ok, spd, pt = self.compute_intercept_for_unit(u, m, for_emergency)
                data[('uav', u.uav_id, m.missile_id)] = (ok, spd, pt)

        return data

    # ==================== BDI + CNP 协同拦截 ====================

    def bdi_cnp_engage(self, missiles: List[Missile],
                        for_emergency: bool
                        ) -> Tuple[Dict[int, Tuple[float, float, Optional[Tuple[float, float]]]],
                                   Set[int]]:
        # 更新信念
        ally_counts = {}
        for atype, agents in self.bdi_agents.items():
            ready = sum(1 for a in agents if a.ammo > 0)
            ally_counts[atype] = ready

        for agents in self.bdi_agents.values():
            for agent in agents:
                agent.update_beliefs(missiles, ally_counts)

        # 收集拦截能力
        intercept_data = self.compute_all_intercepts(missiles, for_emergency)

        # 评估期望
        for atype, agents in self.bdi_agents.items():
            for agent in agents:
                agent_results = {}
                for m in missiles:
                    if m.resolved:
                        continue
                    key = (atype, agent.agent_id, m.missile_id)
                    can_intercept, min_speed, _ = intercept_data.get(
                        key, (False, float('inf'), (0, 0)))
                    agent_results[m.missile_id] = (can_intercept, min_speed)
                agent.evaluate_desires(agent_results)

        # CNP 投标 + 接力协调
        self.cnp.reset_cycle()
        self.cnp.layer1_collect_bids(self.bdi_agents, missiles, intercept_data)
        self.cnp.layer2_relay_coordinate(
            missiles, self.missile_cum_probs, self.current_cycle)
        final_contracts = dict(self.cnp.layer2_contracts)

        # 累积统计
        self.total_overlap_events += self.cnp.overlap_events
        self.total_leakage_events += self.cnp.leakage_events
        self.total_overlap_resolved += self.cnp.overlap_resolved
        self.total_leakage_resolved += self.cnp.leakage_resolved
        self.total_relay_events += self.cnp.relay_events

        # 执行拦截
        engagement_results: Dict[int, Tuple[float, float, Optional[Tuple[float, float]]]] = {}
        successfully_assigned: Set[int] = set()

        for missile_id, contracts in final_contracts.items():
            target_missile = None
            for m in missiles:
                if m.missile_id == missile_id and not m.resolved:
                    target_missile = m
                    break
            if target_missile is None:
                continue

            total_cost = 0.0
            survival = 1.0 - self.missile_cum_probs.get(missile_id, 0.0)
            shells_total = target_missile.total_shells_fired_at
            last_ipt = None

            for contract in contracts:
                if survival <= 0.01:
                    break
                unit = None
                if contract.winner_type in ('inner_tower', 'outer_tower'):
                    for t in self.all_towers:
                        atype = 'inner_tower' if t.is_inner else 'outer_tower'
                        if atype == contract.winner_type and t.tower_id == contract.winner_id:
                            unit = t
                            break
                else:
                    for u in self.all_uavs:
                        if u.uav_id == contract.winner_id:
                            unit = u
                            break

                if unit is None or not unit.can_fire:
                    continue

                needed, _, _ = shells_needed(survival, target_missile.speed, shells_total)
                actual_fire = min(needed, contract.shells_assigned, unit.ammo)
                if actual_fire == 0:
                    continue

                fired, cost = unit.fire(actual_fire, contract.min_shell_speed)
                if fired == 0:
                    continue

                if contract.use_reserved:
                    self.total_reserved_used += 1

                total_cost += cost
                for i in range(fired):
                    sn = shells_total + i + 1
                    p_hit = min(sn / target_missile.speed, 1.0)
                    survival *= (1.0 - p_hit)
                shells_total += fired
                last_ipt = contract.intercept_point

                # 更新 BDI Intention
                for atype, agents in self.bdi_agents.items():
                    for agent in agents:
                        if (agent.agent_type == contract.winner_type and
                                agent.agent_id == contract.winner_id):
                            cum_prob_sofar = 1.0 - survival
                            agent.commit_intention(missile_id, cum_prob_sofar)

            target_missile.total_shells_fired_at = shells_total
            cum_prob = 1.0 - survival
            self.missile_cum_probs[missile_id] = cum_prob

            engagement_results[missile_id] = (cum_prob, total_cost, last_ipt)
            successfully_assigned.add(missile_id)

        return engagement_results, successfully_assigned

    # ==================== 周期阶段 ====================

    def phase_resolve(self) -> Tuple[List[Tuple[float, float]],
                                     List[Tuple[float, float]]]:
        destroyed = []
        failed = []
        retry_list: List[Missile] = []

        for pi in self.pending_intercepts:
            m = pi.missile
            if m.resolved:
                continue

            self.total_cost += pi.cost
            roll = self.rng.random()

            if roll < pi.cumulative_prob:
                m.destroyed = True
                m.resolved = True
                self.missiles_destroyed += 1
                destroyed.append(pi.intercept_point)
                self.missile_cum_probs.pop(m.missile_id, None)
                for agents in self.bdi_agents.values():
                    for agent in agents:
                        if agent.intention == m.missile_id:
                            agent.release_intention()
            else:
                # 拦截判定失败
                failed.append(pi.intercept_point)
                m.current_position = pi.intercept_point
                if m.dist_to_base < PROTECTED_RADIUS:
                    # 导弹已在保护区内
                    retry_list.append(m)
                else:
                    self.emergency_queue.append(m)

        # 内层塔最后防线
        for m in retry_list:
            destroyed_by_inner = False
            for t in self.all_towers:
                if not t.is_inner or not t.can_fire:
                    continue
                ok, spd, pt = self.compute_intercept_for_unit(t, m, True)
                if not ok:
                    continue
                shells_to_fire = max(1, min(m.speed, t.ammo))
                fired, cost = t.fire(shells_to_fire, spd)
                if fired > 0:
                    self.total_cost += cost
                    m.destroyed = True
                    m.resolved = True
                    self.missiles_destroyed += 1
                    destroyed.append(pt)
                    self.missile_cum_probs.pop(m.missile_id, None)
                    for agents in self.bdi_agents.values():
                        for agent in agents:
                            if agent.intention == m.missile_id:
                                agent.release_intention()
                    destroyed_by_inner = True
                    break

            if not destroyed_by_inner:
                m.hit_base = True
                m.resolved = True
                self.hit_counter += 1
                self.missile_cum_probs.pop(m.missile_id, None)

        self.pending_intercepts.clear()
        return destroyed, failed

    def phase_emergency(self) -> int:
        hits = 0
        kept: List[Missile] = []

        emergency_missiles = [m for m in self.emergency_queue if not m.resolved]

        if emergency_missiles:
            engagement_results, assigned = self.bdi_cnp_engage(
                emergency_missiles, for_emergency=True)

            for m in emergency_missiles:
                if m.resolved:
                    continue

                if m.dist_to_base < PROTECTED_RADIUS:
                    # 导弹已进入保护区 → 内层塔最后防线拦截
                    destroyed_by_inner = False
                    for t in self.all_towers:
                        if not t.is_inner or not t.can_fire:
                            continue
                        ok, spd, pt = self.compute_intercept_for_unit(t, m, True)
                        if not ok:
                            continue
                        shells_to_fire = max(1, min(m.speed, t.ammo))
                        fired, cost = t.fire(shells_to_fire, spd)
                        if fired > 0:
                            self.total_cost += cost
                            m.destroyed = True
                            m.resolved = True
                            self.missiles_destroyed += 1
                            self.missile_cum_probs.pop(m.missile_id, None)
                            destroyed_by_inner = True
                            break
                    if not destroyed_by_inner:
                        m.hit_base = True
                        m.resolved = True
                        self.hit_counter += 1
                        hits += 1
                        self.missile_cum_probs.pop(m.missile_id, None)
                    continue

                if m.missile_id in engagement_results:
                    cum_prob, cost, ipt = engagement_results[m.missile_id]
                    if cum_prob >= TARGET_HIT_PROB:
                        self.pending_intercepts.append(PendingIntercept(
                            missile=m, cumulative_prob=cum_prob,
                            cost=cost,
                            intercept_point=ipt or m.current_position,
                        ))
                    kept.append(m)
                else:
                    # 无合同 → 内层塔兜底拦截
                    if m.dist_to_base < PROTECTED_RADIUS * 2.0:
                        destroyed_by_inner = False
                        for t in self.all_towers:
                            if not t.is_inner or not t.can_fire:
                                continue
                            ok, spd, pt = self.compute_intercept_for_unit(t, m, True)
                            if not ok:
                                continue
                            shells_to_fire = max(1, min(m.speed, t.ammo))
                            fired, cost = t.fire(shells_to_fire, spd)
                            if fired > 0:
                                self.total_cost += cost
                                m.destroyed = True
                                m.resolved = True
                                self.missiles_destroyed += 1
                                self.missile_cum_probs.pop(m.missile_id, None)
                                destroyed_by_inner = True
                                break
                        if not destroyed_by_inner:
                            if m.dist_to_base < PROTECTED_RADIUS:
                                m.hit_base = True
                                m.resolved = True
                                self.hit_counter += 1
                                hits += 1
                                self.missile_cum_probs.pop(m.missile_id, None)
                            else:
                                kept.append(m)
                    else:
                        kept.append(m)

        self.emergency_queue = kept
        return hits

    def phase_regular(self):
        regular_missiles = [m for m in self.regular_queue if not m.resolved]

        if regular_missiles:
            engagement_results, assigned = self.bdi_cnp_engage(
                regular_missiles, for_emergency=False)

            for m in regular_missiles:
                if m.resolved:
                    continue

                if m.missile_id in engagement_results:
                    cum_prob, cost, ipt = engagement_results[m.missile_id]
                    if cum_prob >= TARGET_HIT_PROB:
                        self.pending_intercepts.append(PendingIntercept(
                            missile=m, cumulative_prob=cum_prob,
                            cost=cost,
                            intercept_point=ipt or m.current_position,
                        ))
                    else:
                        self.emergency_queue.append(m)
                else:
                    self.emergency_queue.append(m)

        self.regular_queue.clear()

    def phase_move(self):
        """导弹移动：pending_intercepts 和 emergency_queue 中的导弹均按各自方向移动。"""
        for pi in self.pending_intercepts:
            m = pi.missile
            if m.resolved:
                continue
            dx, dy = m.direction
            nx = m.current_position[0] + dx * m.speed
            ny = m.current_position[1] + dy * m.speed
            m.current_position = (nx, ny)
            if m.dist_to_base < PROTECTED_RADIUS:
                m.hit_base = True
                m.resolved = True
                self.hit_counter += 1
                self.missile_cum_probs.pop(m.missile_id, None)

        # 紧急导弹也需要移动
        for m in self.emergency_queue:
            if m.resolved:
                continue
            dx, dy = m.direction
            nx = m.current_position[0] + dx * m.speed
            ny = m.current_position[1] + dy * m.speed
            m.current_position = (nx, ny)
            if m.dist_to_base < PROTECTED_RADIUS:
                m.hit_base = True
                m.resolved = True
                self.hit_counter += 1
                self.missile_cum_probs.pop(m.missile_id, None)

    def phase_reload(self):
        """
        换弹补给策略：未开火且弹药不满时自动换弹，保持火力持续性。
        弹药耗尽时进入 exhausted → reloading → ready 三阶段流程。
        结合 stagger_group 实现交错换弹，避免所有 Agent 同时进入换弹状态。
        """
        for t in self.all_towers:
            if t.reload_state == 'reloading':
                t.reload_state = None
                t.ammo = MAX_AMMO
            elif t.reload_state == 'exhausted':
                t.reload_state = 'reloading'
            elif not t.fired_this_cycle and 0 < t.ammo < MAX_AMMO:
                t.reload_state = 'reloading'
            elif t.ammo == 0 and t.reload_state is None:
                if t.fired_this_cycle:
                    t.reload_state = 'exhausted'
            t.fired_this_cycle = False

        for u in self.all_uavs:
            if u.reload_state == 'reloading':
                u.reload_state = None
                u.ammo = UAV_AMMO_CAPACITY
            elif u.reload_state == 'exhausted':
                u.reload_state = 'reloading'
            elif not u.fired_this_cycle and 0 < u.ammo < UAV_AMMO_CAPACITY:
                u.reload_state = 'reloading'
            elif u.ammo == 0 and u.reload_state is None:
                if u.fired_this_cycle:
                    u.reload_state = 'exhausted'
            u.fired_this_cycle = False

        # 切换活跃 stagger 组
        self.active_stagger_group = 1 - self.active_stagger_group

        # 同步 BDI Agent 弹药 + 释放过期 Intention + tick intention
        active_ids = {m.missile_id for m in self.active_missiles()}
        for atype, agents in self.bdi_agents.items():
            for agent in agents:
                if atype in ('inner_tower', 'outer_tower'):
                    for t in self.all_towers:
                        tatype = 'inner_tower' if t.is_inner else 'outer_tower'
                        if tatype == atype and t.tower_id == agent.agent_id:
                            agent.ammo = t.ammo
                            break
                else:
                    for u in self.all_uavs:
                        if u.uav_id == agent.agent_id:
                            agent.ammo = u.ammo
                            break

                if agent.intention is not None:
                    if agent.intention not in active_ids:
                        agent.release_intention()
                    else:
                        target_m = None
                        for m in self.active_missiles():
                            if m.missile_id == agent.intention:
                                target_m = m
                                break
                        if agent.should_release_intention(target_m):
                            agent.release_intention()

    # ==================== 主循环 ====================

    def is_done(self) -> bool:
        if self.current_round < TOTAL_ROUNDS:
            return False
        return (len(self.regular_queue) == 0 and
                len(self.emergency_queue) == 0 and
                len(self.pending_intercepts) == 0)

    def active_missiles(self) -> List[Missile]:
        result = []
        seen = set()
        for pi in self.pending_intercepts:
            if not pi.missile.resolved and pi.missile.missile_id not in seen:
                result.append(pi.missile)
                seen.add(pi.missile.missile_id)
        for m in self.emergency_queue:
            if not m.resolved and m.missile_id not in seen:
                result.append(m)
                seen.add(m.missile_id)
        return result

    def detect_overlap_leakage_positions(self):
        overlap_positions = []
        leakage_positions = []
        active_ids = {m.missile_id for m in self.active_missiles()}

        for missile_id, contracts in self.cnp.layer2_contracts.items():
            if len(contracts) > 1:
                for m in self.active_missiles():
                    if m.missile_id == missile_id:
                        overlap_positions.append(m.current_position)
                        break

        for missile_id in active_ids:
            has_bid = missile_id in self.cnp.layer1_bids and len(self.cnp.layer1_bids[missile_id]) > 0
            is_assigned = missile_id in self.cnp.layer2_contracts
            if has_bid and not is_assigned:
                for m in self.active_missiles():
                    if m.missile_id == missile_id:
                        leakage_positions.append(m.current_position)
                        break

        return overlap_positions, leakage_positions

    def run_cycle(self) -> FinalCycleRecord:
        self.current_cycle += 1

        destroyed_positions, failed_positions = self.phase_resolve()
        hits = self.phase_emergency()

        if self.current_round < TOTAL_ROUNDS:
            self.spawn_missiles()
            self.current_round += 1

        self.phase_regular()
        self.phase_move()
        self.phase_reload()

        overlap_positions, leakage_positions = self.detect_overlap_leakage_positions()

        cnp_assignments: Dict[int, List[Tuple[str, int]]] = {}
        for missile_id, contracts in self.cnp.layer2_contracts.items():
            cnp_assignments[missile_id] = [
                (c.winner_type, c.winner_id) for c in contracts
            ]

        rec = FinalCycleRecord(
            cycle_num=self.current_cycle,
            missiles=self.active_missiles(),
            emergency_ids=[m.missile_id for m in self.emergency_queue if not m.resolved],
            towers=[Tower(t.tower_id, t.position, t.is_inner,
                          t.ammo, t.reload_state, t.fired_this_cycle,
                          t.total_shells_fired, t.total_cost,
                          t.stagger_group)
                    for t in self.all_towers],
            uavs=[UAV(u.uav_id, u.position, u.max_shell_speed,
                      u.ammo, u.total_shells_fired, u.total_cost,
                      u.reload_state, u.fired_this_cycle,
                      u.stagger_group)
                  for u in self.all_uavs],
            hit_base_this_cycle=hits,
            missiles_destroyed=self.missiles_destroyed,
            total_cost=self.total_cost,
            total_hits=self.hit_counter,
            destroyed_positions=destroyed_positions,
            failed_positions=failed_positions,
            overlap_events=self.total_overlap_events,
            leakage_events=self.total_leakage_events,
            overlap_resolved=self.total_overlap_resolved,
            leakage_resolved=self.total_leakage_resolved,
            relay_events=self.total_relay_events,
            inner_activations=self.cnp.inner_tower_activations,
            reserved_used=self.total_reserved_used,
            cnp_assignments=cnp_assignments,
            overlap_positions=overlap_positions,
            leakage_positions=leakage_positions,
        )
        self.cycle_records.append(rec)
        return rec

    def run_full(self) -> Tuple[int, float, List[FinalCycleRecord]]:
        max_c = 10000
        while not self.is_done() and self.current_cycle < max_c:
            self.run_cycle()
            if self.current_cycle % 5 == 0:
                print(f"  Cycle {self.current_cycle}: hits={self.hit_counter}, "
                      f"destroyed={self.missiles_destroyed}, cost={self.total_cost:.0f}, "
                      f"overlap={self.total_overlap_events}, "
                      f"leakage={self.total_leakage_events}, "
                      f"relay={self.total_relay_events}, "
                      f"inner_act={self.cnp.inner_tower_activations}")

        print(f"\nFinal Step 3 finished: {self.current_cycle} cycles, "
              f"{self.current_round} rounds")
        return self.hit_counter, self.total_cost, self.cycle_records


# ==================== 可视化 ====================

def visualize_final(records: List[FinalCycleRecord],
                    static_towers: List[Tower],
                    static_uavs: List[UAV],
                    output_gif: Optional[str] = None):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 10))

    def draw_frame(frame_idx):
        ax.clear()
        if frame_idx >= len(records):
            return
        rec = records[frame_idx]

        ax.set_xlim(-20, MAP_WIDTH + 20)
        ax.set_ylim(-20, MAP_HEIGHT + 20)
        ax.set_aspect('equal')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(
            f'Step 3 Final: Adaptive BDI + Smart CNP  |  '
            f'Cycle {rec.cycle_num}/{len(records)}  |  '
            f'Hits: {rec.total_hits}  |  Cost: {rec.total_cost:.0f}  |  '
            f'Destroyed: {rec.missiles_destroyed}\n'
            f'Overlap: {rec.overlap_events} (resolved: {rec.overlap_resolved})  |  '
            f'Leakage: {rec.leakage_events} (resolved: {rec.leakage_resolved})  |  '
            f'Relay: {rec.relay_events}  |  '
            f'InnerAct: {rec.inner_activations}  |  Reserved: {rec.reserved_used}',
            fontsize=10, fontweight='bold'
        )

        pc = MplCircle(BASE, PROTECTED_RADIUS, fill=True, alpha=0.1,
                       edgecolor='red', linestyle='--', linewidth=1.2,
                       facecolor='pink')
        ax.add_patch(pc)
        ax.text(BASE[0], BASE[1] - PROTECTED_RADIUS - 8,
                'Protected Zone', ha='center', fontsize=7, color='red')

        ax.plot(BASE[0], BASE[1], 'r*', markersize=20,
                markeredgecolor='darkred', markeredgewidth=3, zorder=15)
        ax.text(BASE[0] + 12, BASE[1] + 12, 'BASE',
                fontsize=10, fontweight='bold', color='darkred')

        for t in static_towers:
            color = 'royalblue' if t.is_inner else 'forestgreen'
            marker = 's' if t.is_inner else '^'
            edge = 'navy' if t.is_inner else 'darkgreen'
            ax.scatter(t.position[0], t.position[1], c=color, marker=marker,
                      s=100, edgecolors=edge, linewidths=1.5, zorder=9)

        for t in rec.towers:
            if t.reload_state is not None:
                ax.plot(t.position[0], t.position[1], 'x', color='orange',
                       markersize=14, markeredgewidth=3, zorder=11)
            ax.text(t.position[0] + 4, t.position[1] + 4, str(t.ammo),
                   fontsize=6, color='black', alpha=0.8)

        for uav in rec.uavs:
            color = 'purple' if uav.can_fire else 'gray'
            ax.scatter(uav.position[0], uav.position[1], c=color, marker='D',
                      s=80, edgecolors='indigo', linewidths=1.5, zorder=10)
            if uav.reload_state is not None:
                ax.plot(uav.position[0], uav.position[1], 'x', color='orange',
                       markersize=12, markeredgewidth=2.5, zorder=11)
            ax.text(uav.position[0] + 4, uav.position[1] + 4, str(uav.ammo),
                   fontsize=6, color='white', alpha=0.9)

        relay_colors = ['cyan', 'lime', 'yellow']
        for missile_id, assignments in rec.cnp_assignments.items():
            missile_pos = None
            for m in rec.missiles:
                if m.missile_id == missile_id and not m.resolved:
                    missile_pos = m.current_position
                    break
            if missile_pos is None:
                continue

            for idx, (agent_type, agent_id) in enumerate(assignments):
                agent_pos = None
                if agent_type in ('inner_tower', 'outer_tower'):
                    for t in rec.towers:
                        tatype = 'inner_tower' if t.is_inner else 'outer_tower'
                        if tatype == agent_type and t.tower_id == agent_id:
                            agent_pos = t.position
                            break
                else:
                    for u in rec.uavs:
                        if u.uav_id == agent_id:
                            agent_pos = u.position
                            break

                if agent_pos:
                    color = relay_colors[min(idx, len(relay_colors) - 1)]
                    style = '-' if idx == 0 else '--'
                    ax.plot([agent_pos[0], missile_pos[0]],
                           [agent_pos[1], missile_pos[1]],
                           color, linestyle=style, linewidth=0.6,
                           alpha=0.5, zorder=5)

        emerg_ids = set(rec.emergency_ids)
        for m in rec.missiles:
            is_em = m.missile_id in emerg_ids
            c = 'darkorange' if is_em else 'gold'
            mk = 'X' if is_em else 'o'
            sz = 90 if is_em else 60
            ax.scatter(m.current_position[0], m.current_position[1],
                       c=c, marker=mk, s=sz, edgecolors='black',
                       linewidths=0.6, zorder=8, alpha=0.9)
            ax.plot([m.spawn_point[0], BASE[0]], [m.spawn_point[1], BASE[1]],
                    'gray', linestyle=':', linewidth=0.4, alpha=0.35)

        for pos in rec.overlap_positions:
            ax.plot(pos[0], pos[1], marker='s', color='red',
                   markersize=18, markeredgewidth=2,
                   fillstyle='none', zorder=13, alpha=0.8)

        for pos in rec.leakage_positions:
            ax.plot(pos[0], pos[1], marker='^', color='yellow',
                   markersize=18, markeredgewidth=2,
                   fillstyle='none', zorder=13, alpha=0.9)

        for pos in rec.destroyed_positions:
            ax.plot(pos[0], pos[1], 'x', color='limegreen',
                   markersize=14, markeredgewidth=3, zorder=12)
        for pos in rec.failed_positions:
            ax.plot(pos[0], pos[1], 'x', color='magenta',
                   markersize=14, markeredgewidth=3, zorder=12)

        legend_items = [
            mpatches.Patch(color='royalblue', label='Inner Tower (v<=20, cost=300)'),
            mpatches.Patch(color='forestgreen', label='Outer Tower (v<=5, cost=100v)'),
            mpatches.Patch(color='purple', label='Patrol UAV (v<=3, cost=50v)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                       markersize=8, label='Regular Missile'),
            plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='darkorange',
                       markersize=8, label='Emergency Missile'),
            plt.Line2D([0], [0], color='cyan', linewidth=1.5, linestyle='-',
                       label='CNP 1st Relay'),
            plt.Line2D([0], [0], color='lime', linewidth=1.5, linestyle='--',
                       label='CNP 2nd Relay'),
            plt.Line2D([0], [0], marker='s', color='red', markersize=8,
                       fillstyle='none', label='Fire Overlap'),
            plt.Line2D([0], [0], marker='^', color='yellow', markersize=8,
                       fillstyle='none', label='Fire Leakage'),
            plt.Line2D([0], [0], marker='x', color='limegreen', markersize=8,
                       label='Missile Destroyed'),
        ]
        ax.legend(handles=legend_items, loc='upper right',
                 fontsize=7, framealpha=0.85, ncol=1)
        ax.grid(True, alpha=0.2, linestyle='--')

    anim = FuncAnimation(fig, draw_frame, frames=len(records),
                         interval=600, repeat=False, blit=False)
    plt.tight_layout()

    if output_gif:
        print(f"Saving final animation to {output_gif}...")
        writer = PillowWriter(fps=1.5)
        anim.save(output_gif, writer=writer)
        print(f"Final animation saved to {output_gif}")
    else:
        plt.show()

    return anim


# ==================== 报告 ====================

def print_final_report(hits: int, cost: float, records: List[FinalCycleRecord]):
    print("\n" + "=" * 80)
    print("  Step 3 Final: Adaptive BDI + Smart CNP")
    print("  (自适应阈值 + 交错换弹 + 弹药预留 + 内层塔精准激活)")
    print("=" * 80)
    print(f"  Total simulation cycles : {len(records)}")
    print(f"  Base hit count          : {hits}")
    print(f"  Total system cost       : {cost:.2f}")

    if records:
        last = records[-1]
        print(f"  Missiles destroyed      : {last.missiles_destroyed}")
        total_spawned = last.missiles_destroyed + hits
        print(f"  Total spawned missiles  : {total_spawned}")
        print(f"  Cost per kill           : {cost / max(last.missiles_destroyed, 1):.1f}")
        print(f"  Intercept rate          : {last.missiles_destroyed / max(total_spawned, 1) * 100:.1f}%")

    print()
    print("  " + "-" * 78)
    print("  DEADLOCK DETECTION & RELAY STATISTICS")
    print("  " + "-" * 78)
    if records:
        last = records[-1]
        print(f"  Fire Overlap Events    : {last.overlap_events}")
        print(f"    - Resolved by CNP    : {last.overlap_resolved}")
        ov_rate = last.overlap_resolved / max(last.overlap_events, 1) * 100
        print(f"    - Resolution Rate    : {ov_rate:.1f}%")
        print(f"  Fire Leakage Events    : {last.leakage_events}")
        print(f"    - Resolved by CNP    : {last.leakage_resolved}")
        lk_rate = last.leakage_resolved / max(last.leakage_events, 1) * 100
        print(f"    - Resolution Rate    : {lk_rate:.1f}%")
        print(f"  Heterogeneous Relay    : {last.relay_events}")
        total_deadlock = last.overlap_events + last.leakage_events
        total_resolved = last.overlap_resolved + last.leakage_resolved
        print(f"  Overall Improvement    : {total_resolved / max(total_deadlock, 1) * 100:.1f}%")
    print("  " + "-" * 78)

    print()
    print("  NEW FEATURES (Final Version):")
    print("  " + "-" * 78)
    if records:
        last = records[-1]
        print(f"  Inner Tower Activations   : {last.inner_activations}")
        print(f"  Reserved Ammo Used        : {last.reserved_used}")
        print(f"  Adaptive Prob Threshold   : {MIN_TARGET_PROB:.0%} ~ {MAX_TARGET_PROB:.0%}")
    print("  " + "-" * 78)

    print()
    print("  Air Defense Tower Statistics:")
    print("  " + "-" * 78)
    if records:
        for t in records[-1].towers:
            t_type = "INNER" if t.is_inner else "OUTER"
            print(f"  ID{t.tower_id:<3} ({t.position[0]:.0f},{t.position[1]:.0f})  "
                  f"{t_type:<6}  Fired:{t.total_shells_fired:<5}  "
                  f"Cost:{t.total_cost:<12.2f}  Ammo:{t.ammo}  Grp:{t.stagger_group}")

    print()
    print("  Patrol UAV Statistics:")
    print("  " + "-" * 78)
    if records:
        for u in records[-1].uavs:
            print(f"  UAV{u.uav_id:<2} ({u.position[0]:.0f},{u.position[1]:.0f})  "
                  f"Fired:{u.total_shells_fired:<5}  "
                  f"Cost:{u.total_cost:<12.2f}  Ammo:{u.ammo}  Grp:{u.stagger_group}")
    print("=" * 80)


# ==================== 对比汇总 ====================

def run_comparison():
    """运行所有版本并输出对比"""
    import importlib.util
    print("\n" + "=" * 70)
    print("  RUNNING ALL VERSIONS FOR COMPARISON")
    print("=" * 70)

    from p1and2 import (
        SimulationEngine, Step2SimulationEngine,
        compute_min_intercept_speed, shells_needed,
        BASE, PROTECTED_RADIUS, visualize_step2,
    )

    results = {}

    # Step 1
    print("\n--- Step 1: Baseline (Towers only, Bid-based) ---")
    eng1 = SimulationEngine(random_seed=42)
    h1, c1, r1 = eng1.run_full()
    results['Step1'] = (h1, c1, len(r1), r1[-1].missiles_destroyed)

    # Step 2
    print("\n--- Step 2: +UAV, Nearest-First ---")
    eng2 = Step2SimulationEngine(random_seed=42)
    h2, c2, r2 = eng2.run_full()
    results['Step2'] = (h2, c2, len(r2), r2[-1].missiles_destroyed)

    # Greedy baseline (classic)
    print("\n--- Greedy Baseline (Classic) ---")

    class GreedySimulationEngine(Step2SimulationEngine):
        """Greedy: single best interceptor by p_hit / cost_per_shell."""

        def engage(self, missile, for_emergency: bool
                   ) -> Tuple[float, float, Optional[Tuple[float, float]]]:
            interceptors = []

            # Towers
            for tower in self.all_towers:
                if tower.can_fire:
                    res = self.compute_intercept(tower, missile, for_emergency)
                    if res.can_intercept:
                        cost_per_shell = tower.get_cost_per_shell(res.min_shell_speed)
                        p_hit = min(1.0 / missile.speed, 1.0)
                        score = p_hit / cost_per_shell if cost_per_shell > 0 else 0.0
                        interceptors.append({
                            'unit': tower,
                            'score': score,
                            'min_speed': res.min_shell_speed,
                            'intercept_point': res.intercept_point
                        })

            # UAVs
            for uav in self.all_uavs:
                if uav.can_fire:
                    protected_r = 0.0 if for_emergency else PROTECTED_RADIUS
                    ok, spd, pt, _ = compute_min_intercept_speed(
                        missile.current_position, float(missile.speed), BASE,
                        uav.position, uav.max_shell_speed, protected_r
                    )
                    if ok:
                        cost_per_shell = uav.get_cost_per_shell(spd)
                        p_hit = min(1.0 / missile.speed, 1.0)
                        score = p_hit / cost_per_shell if cost_per_shell > 0 else 0.0
                        interceptors.append({
                            'unit': uav,
                            'score': score,
                            'min_speed': spd,
                            'intercept_point': pt
                        })

            if not interceptors:
                return 0.0, 0.0, None

            interceptors.sort(key=lambda x: x['score'], reverse=True)
            chosen = next((i for i in interceptors if i['unit'].can_fire), None)
            if chosen is None:
                return 0.0, 0.0, None

            unit = chosen['unit']
            shell_speed = chosen['min_speed']
            intercept_point = chosen['intercept_point']

            survival = 1.0
            shells_total = missile.total_shells_fired_at
            total_cost = 0.0

            needed, _, _ = shells_needed(survival, missile.speed, shells_total)
            if needed == 0:
                return 0.0, 0.0, intercept_point

            fired, cost = unit.fire(needed, shell_speed)
            if fired == 0:
                return 0.0, 0.0, intercept_point

            total_cost += cost
            for i in range(fired):
                sn = shells_total + i + 1
                p_hit = min(sn / missile.speed, 1.0)
                survival *= (1.0 - p_hit)
            shells_total += fired
            missile.total_shells_fired_at = shells_total

            return 1.0 - survival, total_cost, intercept_point

    eng_g = GreedySimulationEngine(random_seed=42)
    h_g, c_g, r_g = eng_g.run_full()
    g_last = r_g[-1]
    results['Greedy'] = (h_g, c_g, len(r_g), g_last.missiles_destroyed)
    print("\nGenerating greedy animation (saving to simulation_greedy_output.gif)...")
    visualize_step2(r_g, eng_g.all_towers, eng_g.all_uavs,
                    output_gif="simulation_greedy_output.gif")

    print("\n--- Step 3: +BDI+CNP (Original)---")
    spec_o = importlib.util.spec_from_file_location('step3_orig', '3.py')
    mod_o = importlib.util.module_from_spec(spec_o)
    spec_o.loader.exec_module(mod_o)
    eng3_orig = mod_o.Step3SimulationEngine(random_seed=42)
    h3o, c3o, r3o = eng3_orig.run_full()
    o3 = r3o[-1]
    results['Step3_Orig'] = (h3o, c3o, len(r3o), o3.missiles_destroyed,
                              o3.overlap_events, o3.overlap_resolved,
                              o3.leakage_events, o3.leakage_resolved)

    print("\n--- Step 3 Optimized: Enhanced BDI + Relay CNP ---")
    spec_op = importlib.util.spec_from_file_location('step3_opt', '3_optimized.py')
    mod_op = importlib.util.module_from_spec(spec_op)
    spec_op.loader.exec_module(mod_op)
    eng3_opt = mod_op.OptimizedStep3Engine(random_seed=42)
    h3op, c3op, r3op = eng3_opt.run_full()
    o3o = r3op[-1]
    results['Step3_Opt'] = (h3op, c3op, len(r3op), o3o.missiles_destroyed,
                             o3o.overlap_events, o3o.overlap_resolved,
                             o3o.leakage_events, o3o.leakage_resolved,
                             o3o.relay_events)

    
    print("\n--- Step 3 Final: Adaptive BDI + Smart CNP ---")
    eng3_final = FinalStep3Engine(random_seed=42)
    h3f, c3f, r3f = eng3_final.run_full()
    o3f = r3f[-1]
    results['Step3_Final'] = (h3f, c3f, len(r3f), o3f.missiles_destroyed,
                               o3f.overlap_events, o3f.overlap_resolved,
                               o3f.leakage_events, o3f.leakage_resolved,
                               o3f.relay_events, o3f.inner_activations,
                               o3f.reserved_used)

    # Comparison chart (line plot)
    labels = ["Step1", "Step2", "Greedy", "Step3 Final"]
    series = [results['Step1'], results['Step2'], results['Greedy'], results['Step3_Final']]
    cost_per_kill = [s[1] / max(s[3], 1) for s in series]
    total_cost = [s[1] for s in series]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    ax.plot(labels, cost_per_kill, marker='o', linewidth=2, color='#1f77b4', label='Cost per Kill')
    ax.set_ylabel('Cost per Kill')
    ax.set_xlabel('Method')
    ax.set_title('Cost Efficiency Comparison')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(labels, total_cost, marker='s', linewidth=2, color='#d62728', label='Total Cost')
    ax2.set_ylabel('Total Cost')
    fig.tight_layout()
    fig.savefig('compare_methods.png', dpi=300)
    plt.close(fig)

    # 打印对比表
    print("\n\n" + "=" * 100)
    print("  FINAL COMPARISON: ALL FIVE VERSIONS")
    print("=" * 100)
    header = f"  {'Metric':<32} {'Step1':>10} {'Step2':>10} {'S3_Orig':>10} {'S3_Opt':>10} {'S3_Final':>10}"
    print(header)
    print("  " + "-" * 98)

    s1 = results['Step1']
    s2 = results['Step2']
    s3 = results['Step3_Orig']
    s4 = results['Step3_Opt']
    s5 = results['Step3_Final']

    rows = [
        ("Base Hits", s1[0], s2[0], s3[0], s4[0], s5[0]),
        ("Missiles Destroyed", s1[3], s2[3], s3[3], s4[3], s5[3]),
        ("Total Spawned", s1[0]+s1[3], s2[0]+s2[3], s3[0]+s3[3], s4[0]+s4[3], s5[0]+s5[3]),
        ("Total Cost", f"{s1[1]:.0f}", f"{s2[1]:.0f}", f"{s3[1]:.0f}", f"{s4[1]:.0f}", f"{s5[1]:.0f}"),
        ("Total Cycles", s1[2], s2[2], s3[2], s4[2], s5[2]),
        ("Cost per Kill", f"{s1[1]/max(s1[3],1):.0f}", f"{s2[1]/max(s2[3],1):.0f}",
         f"{s3[1]/max(s3[3],1):.0f}", f"{s4[1]/max(s4[3],1):.0f}",
         f"{s5[1]/max(s5[3],1):.0f}"),
    ]

    for name, v1, v2, v3, v4, v5 in rows:
        if isinstance(v1, float):
            v1 = f"{v1:.0f}"
        if isinstance(v2, float):
            v2 = f"{v2:.0f}"
        if isinstance(v3, float):
            v3 = f"{v3:.0f}"
        if isinstance(v4, float):
            v4 = f"{v4:.0f}"
        if isinstance(v5, float):
            v5 = f"{v5:.0f}"
        print(f"  {name:<32} {str(v1):>10} {str(v2):>10} {str(v3):>10} {str(v4):>10} {str(v5):>10}")

    print()
    print("  " + "-" * 98)
    print("  DEADLOCK & OPTIMIZATION METRICS")
    print("  " + "-" * 98)

    deadlock_rows = [
        ("Overlap Events", "--", "--", s3[4], s4[4], s5[4]),
        ("Overlap Resolved", "--", "--", s3[5], s4[5], s5[5]),
        ("Leakage Events", "--", "--", s3[6], s4[6], s5[6]),
        ("Leakage Resolved", "--", "--", s3[7], s4[7], s5[7]),
        ("Relay Events", "--", "--", "--", s4[8], s5[8]),
        ("Inner Tower Act.", "--", "--", "--", "--", s5[9]),
        ("Reserved Used", "--", "--", "--", "--", s5[10]),
    ]

    for name, v1, v2, v3, v4, v5 in deadlock_rows:
        print(f"  {name:<32} {str(v1):>10} {str(v2):>10} {str(v3):>10} {str(v4):>10} {str(v5):>10}")

    print()
    print("  " + "-" * 98)
    print("  INTERCEPT RATE & EFFICIENCY")
    print("  " + "-" * 98)
    for label, val1, val2, val3, val4, val5 in [
        ("Intercept Rate (%)",
         f"{s1[3]/max(s1[0]+s1[3],1)*100:.1f}",
         f"{s2[3]/max(s2[0]+s2[3],1)*100:.1f}",
         f"{s3[3]/max(s3[0]+s3[3],1)*100:.1f}",
         f"{s4[3]/max(s4[0]+s4[3],1)*100:.1f}",
         f"{s5[3]/max(s5[0]+s5[3],1)*100:.1f}"),
        ("CNP Improve (%)",
         "--", "--",
         f"{(s3[5]+s3[7])/max(s3[4]+s3[6],1)*100:.1f}",
         f"{(s4[5]+s4[7])/max(s4[4]+s4[6],1)*100:.1f}",
         f"{(s5[5]+s5[7])/max(s5[4]+s5[6],1)*100:.1f}"),
    ]:
        print(f"  {label:<32} {str(val1):>10} {str(val2):>10} {str(val3):>10} {str(val4):>10} {str(val5):>10}")

    print("=" * 100)

    print("\nGenerating final animation...")
    visualize_final(r3f, eng3_final.all_towers, eng3_final.all_uavs,
                    output_gif="simulation_final_output.gif")
    print("Done! Check 'simulation_final_output.gif'.")

    return results


def run_final_only():
    """仅运行最终版"""
    print("\n" + "=" * 70)
    print("  Step 3 Final: Adaptive BDI + Smart CNP")
    print("  自适应阈值 + 交错换弹 + 弹药预留 + 内层塔精准激活")
    print("=" * 70)

    engine = FinalStep3Engine(random_seed=42)
    print("Running final simulation (20 rounds + cleanup)...")
    hits, total_cost, records = engine.run_full()

    print_final_report(hits, total_cost, records)

    output_path = "simulation_final_output.gif"
    print(f"\nGenerating animation (saving to {output_path})...")
    visualize_final(records, engine.all_towers, engine.all_uavs,
                    output_gif=output_path)

    print("\nFinal simulation completed!")
    return engine, records


if __name__ == "__main__":
    try:
        run_comparison()
    except ImportError:
        run_final_only()
