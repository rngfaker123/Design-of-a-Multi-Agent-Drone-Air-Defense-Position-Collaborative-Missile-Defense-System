import heapq
import random
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

import matplotlib
matplotlib.use('TkAgg')  
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  
matplotlib.rcParams['font.family'] = 'sans-serif'

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


# ===================== 1. 仿真参数配置 =====================
@dataclass
class SimConfig:
    # 战场空间参数
    width: float = 650.0
    height: float = 450.0
    base_x: float = 600.0
    base_y: float = 50.0
    base_radius: float = 100.0
    radar_range: float = 350.0

    # 来袭目标
    num_waves: int = 100
    wave_target_min: int = 5
    wave_target_max: int = 12
    decoy_ratio: float = 0.5
    decoy_speed_mean: float = 1.8
    decoy_speed_std: float = 0.8
    missile_speed_mean: float = 3.5
    missile_speed_std: float = 0.8
    target_spawn_x: float = 80.0

    # 防空阵地
    inner_battery_count: int = 9
    outer_battery_count: int = 9
    inner_max_ammo: int = 40
    outer_max_ammo: int = 60
    inner_cooldown: float = 1.2
    outer_cooldown: float = 0.8
    inner_hit_prob_base: float = 0.97
    outer_hit_prob_base: float = 0.88
    inner_min_range: float = 20.0
    inner_max_range: float = 220.0
    outer_min_range: float = 30.0
    outer_max_range: float = 320.0
    inner_shot_cost: float = 300.0
    outer_shot_cost: float = 100.0
    hit_decay_lambda: float = 0.6

    # 无人机（更强诱饵拦截能力）
    uav_count: int = 10
    uav_max_energy: float = 130.0
    uav_min_energy: float = 5.0
    uav_speed: float = 10.0
    uav_energy_per_dist: float = 0.15
    uav_intercept_cost: float = 50.0
    uav_max_intercept_time: float = 70.0
    uav_score_alpha: float = 0.7
    uav_score_beta: float = 0.3

    # 协同竞标
    bid_threshold: float = 0.1
    timeout_threshold: float = 5.0

    # 可视化
    vis_time_step: float = 0.18
    vis_fps: int = 30

    mc_runs: int = 1000
    random_seed: Optional[int] = 42



@dataclass
class Target:
    """来袭目标实体：诱饵/导弹"""
    tid: int
    target_type: int  # 0=慢速诱饵，1=高速导弹
    x: float
    y: float
    speed: float
    theta: float  # 航向角（弧度）
    alive: bool = True
    spawn_time: float = 0.0
    detected: bool = False
    death_x: Optional[float] = None  # 击毁时X坐标
    death_y: Optional[float] = None  # 击毁时Y坐标

    def get_position(self, current_time: float):
        """根据当前仿真时间，计算目标实时坐标（平滑移动）"""
        # 已击毁则固定在死亡坐标，不再移动
        if not self.alive and self.death_x is not None and self.death_y is not None:
            return self.death_x, self.death_y
        dt = current_time - self.spawn_time
        x = self.x + self.speed * math.cos(self.theta) * dt
        y = self.y + self.speed * math.sin(self.theta) * dt
        return x, y


@dataclass
class AirDefenseBattery:
    """防空阵地实体"""
    bid: int
    x: float
    y: float
    max_ammo: int
    current_ammo: int
    cooldown_time: float
    is_cooldown: bool = False
    cooldown_end_time: float = 0.0
    hit_prob_base: float = 0.8
    min_range: float = 50.0
    max_range: float = 200.0
    shot_cost: float = 100.0
    layer: str = "outer"

    def can_fire(self, current_time: float) -> bool:
        if self.current_ammo <= 0:
            return False
        if self.is_cooldown and current_time < self.cooldown_end_time:
            return False
        return True

    def get_hit_probability(self, target: Target, decay_lambda: float, current_time: float) -> float:
        tx, ty = target.get_position(current_time)
        dist = math.hypot(tx - self.x, ty - self.y)
        if dist < self.min_range or dist > self.max_range:
            return 0.0
        if self.current_ammo <= 0:
            return 0.0
        range_factor = max(0, dist - self.min_range) / (self.max_range - self.min_range)
        return self.hit_prob_base * math.exp(-decay_lambda * range_factor)

    def fire(self, current_time: float) -> bool:
        if not self.can_fire(current_time):
            return False
        self.current_ammo -= 1
        self.is_cooldown = True
        self.cooldown_end_time = current_time + self.cooldown_time
        return True


@dataclass
class UAV:
    """巡飞无人机实体"""
    uid: int
    x: float
    y: float
    max_energy: float
    current_energy: float
    speed: float
    energy_per_dist: float
    min_energy: float
    status: str = "patrol"  # patrol / intercepting / returning / dead
    target_id: Optional[int] = None
    task_start_time: float = 0.0
    task_end_time: float = 0.0
    start_x: float = 0.0
    start_y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    home_x: float = 0.0
    home_y: float = 0.0

    def can_intercept(self) -> bool:
        return self.status == "patrol" and self.current_energy > self.min_energy

    def get_position(self, current_time: float):
        """计算无人机实时坐标，根据状态做线性插值"""
        if self.status == "patrol" or self.status == "dead":
            return self.x, self.y

        dt_total = self.task_end_time - self.task_start_time
        if dt_total <= 1e-6:
            return self.target_x, self.target_y

        dt = min(max(0, current_time - self.task_start_time), dt_total)
        ratio = dt / dt_total
        x = self.start_x + (self.target_x - self.start_x) * ratio
        y = self.start_y + (self.target_y - self.start_y) * ratio
        return x, y

    def get_intercept_score(self, target: Target, current_time: float,
                            max_time: float, alpha: float, beta: float) -> float:
        if not self.can_intercept():
            return 0.0
        tx, ty = target.get_position(current_time)
        dist = math.hypot(tx - self.x, ty - self.y)
        ttc = dist / self.speed
        if ttc > max_time:
            return 0.0
        time_factor = math.exp(-ttc / max_time)
        energy_factor = (self.current_energy - self.min_energy) / (self.max_energy - self.min_energy)
        return alpha * time_factor + beta * energy_factor

    def start_intercept(self, target: Target, current_time: float):
        tx, ty = target.get_position(current_time)
        dist = math.hypot(tx - self.x, ty - self.y)
        self.status = "intercepting"
        self.target_id = target.tid
        self.task_start_time = current_time
        self.task_end_time = current_time + dist / self.speed
        self.start_x, self.start_y = self.x, self.y
        self.target_x, self.target_y = tx, ty
        self.current_energy -= dist * self.energy_per_dist

    def start_return(self, current_time: float):
        dist = math.hypot(self.x - self.home_x, self.y - self.home_y)
        self.status = "returning"
        self.target_id = None
        self.task_start_time = current_time
        self.task_end_time = current_time + dist / self.speed
        self.start_x, self.start_y = self.x, self.y
        self.target_x, self.target_y = self.home_x, self.home_y
        self.current_energy -= dist * self.energy_per_dist


# ===================== 3. 离散事件仿真引擎 =====================
class AirDefenseSimulation:
    def __init__(self, config: SimConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.current_time = 0.0
        self.event_queue = []
        self.event_seq = 0

        self.targets: Dict[int, Target] = {}
        self.batteries: Dict[int, AirDefenseBattery] = {}
        self.uavs: Dict[int, UAV] = {}
        self.next_tid = 0
        self.next_bid = 0
        self.next_uid = 0

        self.total_spawned = 0
        self.total_destroyed = 0
        self.total_cost = 0.0
        self.base_hit = False
        self.base_hit_time = float('inf')
        self.finished = False

        self._deploy_batteries()
        self._deploy_uavs()
        self._schedule_wave(0.0)

    def _push_event(self, time: float, event_type: str, data: dict, priority: int = 5):
        self.event_seq += 1
        heapq.heappush(
            self.event_queue,
            (time, priority, self.event_seq, event_type, data)
        )

    def _deploy_batteries(self):
        cfg = self.config
        radius = 60.0
        for i in range(cfg.inner_battery_count):
            angle = 2 * math.pi * i / cfg.inner_battery_count
            x = cfg.base_x + radius * math.cos(angle)
            y = cfg.base_y + radius * math.sin(angle)
            y = max(10, min(cfg.height - 10, y))
            self.batteries[self.next_bid] = AirDefenseBattery(
                bid=self.next_bid, x=x, y=y,
                max_ammo=cfg.inner_max_ammo,
                current_ammo=cfg.inner_max_ammo,
                cooldown_time=cfg.inner_cooldown,
                hit_prob_base=cfg.inner_hit_prob_base,
                min_range=cfg.inner_min_range,
                max_range=cfg.inner_max_range,
                shot_cost=cfg.inner_shot_cost,
                layer="inner"
            )
            self.next_bid += 1


        outer_total = cfg.outer_battery_count
        col_x = 450.0
        col_y_start = 0.0
        col_y_end = 120.0
        row_y = 160.0
        row_x_start = 450.0
        row_x_end = 640.0

        col_count = outer_total // 2
        row_count = outer_total - col_count

        if col_count > 0:
            y_step_col = (col_y_end - col_y_start) / max(1, col_count - 1)
            for i in range(col_count):
                x = col_x
                y = col_y_start + i * y_step_col
                x = max(10, min(cfg.width - 10, x))
                y = max(10, min(cfg.height - 10, y))
                self.batteries[self.next_bid] = AirDefenseBattery(
                    bid=self.next_bid, x=x, y=y,
                    max_ammo=cfg.outer_max_ammo,
                    current_ammo=cfg.outer_max_ammo,
                    cooldown_time=cfg.outer_cooldown,
                    hit_prob_base=cfg.outer_hit_prob_base,
                    min_range=cfg.outer_min_range,
                    max_range=cfg.outer_max_range,
                    shot_cost=cfg.outer_shot_cost,
                    layer="outer"
                )
                self.next_bid += 1
        if row_count > 0:
            x_step_row = (row_x_end - row_x_start) / max(1, row_count - 1)
            for i in range(row_count):
                x = row_x_start + i * x_step_row
                y = row_y
                x = max(10, min(cfg.width - 10, x))
                y = max(10, min(cfg.height - 10, y))
                self.batteries[self.next_bid] = AirDefenseBattery(
                    bid=self.next_bid, x=x, y=y,
                    max_ammo=cfg.outer_max_ammo,
                    current_ammo=cfg.outer_max_ammo,
                    cooldown_time=cfg.outer_cooldown,
                    hit_prob_base=cfg.outer_hit_prob_base,
                    min_range=cfg.outer_min_range,
                    max_range=cfg.outer_max_range,
                    shot_cost=cfg.outer_shot_cost,
                    layer="outer"
                )
                self.next_bid += 1

    def _deploy_uavs(self):
        cfg = self.config
        uid = self.next_uid
        uav_total = cfg.uav_count
        half = uav_total // 2

   
        for i in range(half):
            x = 640
            y = 200 + i * 50
            y = max(20, min(cfg.height - 20, y))
            self.uavs[uid] = UAV(
                uid=uid, x=x, y=y,
                max_energy=cfg.uav_max_energy,
                current_energy=cfg.uav_max_energy,
                speed=cfg.uav_speed,
                energy_per_dist=cfg.uav_energy_per_dist,
                min_energy=cfg.uav_min_energy,
                home_x=x, home_y=y
            )
            uid += 1


        remain = uav_total - half
        start_x = 320
        for i in range(remain):
            x = start_x + i * 80
            y = 60
            x = max(20, min(cfg.width - 20, x))
            self.uavs[uid] = UAV(
                uid=uid, x=x, y=y,
                max_energy=cfg.uav_max_energy,
                current_energy=cfg.uav_max_energy,
                speed=cfg.uav_speed,
                energy_per_dist=cfg.uav_energy_per_dist,
                min_energy=cfg.uav_min_energy,
                home_x=x, home_y=y
            )
            uid += 1
        self.next_uid = uid

    def _schedule_wave(self, time: float):
        self._push_event(time, "spawn_wave", {}, priority=3)

    def _calc_arrival_time(self, target: Target, cx: float, cy: float, r: float) -> Optional[float]:
        x0, y0 = target.x, target.y
        vx = target.speed * math.cos(target.theta)
        vy = target.speed * math.sin(target.theta)
        dx0 = x0 - cx
        dy0 = y0 - cy

        a = vx ** 2 + vy ** 2
        b = 2 * (dx0 * vx + dy0 * vy)
        c = dx0 ** 2 + dy0 ** 2 - r ** 2
        delta = b ** 2 - 4 * a * c

        if delta < 0:
            return None
        sqrt_d = math.sqrt(delta)
        t1 = (-b - sqrt_d) / (2 * a)
        t2 = (-b + sqrt_d) / (2 * a)

        for t in [t1, t2]:
            if t > 1e-6:
                return target.spawn_time + t
        return None

    def _spawn_wave(self):
        cfg = self.config
        num = random.randint(cfg.wave_target_min, cfg.wave_target_max)

        for _ in range(num):
            is_decoy = random.random() < cfg.decoy_ratio
            t_type = 0 if is_decoy else 1
            y = random.uniform(100, cfg.height - 50)

            if t_type == 0:
                speed = max(1.0, random.gauss(cfg.decoy_speed_mean, cfg.decoy_speed_std))
            else:
                speed = max(4.0, random.gauss(cfg.missile_speed_mean, cfg.missile_speed_std))

            dx = cfg.base_x - cfg.target_spawn_x
            dy = cfg.base_y - y
            theta = math.atan2(dy, dx) + random.uniform(-0.1, 0.1)

            target = Target(
                tid=self.next_tid, target_type=t_type,
                x=cfg.target_spawn_x, y=y,
                speed=speed, theta=theta,
                spawn_time=self.current_time
            )
            self.targets[self.next_tid] = target
            self.total_spawned += 1
            self.next_tid += 1

            detect_time = self._calc_arrival_time(
                target, cfg.base_x, cfg.base_y, cfg.radar_range
            )
            if detect_time:
                self._push_event(detect_time, "detect", {"tid": target.tid}, priority=4)
            else:
                hit_time = self._calc_arrival_time(
                    target, cfg.base_x, cfg.base_y, cfg.base_radius
                )
                if hit_time:
                    self._push_event(hit_time, "base_hit", {"tid": target.tid}, priority=1)

        wave_interval = random.uniform(10, 20)
        # if self.total_spawned < cfg.num_waves * cfg.wave_target_min:
        self._schedule_wave(self.current_time + wave_interval)

    def _handle_detect(self, tid: int):
        if tid not in self.targets or not self.targets[tid].alive:
            return
        self.targets[tid].detected = True
        self._push_event(self.current_time, "bidding", {"tid": tid}, priority=4)

    def _handle_bidding(self, tid: int):
        cfg = self.config
        if tid not in self.targets or not self.targets[tid].alive:
            return
        target = self.targets[tid]
        tx, ty = target.get_position(self.current_time)
        dist_to_base = math.hypot(tx - cfg.base_x, ty - cfg.base_y)
        candidates = []

        if dist_to_base > cfg.radar_range:
            if target.target_type == 0:
                for uid, uav in self.uavs.items():
                    score = uav.get_intercept_score(
                        target, self.current_time, cfg.uav_max_intercept_time,
                        cfg.uav_score_alpha, cfg.uav_score_beta
                    )
                    if score >= cfg.bid_threshold:
                        candidates.append((score, uav.current_energy, "uav", uid))
        # 区间2：雷达圈内、内层最大射程外 inner_max_range < dist ≤ radar_range → 外层防空全部拦截
        elif cfg.inner_max_range < dist_to_base <= cfg.radar_range:
            for bid, bat in self.batteries.items():
                if bat.layer != "outer":
                    continue
                if not bat.can_fire(self.current_time):
                    continue
                prob = bat.get_hit_probability(target, cfg.hit_decay_lambda, self.current_time)
                if prob > 0:
                    candidates.append((prob, bat.current_ammo, "battery", bid))
        # 区间3：内层防御圈 dist ≤ inner_max_range → 内层防空兜底拦截
        else:
            for bid, bat in self.batteries.items():
                if bat.layer != "inner":
                    continue
                if not bat.can_fire(self.current_time):
                    continue
                prob = bat.get_hit_probability(target, cfg.hit_decay_lambda, self.current_time)
                if prob > 0:
                    candidates.append((prob, bat.current_ammo, "battery", bid))

        if not candidates:
            hit_time = self._calc_arrival_time(
                target, cfg.base_x, cfg.base_y, cfg.base_radius
            )
            if hit_time:
                self._push_event(hit_time, "base_hit", {"tid": tid}, priority=1)
            return

        candidates.sort(key=lambda x: (-x[0], -x[1]))
        score, _, e_type, e_id = candidates[0]

        if e_type == "uav":
            uav = self.uavs[e_id]
            uav.start_intercept(target, self.current_time)
            self.total_cost += cfg.uav_intercept_cost
            self._push_event(
                uav.task_end_time, "intercept_result",
                {"type": "uav", "id": e_id, "tid": tid, "prob": score},
                priority=3
            )
        else:
            bat = self.batteries[e_id]
            bat.fire(self.current_time)
            self.total_cost += bat.shot_cost
            tx, ty = target.get_position(self.current_time)
            dist = math.hypot(tx - bat.x, ty - bat.y)
            proj_speed = 50.0 if bat.layer == "inner" else 5.0
            flight_time = dist / proj_speed
            self._push_event(
                self.current_time + flight_time, "intercept_result",
                {"type": "battery", "id": e_id, "tid": tid, "prob": score},
                priority=3
            )
            self._push_event(
                bat.cooldown_end_time, "cooldown_end",
                {"bid": e_id}, priority=5
            )

    def _handle_intercept(self, data: dict):
        tid = data["tid"]
        hit_prob = data["prob"]
        e_type = data["type"]
        e_id = data["id"]

        # 目标已被击毁，直接释放资源
        if tid not in self.targets or not self.targets[tid].alive:
            self._release_entity(data)
            return

        target = self.targets[tid]
        hit = random.random() < hit_prob
        if hit:
            target.alive = False
            # 记录击毁坐标，永久固定
            target.death_x, target.death_y = target.get_position(self.current_time)
            self.total_destroyed += 1
            print(f"【{self.current_time:.2f}】{e_type}{e_id} 成功击毁目标{tid}，击毁坐标({target.death_x:.1f},{target.death_y:.1f})")
        else:
            # 拦截失败，重新发起竞标
            print(f"【{self.current_time:.2f}】{e_type}{e_id} 拦截目标{tid}失败，重新竞标")
            self._push_event(
                self.current_time + 0.5, "bidding",
                {"tid": tid}, priority=4
            )
        self._release_entity(data)

    def _release_entity(self, data: dict):
        e_type = data["type"]
        e_id = data["id"]
        if e_type == "uav" and e_id in self.uavs:
            uav = self.uavs[e_id]
            uav.x, uav.y = uav.target_x, uav.target_y
            uav.start_return(self.current_time)
            # 推送无人机返航结束事件
            self._push_event(
                uav.task_end_time, "uav_return_end",
                {"uid": e_id}, priority=5
            )
        # 防空阵地无需返航，仅等待冷却结束，无需额外操作

    def _handle_cooldown_end(self, bid: int):
        if bid in self.batteries:
            self.batteries[bid].is_cooldown = False

    def _handle_uav_return_end(self, uid: int):
        if uid in self.uavs:
            uav = self.uavs[uid]
            uav.x = uav.home_x
            uav.y = uav.home_y
            uav.status = "patrol" if uav.current_energy > uav.min_energy else "dead"

    def _handle_base_hit(self, tid: int):
        if tid not in self.targets or not self.targets[tid].alive:
            return
        self.base_hit = True
        self.base_hit_time = self.current_time
        self.finished = True
        self.event_queue.clear()

    def step_forward(self, end_time: float):
        """推进仿真到指定时间，处理所有到期事件"""
        start_time = self.current_time

        while self.event_queue and not self.finished:
            ev_time, _, _, event_type, data = self.event_queue[0]
            if ev_time > end_time:
                break
            heapq.heappop(self.event_queue)
            self.current_time = ev_time

            if event_type == "spawn_wave":
                self._spawn_wave()
            elif event_type == "detect":
                self._handle_detect(data["tid"])
            elif event_type == "bidding":
                self._handle_bidding(data["tid"])
            elif event_type == "intercept_result":
                self._handle_intercept(data)
            elif event_type == "cooldown_end":
                self._handle_cooldown_end(data["bid"])
            elif event_type == "uav_return_end":
                self._handle_uav_return_end(data["uid"])
            elif event_type == "base_hit":
                self._handle_base_hit(data["tid"])
        self.current_time = end_time

        # 处理无人机返航结束事件
        for uid, uav in self.uavs.items():
            if uav.status == "returning" and self.current_time >= uav.task_end_time:
                self._handle_uav_return_end(uid)

        # 所有事件处理完毕，仿真结束
        if not self.event_queue and not self.base_hit:
            self.finished = True

        return self.current_time - start_time


    def get_metrics(self) -> dict:
        end_time = self.base_hit_time if self.base_hit else self.current_time
        total_max_ammo = sum(b.max_ammo for b in self.batteries.values())
        total_remain_ammo = sum(b.current_ammo for b in self.batteries.values())

        return {
            "base_hit": self.base_hit,
            "survival_time": end_time,
            "intercept_rate": self.total_destroyed / max(1, self.total_spawned),
            "ammo_retention_rate": total_remain_ammo / max(1, total_max_ammo),
            "total_cost": self.total_cost,
            "unit_kill_cost": self.total_cost / max(1, self.total_destroyed),
            "total_targets": self.total_spawned,
            "destroyed_targets": self.total_destroyed
        }


# ===================== 4. 战场可视化模块 =====================
class BattlefieldVisualizer:
    def __init__(self, sim: AirDefenseSimulation, config: SimConfig):
        self.sim = sim
        self.cfg = config
        self.fig, self.ax = plt.subplots(figsize=(12, 8), dpi=100)
        self._init_canvas()
        self._init_artists()
        # 存储击毁目标叉号绘制对象，每帧清空重绘
        self.dead_markers = []

    def _init_canvas(self):
        cfg = self.cfg
        self.ax.set_xlim(-20, cfg.width + 20)
        self.ax.set_ylim(-20, cfg.height + 20)
        self.ax.set_aspect('equal')
        self.ax.set_title(
            '分层防空仿真：外圈无人机拦截诱饵 | 雷达内外层防空 | 核心内层兜底',
            fontsize=12)
        self.ax.set_xlabel('X coordinate')
        self.ax.set_ylabel('Y coordinate')
        self.ax.grid(True, alpha=0.3)

        # 绘制固定元素：基地、雷达范围
        base_circle = Circle((cfg.base_x, cfg.base_y), cfg.base_radius,
                             color='darkblue', alpha=0.15, label='核心保护区')
        radar_circle = Circle((cfg.base_x, cfg.base_y), cfg.radar_range,
                              color='gray', alpha=0.1, linestyle='--', label='预警雷达范围')
        inner_def_circle = Circle((cfg.base_x, cfg.base_y), cfg.inner_max_range,
                                  color='red', alpha=0.08, linestyle=':', label='内层防空射程圈')
        self.ax.add_patch(base_circle)
        self.ax.add_patch(radar_circle)
        self.ax.add_patch(inner_def_circle)
        self.ax.scatter(cfg.base_x, cfg.base_y, marker='*', s=200,
                        color='darkblue', zorder=5, label='指挥中心')

    def _init_artists(self):
        # 防空阵地
        self.inner_bat_scatter = self.ax.scatter([], [], marker='s', s=120,
                                                 color='crimson', zorder=4, label='内层防空阵地(核心兜底)')
        self.outer_bat_scatter = self.ax.scatter([], [], marker='s', s=100,
                                                 color='darkorange', zorder=4, label='外层防空阵地(雷达圈内拦截)')

        # 无人机
        self.uav_patrol = self.ax.scatter([], [], marker='^', s=80,
                                          color='royalblue', zorder=4, label='巡逻无人机(仅雷达外拦截诱饵)')
        self.uav_intercept = self.ax.scatter([], [], marker='^', s=80,
                                             color='limegreen', zorder=4, label='拦截中无人机')

        # 来袭目标
        self.decoy_scatter = self.ax.scatter([], [], marker='o', s=40,
                                             color='gold', alpha=0.8, label='慢速诱饵')
        self.missile_scatter = self.ax.scatter([], [], marker='o', s=70,
                                               color='red', alpha=0.9, label='高速导弹')

        # 统计文本
        self.stats_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                       va='top', fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        self.ax.legend(loc='upper right', fontsize=9)

    def _update_positions(self):
        sim = self.sim
        t = sim.current_time
        # 清空原有文字标注
        for txt in self.ax.texts[1:]:
            txt.remove()
        # 清除上一帧所有击毁叉号
        for marker in self.dead_markers:
            marker.remove()
        self.dead_markers.clear()

        # 更新阵地位置（固定）
        inner_x, inner_y = [], []
        outer_x, outer_y = [], []
        for bat in sim.batteries.values():
            if bat.layer == 'inner':
                inner_x.append(bat.x)
                inner_y.append(bat.y)
            else:
                outer_x.append(bat.x)
                outer_y.append(bat.y)
            # 绘制弹药文字
            cool_text = "冷却" if bat.is_cooldown and t < bat.cooldown_end_time else ""
            self.ax.text(bat.x + 5, bat.y, f"{bat.current_ammo}{cool_text}", fontsize=8)
        self.inner_bat_scatter.set_offsets(np.c_[inner_x, inner_y])
        self.outer_bat_scatter.set_offsets(np.c_[outer_x, outer_y])

        # 更新无人机位置
        patrol_x, patrol_y = [], []
        intercept_x, intercept_y = [], []
        for uav in sim.uavs.values():
            x, y = uav.get_position(t)
            if uav.status == 'patrol':
                patrol_x.append(x)
                patrol_y.append(y)
            elif uav.status == 'intercepting' or uav.status == 'returning':
                intercept_x.append(x)
                intercept_y.append(y)
        self.uav_patrol.set_offsets(np.c_[patrol_x, patrol_y] if patrol_x else np.empty((0, 2)))
        self.uav_intercept.set_offsets(np.c_[intercept_x, intercept_y] if intercept_x else np.empty((0, 2)))

        # 更新存活目标 + 绘制击毁目标红叉
        decoy_x, decoy_y = [], []
        missile_x, missile_y = [], []
        for target in sim.targets.values():
            x, y = target.get_position(t)
            if not target.alive:
                # 绘制红色叉号标记击毁目标，永久固定
                cross = self.ax.scatter(x, y, marker='x', s=120, color='red', linewidth=2, zorder=10)
                self.dead_markers.append(cross)
                continue
            if target.target_type == 0:
                decoy_x.append(x)
                decoy_y.append(y)
            else:
                missile_x.append(x)
                missile_y.append(y)
        self.decoy_scatter.set_offsets(np.c_[decoy_x, decoy_y] if decoy_x else np.empty((0, 2)))
        self.missile_scatter.set_offsets(np.c_[missile_x, missile_y] if missile_x else np.empty((0, 2)))

        # 更新统计文本
        metrics = sim.get_metrics()
        status = "基地已被击中" if metrics['base_hit'] else "防御中"
        self.stats_text.set_text(
            f"仿真时间: {t:.1f}\n"
            f"状态: {status}\n"
            f"来袭目标: {metrics['total_targets']} | 已拦截: {metrics['destroyed_targets']}\n"
            f"拦截率: {metrics['intercept_rate'] * 100:.1f}%\n"
            f"弹药留存: {metrics['ammo_retention_rate'] * 100:.1f}%\n"
            f"总成本: {metrics['total_cost']:.0f} | 单位击杀: {metrics['unit_kill_cost']:.1f}"
        )

    def animate(self, frame):
        if not self.sim.finished:
            next_time = self.sim.current_time + self.cfg.vis_time_step
            self.sim.step_forward(next_time)
        self._update_positions()
        # 返回所有需要刷新的绘图对象
        all_artists = [self.inner_bat_scatter, self.outer_bat_scatter,
                       self.uav_patrol, self.uav_intercept,
                       self.decoy_scatter, self.missile_scatter,
                       self.stats_text] + self.dead_markers
        return all_artists

    def run(self):
        interval = 1000 / self.cfg.vis_fps
        anim = FuncAnimation(
            self.fig, self.animate,
            interval=interval, blit=False,
            cache_frame_data=False
        )
        plt.tight_layout()
        plt.show(block=True)
        return anim


if __name__ == "__main__":
    config = SimConfig(random_seed=42)

    print("启动分层防空可视化演示：无人机雷达外拦截诱饵、外层雷达内拦截、内层核心兜底")
    sim = AirDefenseSimulation(config)
    vis = BattlefieldVisualizer(sim, config)
    anim = vis.run()  
    input("按回车退出...")