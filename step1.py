"""
捣蛋对对碰 - 导弹防御模拟系统
Missile Defense Simulation

Features:
- Map 650x450, base at (600,50), protected radius 100
- 7 inner towers (max shell speed 20, cost 300/shell)
- 6 outer towers (max shell speed 5, cost 100*v/shell)
- 20 rounds, 5-15 missiles each, speed ~ N(4,2) rounded to int
- Physics-based minimum intercept speed calculation
- Emergency queue for missiles that can't be intercepted normally
- Hit probability: p = shell_number / missile_speed
- Bid function: p_hit / cost
- matplotlib animation (frame-by-frame visualization)
"""
import numpy as np
import random
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle as MplCircle
import matplotlib.patches as mpatches
import os

# 常量配置
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
TARGET_HIT_PROB = 0.90

# 数据结构

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


@dataclass
class Tower:
    tower_id: int
    position: Tuple[float, float]
    is_inner: bool
    ammo: int = MAX_AMMO
    # reload_state: None=ready, 'reloading'=next cycle ready, 'exhausted'=next cycle reloading
    reload_state: Optional[str] = None
    fired_this_cycle: bool = False
    total_shells_fired: int = 0
    total_cost: float = 0.0

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
        actual = min(count, self.ammo)
        cost = actual * self.get_cost_per_shell(shell_speed)
        self.ammo -= actual
        self.total_shells_fired += actual
        self.total_cost += cost
        if actual > 0:
            self.fired_this_cycle = True
        return actual, cost


@dataclass
class InterceptResult:
    tower: Tower
    can_intercept: bool
    min_shell_speed: float
    intercept_point: Tuple[float, float]
    intercept_time: float


@dataclass
class PendingIntercept:
    """上一周期发起的拦截任务，下一周期结算。"""
    missile: Missile
    cumulative_prob: float
    cost: float
    intercept_point: Tuple[float, float]


@dataclass
class CycleRecord:
    cycle_num: int
    missiles: List[Missile] = field(default_factory=list)
    emergency_ids: List[int] = field(default_factory=list)
    towers: List[Tower] = field(default_factory=list)
    hit_base_this_cycle: int = 0
    missiles_destroyed: int = 0
    total_cost: float = 0.0
    total_hits: int = 0
    destroyed_positions: List[Tuple[float, float]] = field(default_factory=list)
    failed_positions: List[Tuple[float, float]] = field(default_factory=list)


# 拦截相关的几何/物理计算

def compute_min_intercept_speed(
    missile_pos: Tuple[float, float],
    missile_speed: float,
    missile_target: Tuple[float, float],
    tower_pos: Tuple[float, float],
    max_shell_speed: float,
    protected_radius: float = PROTECTED_RADIUS,
) -> Tuple[bool, float, Tuple[float, float], float]:
    """计算在保护区外完成拦截所需的最小弹速。"""
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

    # 最优拦截距离（最小化 |A + d*u| / (d/v_m)）
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
        # 需要更早拦截
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


def shells_needed(survival: float, missile_speed: int, start_num: int) -> Tuple[int, float, float]:
    """计算达到 P(hit) >= 0.90 所需的炮弹数。"""
    n = 0
    while survival > 0.10 and n < 1000:
        sn = start_num + n + 1
        p_hit = min(sn / missile_speed, 1.0)
        survival *= (1.0 - p_hit)
        n += 1
        if p_hit >= 1.0:
            break
    return n, survival, 1.0 - survival


# 仿真主逻辑

class SimulationEngine:
    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)

        # 防空塔
        self.all_towers: List[Tower] = []
        tid = 0
        for pos in INNER_TOWER_POSITIONS:
            self.all_towers.append(Tower(tid, pos, True))
            tid += 1
        for pos in OUTER_TOWER_POSITIONS:
            self.all_towers.append(Tower(tid, pos, False))
            tid += 1

        # 队列
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
        self.cycle_records: List[CycleRecord] = []

    # 生成来袭导弹
    def spawn_point(self) -> Tuple[float, float]:
        if self.rng.random() < 0.5:
            return (float(self.rng.randint(0, MAP_WIDTH)), 400.0)
        return (50.0, float(self.rng.randint(0, MAP_HEIGHT)))

    def spawn_speed(self) -> int:
        s = self.rng.gauss(MISSILE_SPEED_MEAN, math.sqrt(MISSILE_SPEED_VARIANCE))
        return max(1, int(round(s)))

    def spawn_missiles(self):
        n = self.rng.randint(MIN_MISSILES_PER_ROUND, MAX_MISSILES_PER_ROUND)
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

    # 拦截计算
    def compute_intercept(self, tower: Tower, missile: Missile,
                          for_emergency: bool) -> InterceptResult:
        pr = 0.0 if for_emergency else PROTECTED_RADIUS
        ok, spd, pt, tm = compute_min_intercept_speed(
            missile.current_position, float(missile.speed), BASE,
            tower.position, tower.max_shell_speed, pr,
        )
        return InterceptResult(tower, ok, spd, pt, tm)

    def bid_value(self, res: InterceptResult, missile: Missile) -> float:
        if not res.can_intercept:
            return -1.0
        p_hit = 1.0 / missile.speed
        cost = res.tower.get_cost_per_shell(res.min_shell_speed)
        return p_hit / cost if cost > 0 else float('inf')

    # 交战流程
    def engage(self, missile: Missile, for_emergency: bool
              ) -> Tuple[float, float, Optional[Tuple[float, float]]]:
        """对单枚导弹，按性价比挑选塔位逐步拦截。"""
        results = []
        for t in self.all_towers:
            if t.can_fire:
                results.append(self.compute_intercept(t, missile, for_emergency))

        results.sort(key=lambda r: self.bid_value(r, missile), reverse=True)

        survival = 1.0
        shells_total = missile.total_shells_fired_at
        total_cost = 0.0
        last_ipt: Optional[Tuple[float, float]] = None

        for res in results:
            if not res.can_intercept or survival <= 0.10:
                break
            tower = res.tower
            if not tower.can_fire or tower.ammo <= 0:
                continue

            needed, _, _ = shells_needed(survival, missile.speed, shells_total)
            if needed == 0:
                break

            fired, cost = tower.fire(needed, res.min_shell_speed)
            if fired == 0:
                continue

            total_cost += cost
            for i in range(fired):
                sn = shells_total + i + 1
                p_hit = min(sn / missile.speed, 1.0)
                survival *= (1.0 - p_hit)
            shells_total += fired
            last_ipt = res.intercept_point

        missile.total_shells_fired_at = shells_total
        return 1.0 - survival, total_cost, last_ipt

    # 周期流程
    def phase_resolve(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """结算上一周期的拦截结果。"""
        destroyed = []
        failed = []
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
            else:
                failed.append(pi.intercept_point)
                m.current_position = pi.intercept_point
                if m.dist_to_base < 1.0:
                    m.hit_base = True
                    m.resolved = True
                    self.hit_counter += 1
                else:
                    self.emergency_queue.append(m)

        self.pending_intercepts.clear()
        return destroyed, failed

    def phase_emergency(self) -> int:
        """处理紧急队列。"""
        hits = 0
        kept: List[Missile] = []

        for m in self.emergency_queue:
            if m.resolved:
                continue

            cum_prob, cost, ipt = self.engage(m, for_emergency=True)

            if cum_prob >= TARGET_HIT_PROB:
                self.pending_intercepts.append(PendingIntercept(
                    missile=m, cumulative_prob=cum_prob,
                    cost=cost, intercept_point=ipt or m.current_position,
                ))
                kept.append(m)
            else:
                m.hit_base = True
                m.resolved = True
                self.hit_counter += 1
                hits += 1

        self.emergency_queue = kept
        return hits

    def phase_regular(self):
        """处理常规队列。"""
        for m in self.regular_queue:
            if m.resolved:
                continue

            can_any = False
            for t in self.all_towers:
                if t.can_fire and self.compute_intercept(t, m, False).can_intercept:
                    can_any = True
                    break

            if not can_any:
                self.emergency_queue.append(m)
                continue

            cum_prob, cost, ipt = self.engage(m, for_emergency=False)

            if cum_prob >= TARGET_HIT_PROB:
                self.pending_intercepts.append(PendingIntercept(
                    missile=m, cumulative_prob=cum_prob,
                    cost=cost, intercept_point=ipt or m.current_position,
                ))
            else:
                self.emergency_queue.append(m)

        self.regular_queue.clear()

    def phase_move(self):
        """推进导弹位置。"""
        for pi in self.pending_intercepts:
            m = pi.missile
            if m.resolved:
                continue
            dx, dy = m.direction
            nx = m.current_position[0] + dx * m.speed
            ny = m.current_position[1] + dy * m.speed
            m.current_position = (nx, ny)
            if m.dist_to_base < 1.0:
                m.hit_base = True
                m.resolved = True
                self.hit_counter += 1

        for m in self.emergency_queue:
            if m.resolved:
                continue
            if m.dist_to_base < 1.0:
                m.hit_base = True
                m.resolved = True
                self.hit_counter += 1

    def phase_reload(self):
        """处理塔位换弹。"""
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

    # 主循环
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

    def run_cycle(self) -> CycleRecord:
        self.current_cycle += 1

        destroyed_positions, failed_positions = self.phase_resolve()
        hits = self.phase_emergency()

        if self.current_round < TOTAL_ROUNDS:
            self.spawn_missiles()
            self.current_round += 1

        self.phase_regular()
        self.phase_move()
        self.phase_reload()

        rec = CycleRecord(
            cycle_num=self.current_cycle,
            missiles=self.active_missiles(),
            emergency_ids=[m.missile_id for m in self.emergency_queue if not m.resolved],
            towers=[Tower(t.tower_id, t.position, t.is_inner,
                         t.ammo, t.reload_state, t.fired_this_cycle,
                         t.total_shells_fired, t.total_cost)
                    for t in self.all_towers],
            hit_base_this_cycle=hits,
            missiles_destroyed=self.missiles_destroyed,
            total_cost=self.total_cost,
            total_hits=self.hit_counter,
            destroyed_positions=destroyed_positions,
            failed_positions=failed_positions,
        )
        self.cycle_records.append(rec)
        return rec

    def run_full(self) -> Tuple[int, float, List[CycleRecord]]:
        max_c = 10000
        while not self.is_done() and self.current_cycle < max_c:
            self.run_cycle()
            if self.current_cycle % 5 == 0:
                print(f"  Cycle {self.current_cycle}: hits={self.hit_counter}, "
                      f"destroyed={self.missiles_destroyed}, cost={self.total_cost:.0f}")

        print(f"\nSimulation finished: {self.current_cycle} cycles, "
              f"{self.current_round} rounds")
        return self.hit_counter, self.total_cost, self.cycle_records


# 可视化

def visualize(records: List[CycleRecord], static_towers: List[Tower],
              output_gif: Optional[str] = None):
    """生成动画，可选保存为 GIF。"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(13, 9))

    def draw(frame_idx):
        ax.clear()
        if frame_idx >= len(records):
            return
        rec = records[frame_idx]

        ax.set_xlim(-20, MAP_WIDTH + 20)
        ax.set_ylim(-20, MAP_HEIGHT + 20)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Missile Defense Simulation  |  Cycle {rec.cycle_num}/{len(records)}  |  '
                     f'Hits: {rec.total_hits}  |  Cost: {rec.total_cost:.0f}  |  '
                     f'Destroyed: {rec.missiles_destroyed}',
                     fontsize=12, fontweight='bold')

        pc = MplCircle(BASE, PROTECTED_RADIUS, fill=True, alpha=0.1,
                       edgecolor='red', linestyle='--', linewidth=1.2,
                       facecolor='pink')
        ax.add_patch(pc)
        ax.text(BASE[0], BASE[1] - PROTECTED_RADIUS - 8, 'Protected Zone',
                ha='center', fontsize=7, color='red')

        ax.plot(BASE[0], BASE[1], 'r*', markersize=20, markeredgecolor='darkred',
                markeredgewidth=3, zorder=15)
        ax.text(BASE[0] + 12, BASE[1] + 12, 'BASE', fontsize=10,
                fontweight='bold', color='darkred')

        for t in static_towers:
            if t.is_inner:
                ax.scatter(t.position[0], t.position[1], c='royalblue', marker='s',
                           s=100, edgecolors='navy', linewidths=1.5, zorder=9)
        for t in static_towers:
            if not t.is_inner:
                ax.scatter(t.position[0], t.position[1], c='forestgreen', marker='^',
                           s=100, edgecolors='darkgreen', linewidths=1.5, zorder=9)

        for t in rec.towers:
            if t.reload_state is not None:
                ax.plot(t.position[0], t.position[1], 'x', color='orange',
                        markersize=14, markeredgewidth=3, zorder=11)
            ax.text(t.position[0] + 4, t.position[1] + 4, str(t.ammo),
                    fontsize=6, color='black', alpha=0.8)

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

        for pos in rec.destroyed_positions:
            ax.plot(pos[0], pos[1], 'x', color='limegreen', markersize=14,
                   markeredgewidth=3, zorder=12)
        for pos in rec.failed_positions:
            ax.plot(pos[0], pos[1], 'x', color='magenta', markersize=14,
                   markeredgewidth=3, zorder=12)

        legend_elements = [
            mpatches.Patch(color='royalblue', label='Inner Tower (v<=20, cost=300)'),
            mpatches.Patch(color='forestgreen', label='Outer Tower (v<=5, cost=100v)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                       markersize=8, label='Regular Missile'),
            plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='darkorange',
                       markersize=8, label='Emergency Missile'),
            plt.Line2D([0], [0], marker='x', color='orange', markersize=8,
                       label='Reloading'),
            plt.Line2D([0], [0], marker='x', color='limegreen', markersize=8,
                       label='Missile Destroyed'),
            plt.Line2D([0], [0], marker='x', color='magenta', markersize=8,
                       label='Intercept Failed'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                 framealpha=0.85)
        ax.grid(True, alpha=0.2, linestyle='--')

    anim = FuncAnimation(fig, draw, frames=len(records), interval=800,
                         repeat=False, blit=False)
    plt.tight_layout()

    if output_gif:
        print(f"Saving animation to {output_gif}...")
        writer = PillowWriter(fps=1)
        anim.save(output_gif, writer=writer)
        print(f"Animation saved to {output_gif}")
    else:
        plt.show()

    return anim


def print_report(hits: int, cost: float, records: List[CycleRecord]):
    print("\n" + "=" * 65)
    print("      Missile Defense Simulation - Final Report")
    print("=" * 65)
    print(f"  Total cycles       : {len(records)}")
    print(f"  Base hit counter   : {hits}")
    print(f"  Total cost         : {cost:.2f}")
    if records:
        last = records[-1]
        print(f"  Missiles destroyed : {last.missiles_destroyed}")
        print(f"  Est. total spawned : {last.missiles_destroyed + hits}")
    print()
    print("  Tower Statistics:")
    print("  " + "-" * 63)
    if records:
        for t in records[-1].towers:
            ttype = "INNER" if t.is_inner else "OUTER"
            print(f"  ID{t.tower_id:<3} ({t.position[0]:.0f},{t.position[1]:.0f})  "
                  f"{ttype:<6}  Fired:{t.total_shells_fired:<5}  "
                  f"Cost:{t.total_cost:<12.2f}  Ammo:{t.ammo}")
    print("=" * 65)


def main():
    print("=" * 50)
    print("Initializing simulation engine...")
    eng = SimulationEngine(random_seed=42)

    print("Running simulation (20 rounds + cleanup)...")
    hits, cost, records = eng.run_full()

    print_report(hits, cost, records)

    save_path = "simulation_output.gif"
    print(f"\nGenerating animation (saving to {save_path})...")
    visualize(records, eng.all_towers, output_gif=save_path)

    print("\nDone! Check 'simulation_output.gif' for the animation.")


if __name__ == "__main__":
    main()
