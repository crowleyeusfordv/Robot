#!/usr/bin/env python3

import math

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


OBSTACLES = [
    {'x': -1.15, 'y': 0.85, 'radius': 0.32},
    {'x': 1.25, 'y': -0.75, 'radius': 0.34},
    {'x': 0.20, 'y': -2.05, 'radius': 0.30},
]


class ComplexGameController(object):
    def __init__(self):
        self.pursuer_pub = rospy.Publisher('/complex/pursuer/control', Twist, queue_size=10)
        self.evader_pub = rospy.Publisher('/complex/evader/control', Twist, queue_size=10)
        self.metrics_pub = rospy.Publisher('/complex/game_metrics', Float64MultiArray, queue_size=10)
        rospy.Subscriber('/complex/model_states', ModelStates, self.state_callback)

        self.rate_hz = rospy.get_param('~rate_hz', 30.0)
        self.start_theta = rospy.get_param('~start_theta', 0.32)
        self.period = rospy.get_param('~period', 28.0)
        self.phase_gap = rospy.get_param('~phase_gap', 0.78)
        self.horizon_steps = rospy.get_param('~horizon_steps', 18)
        self.horizon_dt = rospy.get_param('~horizon_dt', 0.12)

        self.pursuer_max_accel = rospy.get_param('~pursuer_max_accel', 7.2)
        self.evader_max_accel = rospy.get_param('~evader_max_accel', 5.0)
        self.track_kp = rospy.get_param('~track_kp', 3.15)
        self.track_kd = rospy.get_param('~track_kd', 1.95)
        self.pursuit_gain = rospy.get_param('~pursuit_gain', 1.85)
        self.evasion_gain = rospy.get_param('~evasion_gain', 1.45)
        self.safe_distance = rospy.get_param('~safe_distance', 1.45)
        self.catchup_distance = rospy.get_param('~catchup_distance', 1.65)
        self.catchup_gain = rospy.get_param('~catchup_gain', 2.25)
        self.initial_boost_duration = rospy.get_param('~initial_boost_duration', 4.0)
        self.initial_boost_gain = rospy.get_param('~initial_boost_gain', 1.10)
        self.obstacle_influence = rospy.get_param('~obstacle_influence', 0.95)
        self.obstacle_gain = rospy.get_param('~obstacle_gain', 1.65)
        self.ripple_count = rospy.get_param('~ripple_count', 9)
        self.ripple_amplitude = rospy.get_param('~ripple_amplitude', 0.13)

        self.start_time = rospy.get_time()
        self.states = {
            'pursuer_complex': {'pos': self.heart_reference(self.start_theta)[0], 'vel': (0.0, 0.0)},
            'evader_complex': {'pos': self.heart_reference(self.start_theta + self.phase_gap)[0], 'vel': (0.0, 0.0)},
        }

    @staticmethod
    def base_heart_point(theta):
        x = 16.0 * math.sin(theta) ** 3
        y = (
            13.0 * math.cos(theta)
            - 5.0 * math.cos(2.0 * theta)
            - 2.0 * math.cos(3.0 * theta)
            - math.cos(4.0 * theta)
        )
        scale = 0.22
        return scale * x, scale * y + 0.55

    def rippled_heart_point(self, theta):
        base = self.base_heart_point(theta)
        delta = 0.001
        ahead = self.base_heart_point(theta + delta)
        behind = self.base_heart_point(theta - delta)
        tangent = (ahead[0] - behind[0], ahead[1] - behind[1])
        tangent_norm = math.hypot(tangent[0], tangent[1])
        if tangent_norm < 1e-6:
            return base

        normal = (-tangent[1] / tangent_norm, tangent[0] / tangent_norm)
        ripple = self.ripple_amplitude * math.sin(float(self.ripple_count) * theta)
        return base[0] + ripple * normal[0], base[1] + ripple * normal[1]

    def heart_reference(self, theta):
        delta = 0.006
        pos = self.rippled_heart_point(theta)
        ahead = self.rippled_heart_point(theta + delta)
        behind = self.rippled_heart_point(theta - delta)
        omega = 2.0 * math.pi / self.period
        vel = (
            (ahead[0] - behind[0]) / (2.0 * delta) * omega,
            (ahead[1] - behind[1]) / (2.0 * delta) * omega,
        )
        return pos, vel

    def reference_turn_demand(self, theta):
        delta = 0.015
        before = self.rippled_heart_point(theta - delta)
        current = self.rippled_heart_point(theta)
        after = self.rippled_heart_point(theta + delta)
        heading_1 = math.atan2(current[1] - before[1], current[0] - before[0])
        heading_2 = math.atan2(after[1] - current[1], after[0] - current[0])
        change = math.atan2(math.sin(heading_2 - heading_1), math.cos(heading_2 - heading_1))
        return abs(change) / delta

    def state_callback(self, msg):
        for index, name in enumerate(msg.name):
            if name not in self.states:
                continue
            pose = msg.pose[index]
            twist = msg.twist[index]
            self.states[name] = {
                'pos': (pose.position.x, pose.position.y),
                'vel': (twist.linear.x, twist.linear.y),
            }

    def tracking_control(self, state, ref_pos, ref_vel):
        px, py = state['pos']
        vx, vy = state['vel']
        return (
            self.track_kp * (ref_pos[0] - px) + self.track_kd * (ref_vel[0] - vx),
            self.track_kp * (ref_pos[1] - py) + self.track_kd * (ref_vel[1] - vy),
        )

    def obstacle_barrier(self, pos):
        bias_x = 0.0
        bias_y = 0.0
        cost = 0.0
        closest_margin = 99.0

        for obstacle in OBSTACLES:
            dx = pos[0] - obstacle['x']
            dy = pos[1] - obstacle['y']
            distance = max(math.hypot(dx, dy), 0.001)
            margin = distance - obstacle['radius']
            closest_margin = min(closest_margin, margin)
            influence_margin = self.obstacle_influence - margin
            if influence_margin <= 0.0:
                continue

            direction = (dx / distance, dy / distance)
            strength = self.obstacle_gain * (influence_margin / self.obstacle_influence) ** 2
            bias_x += strength * direction[0]
            bias_y += strength * direction[1]
            cost += strength * strength

        return (bias_x, bias_y), cost, closest_margin

    def saturate(self, control, limit):
        norm = math.hypot(control[0], control[1])
        if norm <= limit or norm == 0.0:
            return control
        scale = limit / norm
        return control[0] * scale, control[1] * scale

    @staticmethod
    def add(a, b):
        return a[0] + b[0], a[1] + b[1]

    @staticmethod
    def scale(a, gain):
        return a[0] * gain, a[1] * gain

    @staticmethod
    def predict_step(state, control, dt):
        return {
            'pos': (
                state['pos'][0] + state['vel'][0] * dt + 0.5 * control[0] * dt * dt,
                state['pos'][1] + state['vel'][1] * dt + 0.5 * control[1] * dt * dt,
            ),
            'vel': (
                state['vel'][0] + control[0] * dt,
                state['vel'][1] + control[1] * dt,
            ),
        }

    def horizon_game_bias(self, theta, pursuer, evader):
        pursuer_bias = (0.0, 0.0)
        evader_bias = (0.0, 0.0)
        p_pred = dict(pursuer)
        e_pred = dict(evader)

        for step in range(1, self.horizon_steps + 1):
            discount = 1.0 / (1.0 + 0.12 * step)
            future_theta = theta + (2.0 * math.pi / self.period) * self.horizon_dt * step
            p_ref, p_ref_vel = self.heart_reference(future_theta)
            e_ref, e_ref_vel = self.heart_reference(future_theta + self.phase_gap)

            p_track = self.tracking_control(p_pred, p_ref, p_ref_vel)
            e_track = self.tracking_control(e_pred, e_ref, e_ref_vel)

            dx = e_pred['pos'][0] - p_pred['pos'][0]
            dy = e_pred['pos'][1] - p_pred['pos'][1]
            distance = max(math.hypot(dx, dy), 0.001)
            direction = (dx / distance, dy / distance)

            p_game = self.scale(direction, self.pursuit_gain * discount)
            safe_pressure = max(0.0, self.safe_distance - distance) / self.safe_distance
            e_game = self.scale(direction, self.evasion_gain * safe_pressure * discount)

            p_obstacle, _, _ = self.obstacle_barrier(p_pred['pos'])
            e_obstacle, _, _ = self.obstacle_barrier(e_pred['pos'])

            pursuer_bias = self.add(pursuer_bias, self.add(p_game, self.scale(p_obstacle, discount)))
            evader_bias = self.add(evader_bias, self.add(e_game, self.scale(e_obstacle, discount)))

            p_pred = self.predict_step(
                p_pred,
                self.saturate(self.add(p_track, p_game), self.pursuer_max_accel),
                self.horizon_dt,
            )
            e_pred = self.predict_step(
                e_pred,
                self.saturate(self.add(e_track, e_game), self.evader_max_accel),
                self.horizon_dt,
            )

        normalizer = 1.0 / float(max(self.horizon_steps, 1))
        return self.scale(pursuer_bias, normalizer), self.scale(evader_bias, normalizer)

    def compute_costs(self, theta, pursuer, evader, p_ref, e_ref, p_control, e_control):
        dx = evader['pos'][0] - pursuer['pos'][0]
        dy = evader['pos'][1] - pursuer['pos'][1]
        distance = max(math.hypot(dx, dy), 0.001)
        _, p_obstacle_cost, p_margin = self.obstacle_barrier(pursuer['pos'])
        _, e_obstacle_cost, e_margin = self.obstacle_barrier(evader['pos'])

        p_track = (pursuer['pos'][0] - p_ref[0]) ** 2 + (pursuer['pos'][1] - p_ref[1]) ** 2
        e_track = (evader['pos'][0] - e_ref[0]) ** 2 + (evader['pos'][1] - e_ref[1]) ** 2
        p_effort = p_control[0] ** 2 + p_control[1] ** 2
        e_effort = e_control[0] ** 2 + e_control[1] ** 2

        pursuer_cost = 4.0 * p_track + 0.75 * distance * distance + 0.06 * p_effort + p_obstacle_cost
        evader_cost = 4.0 * e_track + 1.6 / (distance * distance + 0.08) + 0.06 * e_effort + e_obstacle_cost
        return pursuer_cost, evader_cost, distance, min(p_margin, e_margin), self.reference_turn_demand(theta)

    def game_controls(self):
        elapsed = rospy.get_time() - self.start_time
        theta = self.start_theta + 2.0 * math.pi * elapsed / self.period
        p_ref, p_ref_vel = self.heart_reference(theta)
        e_ref, e_ref_vel = self.heart_reference(theta + self.phase_gap)

        pursuer = self.states['pursuer_complex']
        evader = self.states['evader_complex']

        p_control = self.tracking_control(pursuer, p_ref, p_ref_vel)
        e_control = self.tracking_control(evader, e_ref, e_ref_vel)
        p_bias, e_bias = self.horizon_game_bias(theta, pursuer, evader)
        p_obstacle, _, _ = self.obstacle_barrier(pursuer['pos'])
        e_obstacle, _, _ = self.obstacle_barrier(evader['pos'])

        dx = evader['pos'][0] - pursuer['pos'][0]
        dy = evader['pos'][1] - pursuer['pos'][1]
        distance = max(math.hypot(dx, dy), 0.001)
        direction = (dx / distance, dy / distance)
        catchup_pressure = max(0.0, distance - self.catchup_distance) / self.catchup_distance
        catchup_pressure = min(1.0, catchup_pressure)
        initial_boost = max(0.0, 1.0 - elapsed / self.initial_boost_duration)
        catchup = self.scale(
            direction,
            self.catchup_gain * catchup_pressure + self.initial_boost_gain * initial_boost,
        )

        p_control = self.saturate(
            self.add(self.add(self.add(p_control, p_bias), p_obstacle), catchup),
            self.pursuer_max_accel,
        )
        e_control = self.saturate(
            self.add(self.add(e_control, e_bias), e_obstacle),
            self.evader_max_accel,
        )

        metrics = self.compute_costs(theta, pursuer, evader, p_ref, e_ref, p_control, e_control)
        return p_control, e_control, metrics

    @staticmethod
    def make_twist(control):
        msg = Twist()
        msg.linear.x = control[0]
        msg.angular.z = control[1]
        return msg

    def publish_metrics(self, metrics, p_control, e_control):
        pursuer_cost, evader_cost, distance, closest_margin, turn_demand = metrics
        msg = Float64MultiArray()
        msg.data = [
            distance,
            pursuer_cost,
            evader_cost,
            math.hypot(p_control[0], p_control[1]),
            math.hypot(e_control[0], e_control[1]),
            closest_margin,
            turn_demand,
        ]
        self.metrics_pub.publish(msg)

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            p_control, e_control, metrics = self.game_controls()
            self.pursuer_pub.publish(self.make_twist(p_control))
            self.evader_pub.publish(self.make_twist(e_control))
            self.publish_metrics(metrics, p_control, e_control)
            rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('controller_complex', anonymous=True)
        ComplexGameController().spin()
    except rospy.ROSInterruptException:
        pass
