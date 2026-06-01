#!/usr/bin/env python3

import math

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray


class UmbrellaGameController(object):
    def __init__(self):
        self.pursuer_pub = rospy.Publisher('/umbrella/pursuer/control', Twist, queue_size=10)
        self.evader_pub = rospy.Publisher('/umbrella/evader/control', Twist, queue_size=10)
        self.metrics_pub = rospy.Publisher('/umbrella/game_metrics', Float64MultiArray, queue_size=10)
        rospy.Subscriber('/umbrella/model_states', ModelStates, self.state_callback)

        self.rate_hz = rospy.get_param('~rate_hz', 30.0)
        self.period = rospy.get_param('~period', 36.0)
        self.start_theta = rospy.get_param('~start_theta', 0.18)
        self.phase_gap = rospy.get_param('~phase_gap', 0.58)
        self.horizon_steps = rospy.get_param('~horizon_steps', 20)
        self.horizon_dt = rospy.get_param('~horizon_dt', 0.11)

        self.outer_radius = rospy.get_param('~outer_radius', 3.75)
        self.inner_radius = rospy.get_param('~inner_radius', 0.62)
        self.lobe_count = rospy.get_param('~lobe_count', 8)
        self.lobe_sharpness = rospy.get_param('~lobe_sharpness', 1.55)

        self.pursuer_max_accel = rospy.get_param('~pursuer_max_accel', 7.8)
        self.evader_max_accel = rospy.get_param('~evader_max_accel', 5.6)
        self.track_kp = rospy.get_param('~track_kp', 3.45)
        self.track_kd = rospy.get_param('~track_kd', 2.05)
        self.pursuit_gain = rospy.get_param('~pursuit_gain', 2.15)
        self.evasion_gain = rospy.get_param('~evasion_gain', 1.65)
        self.safe_distance = rospy.get_param('~safe_distance', 1.25)
        self.catchup_distance = rospy.get_param('~catchup_distance', 1.45)
        self.catchup_gain = rospy.get_param('~catchup_gain', 2.30)

        self.start_time = rospy.get_time()
        self.states = {
            'pursuer_umbrella': {'pos': self.umbrella_reference(self.start_theta)[0], 'vel': (0.0, 0.0)},
            'evader_umbrella': {
                'pos': self.umbrella_reference(self.start_theta + self.phase_gap)[0],
                'vel': (0.0, 0.0),
            },
        }

    def umbrella_point(self, theta):
        wave = 0.5 + 0.5 * math.cos(float(self.lobe_count) * theta)
        radius = self.inner_radius + (self.outer_radius - self.inner_radius) * (max(0.0, wave) ** self.lobe_sharpness)
        return radius * math.cos(theta), radius * math.sin(theta)

    def umbrella_reference(self, theta):
        delta = 0.004
        pos = self.umbrella_point(theta)
        ahead = self.umbrella_point(theta + delta)
        behind = self.umbrella_point(theta - delta)
        omega = 2.0 * math.pi / self.period
        vel = (
            (ahead[0] - behind[0]) / (2.0 * delta) * omega,
            (ahead[1] - behind[1]) / (2.0 * delta) * omega,
        )
        return pos, vel

    def reference_turn_demand(self, theta):
        delta = 0.012
        before = self.umbrella_point(theta - delta)
        current = self.umbrella_point(theta)
        after = self.umbrella_point(theta + delta)
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

    def saturate(self, control, limit):
        norm = math.hypot(control[0], control[1])
        if norm <= limit or norm == 0.0:
            return control
        scale = limit / norm
        return control[0] * scale, control[1] * scale

    def horizon_game_bias(self, theta, pursuer, evader):
        pursuer_bias = (0.0, 0.0)
        evader_bias = (0.0, 0.0)
        p_pred = dict(pursuer)
        e_pred = dict(evader)

        for step in range(1, self.horizon_steps + 1):
            discount = 1.0 / (1.0 + 0.11 * step)
            future_theta = theta + (2.0 * math.pi / self.period) * self.horizon_dt * step
            p_ref, p_ref_vel = self.umbrella_reference(future_theta)
            e_ref, e_ref_vel = self.umbrella_reference(future_theta + self.phase_gap)

            p_track = self.tracking_control(p_pred, p_ref, p_ref_vel)
            e_track = self.tracking_control(e_pred, e_ref, e_ref_vel)

            dx = e_pred['pos'][0] - p_pred['pos'][0]
            dy = e_pred['pos'][1] - p_pred['pos'][1]
            distance = max(math.hypot(dx, dy), 0.001)
            direction = (dx / distance, dy / distance)

            p_game = self.scale(direction, self.pursuit_gain * discount)
            safe_pressure = max(0.0, self.safe_distance - distance) / self.safe_distance
            e_game = self.scale(direction, self.evasion_gain * safe_pressure * discount)

            pursuer_bias = self.add(pursuer_bias, p_game)
            evader_bias = self.add(evader_bias, e_game)

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
        p_track = (pursuer['pos'][0] - p_ref[0]) ** 2 + (pursuer['pos'][1] - p_ref[1]) ** 2
        e_track = (evader['pos'][0] - e_ref[0]) ** 2 + (evader['pos'][1] - e_ref[1]) ** 2
        p_effort = p_control[0] ** 2 + p_control[1] ** 2
        e_effort = e_control[0] ** 2 + e_control[1] ** 2

        pursuer_cost = 5.0 * p_track + 0.85 * distance * distance + 0.045 * p_effort
        evader_cost = 5.0 * e_track + 1.75 / (distance * distance + 0.08) + 0.055 * e_effort
        return pursuer_cost, evader_cost, distance, self.reference_turn_demand(theta)

    def game_controls(self):
        elapsed = rospy.get_time() - self.start_time
        theta = self.start_theta + 2.0 * math.pi * elapsed / self.period
        p_ref, p_ref_vel = self.umbrella_reference(theta)
        e_ref, e_ref_vel = self.umbrella_reference(theta + self.phase_gap)

        pursuer = self.states['pursuer_umbrella']
        evader = self.states['evader_umbrella']

        p_control = self.tracking_control(pursuer, p_ref, p_ref_vel)
        e_control = self.tracking_control(evader, e_ref, e_ref_vel)
        p_bias, e_bias = self.horizon_game_bias(theta, pursuer, evader)

        dx = evader['pos'][0] - pursuer['pos'][0]
        dy = evader['pos'][1] - pursuer['pos'][1]
        distance = max(math.hypot(dx, dy), 0.001)
        direction = (dx / distance, dy / distance)
        catchup_pressure = min(1.0, max(0.0, distance - self.catchup_distance) / self.catchup_distance)
        catchup = self.scale(direction, self.catchup_gain * catchup_pressure)

        p_control = self.saturate(self.add(self.add(p_control, p_bias), catchup), self.pursuer_max_accel)
        e_control = self.saturate(self.add(e_control, e_bias), self.evader_max_accel)
        metrics = self.compute_costs(theta, pursuer, evader, p_ref, e_ref, p_control, e_control)
        return p_control, e_control, metrics

    @staticmethod
    def make_twist(control):
        msg = Twist()
        msg.linear.x = control[0]
        msg.angular.z = control[1]
        return msg

    def publish_metrics(self, metrics, p_control, e_control):
        pursuer_cost, evader_cost, distance, turn_demand = metrics
        msg = Float64MultiArray()
        msg.data = [
            distance,
            pursuer_cost,
            evader_cost,
            math.hypot(p_control[0], p_control[1]),
            math.hypot(e_control[0], e_control[1]),
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
        rospy.init_node('controller_umbrella', anonymous=True)
        UmbrellaGameController().spin()
    except rospy.ROSInterruptException:
        pass
