#!/usr/bin/env python3

import math

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist


class HeartPursuitEvasionController(object):
    def __init__(self):
        self.pursuer_pub = rospy.Publisher('/pursuer/control', Twist, queue_size=10)
        self.evader_pub = rospy.Publisher('/evader/control', Twist, queue_size=10)
        rospy.Subscriber('/pursuit_evasion/model_states', ModelStates, self.state_callback)

        self.rate_hz = rospy.get_param('~rate_hz', 30.0)
        self.period = rospy.get_param('~period', 36.0)
        self.phase_gap = rospy.get_param('~phase_gap', 0.62)
        self.max_accel = rospy.get_param('~max_accel', 4.0)
        self.min_separation = rospy.get_param('~min_separation', 0.85)

        self.track_kp = rospy.get_param('~track_kp', 2.1)
        self.track_kd = rospy.get_param('~track_kd', 1.35)
        self.pursuit_gain = rospy.get_param('~pursuit_gain', 0.75)
        self.evasion_gain = rospy.get_param('~evasion_gain', 0.95)

        self.start_time = rospy.get_time()
        self.states = {
            'pursuer_object': {'pos': self.heart_reference(0.0)[0], 'vel': (0.0, 0.0)},
            'evader_object': {'pos': self.heart_reference(self.phase_gap)[0], 'vel': (0.0, 0.0)},
        }

    def heart_reference(self, theta):
        x = 16.0 * math.sin(theta) ** 3
        y = (
            13.0 * math.cos(theta)
            - 5.0 * math.cos(2.0 * theta)
            - 2.0 * math.cos(3.0 * theta)
            - math.cos(4.0 * theta)
        )
        dx = 48.0 * math.sin(theta) ** 2 * math.cos(theta)
        dy = (
            -13.0 * math.sin(theta)
            + 10.0 * math.sin(2.0 * theta)
            + 6.0 * math.sin(3.0 * theta)
            + 4.0 * math.sin(4.0 * theta)
        )

        scale = 0.22
        omega = 2.0 * math.pi / self.period
        pos = (scale * x, scale * y + 0.55)
        vel = (scale * dx * omega, scale * dy * omega)
        return pos, vel

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

    def saturate(self, vector):
        norm = math.hypot(vector[0], vector[1])
        if norm <= self.max_accel or norm == 0.0:
            return vector
        scale = self.max_accel / norm
        return vector[0] * scale, vector[1] * scale

    def tracking_control(self, state, ref_pos, ref_vel):
        px, py = state['pos']
        vx, vy = state['vel']
        return (
            self.track_kp * (ref_pos[0] - px) + self.track_kd * (ref_vel[0] - vx),
            self.track_kp * (ref_pos[1] - py) + self.track_kd * (ref_vel[1] - vy),
        )

    def game_controls(self):
        elapsed = rospy.get_time() - self.start_time
        theta = 2.0 * math.pi * elapsed / self.period

        pursuer_ref_pos, pursuer_ref_vel = self.heart_reference(theta)
        evader_ref_pos, evader_ref_vel = self.heart_reference(theta + self.phase_gap)

        pursuer = self.states['pursuer_object']
        evader = self.states['evader_object']

        pursuer_u = self.tracking_control(pursuer, pursuer_ref_pos, pursuer_ref_vel)
        evader_u = self.tracking_control(evader, evader_ref_pos, evader_ref_vel)

        dx = evader['pos'][0] - pursuer['pos'][0]
        dy = evader['pos'][1] - pursuer['pos'][1]
        distance = max(math.hypot(dx, dy), 0.001)
        direction = (dx / distance, dy / distance)

        pursuer_u = (
            pursuer_u[0] + self.pursuit_gain * direction[0],
            pursuer_u[1] + self.pursuit_gain * direction[1],
        )

        if distance < self.min_separation:
            pressure = (self.min_separation - distance) / self.min_separation
            evader_u = (
                evader_u[0] + self.evasion_gain * pressure * direction[0],
                evader_u[1] + self.evasion_gain * pressure * direction[1],
            )

        return self.saturate(pursuer_u), self.saturate(evader_u)

    @staticmethod
    def make_twist(control):
        msg = Twist()
        msg.linear.x = control[0]
        msg.angular.z = control[1]
        return msg

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            pursuer_u, evader_u = self.game_controls()
            self.pursuer_pub.publish(self.make_twist(pursuer_u))
            self.evader_pub.publish(self.make_twist(evader_u))
            rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('controller', anonymous=True)
        HeartPursuitEvasionController().spin()
    except rospy.ROSInterruptException:
        pass
