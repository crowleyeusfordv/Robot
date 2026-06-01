#!/usr/bin/env python3

import math

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose, Quaternion, Twist


class VehicleState(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0


class ComplexGameDriver(object):
    def __init__(self):
        self.rate_hz = rospy.get_param('~rate_hz', 70.0)
        self.mass = rospy.get_param('~mass', 1.0)
        self.damping = rospy.get_param('~damping', 0.20)
        self.pursuer_max_speed = rospy.get_param('~pursuer_max_speed', 3.65)
        self.evader_max_speed = rospy.get_param('~evader_max_speed', 2.65)
        self.z_height = rospy.get_param('~z_height', 0.12)

        self.states = {
            'pursuer_complex': VehicleState(0.08, 2.08),
            'evader_complex': VehicleState(2.47, 2.94),
        }

        self.set_model_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=20)
        self.state_pub = rospy.Publisher('/complex/model_states', ModelStates, queue_size=10)

        rospy.Subscriber('/complex/pursuer/control', Twist, self.pursuer_callback)
        rospy.Subscriber('/complex/evader/control', Twist, self.evader_callback)

        self.last_time = rospy.get_time()
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.step)

    def pursuer_callback(self, msg):
        self.set_control('pursuer_complex', msg)

    def evader_callback(self, msg):
        self.set_control('evader_complex', msg)

    def set_control(self, name, msg):
        state = self.states[name]
        state.ax = msg.linear.x / self.mass
        state.ay = msg.angular.z / self.mass

    def step(self, _event):
        now = rospy.get_time()
        dt = max(0.0, min(now - self.last_time, 0.07))
        self.last_time = now

        for name, state in self.states.items():
            self.integrate(name, state, dt)
            self.keep_inside_room(state)
            self.publish_model_state(name, state)
        self.publish_combined_state()

    def integrate(self, name, state, dt):
        if dt <= 0.0:
            return
        state.vx += (state.ax - self.damping * state.vx) * dt
        state.vy += (state.ay - self.damping * state.vy) * dt

        speed = math.hypot(state.vx, state.vy)
        max_speed = self.pursuer_max_speed if name == 'pursuer_complex' else self.evader_max_speed
        if speed > max_speed:
            scale = max_speed / speed
            state.vx *= scale
            state.vy *= scale

        state.x += state.vx * dt
        state.y += state.vy * dt

    @staticmethod
    def keep_inside_room(state):
        limit = 4.65
        if state.x < -limit or state.x > limit:
            state.x = max(-limit, min(limit, state.x))
            state.vx *= -0.25
        if state.y < -limit or state.y > limit:
            state.y = max(-limit, min(limit, state.y))
            state.vy *= -0.25

    def publish_model_state(self, name, state):
        msg = ModelState()
        msg.model_name = name
        msg.reference_frame = 'world'
        msg.pose = self.pose_from_state(state)
        msg.twist.linear.x = state.vx
        msg.twist.linear.y = state.vy
        self.set_model_pub.publish(msg)

    def publish_combined_state(self):
        msg = ModelStates()
        for name in ('pursuer_complex', 'evader_complex'):
            state = self.states[name]
            msg.name.append(name)
            msg.pose.append(self.pose_from_state(state))
            twist = Twist()
            twist.linear.x = state.vx
            twist.linear.y = state.vy
            msg.twist.append(twist)
        self.state_pub.publish(msg)

    def pose_from_state(self, state):
        pose = Pose()
        pose.position.x = state.x
        pose.position.y = state.y
        pose.position.z = self.z_height
        yaw = math.atan2(state.vy, state.vx) if math.hypot(state.vx, state.vy) > 0.02 else 0.0
        pose.orientation = self.quaternion_from_yaw(yaw)
        return pose

    @staticmethod
    def quaternion_from_yaw(yaw):
        quat = Quaternion()
        quat.z = math.sin(yaw / 2.0)
        quat.w = math.cos(yaw / 2.0)
        return quat


if __name__ == '__main__':
    try:
        rospy.init_node('driver_complex', anonymous=True)
        ComplexGameDriver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
