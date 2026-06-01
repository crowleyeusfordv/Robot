#!/usr/bin/env python3

import math

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import rospkg
import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray


OBSTACLES = [
    {'x': -1.15, 'y': 0.85, 'radius': 0.32},
    {'x': 1.25, 'y': -0.75, 'radius': 0.34},
    {'x': 0.20, 'y': -2.05, 'radius': 0.30},
]


class ComplexGameRecorder(object):
    def __init__(self):
        self.start_time = rospy.get_time()
        self.samples = {
            'time': [],
            'pursuer_x': [],
            'pursuer_y': [],
            'evader_x': [],
            'evader_y': [],
        }
        self.metrics = {
            'time': [],
            'distance': [],
            'pursuer_cost': [],
            'evader_cost': [],
            'pursuer_u': [],
            'evader_u': [],
            'obstacle_margin': [],
            'turn_demand': [],
        }
        rospy.Subscriber('/complex/model_states', ModelStates, self.state_callback)
        rospy.Subscriber('/complex/game_metrics', Float64MultiArray, self.metrics_callback)
        rospy.on_shutdown(self.visualization)

    def state_callback(self, msg):
        try:
            pursuer_index = msg.name.index('pursuer_complex')
            evader_index = msg.name.index('evader_complex')
        except ValueError:
            return

        pursuer = msg.pose[pursuer_index].position
        evader = msg.pose[evader_index].position
        self.samples['time'].append(rospy.get_time() - self.start_time)
        self.samples['pursuer_x'].append(pursuer.x)
        self.samples['pursuer_y'].append(pursuer.y)
        self.samples['evader_x'].append(evader.x)
        self.samples['evader_y'].append(evader.y)

    def metrics_callback(self, msg):
        if len(msg.data) < 6:
            return
        self.metrics['time'].append(rospy.get_time() - self.start_time)
        self.metrics['distance'].append(msg.data[0])
        self.metrics['pursuer_cost'].append(msg.data[1])
        self.metrics['evader_cost'].append(msg.data[2])
        self.metrics['pursuer_u'].append(msg.data[3])
        self.metrics['evader_u'].append(msg.data[4])
        self.metrics['obstacle_margin'].append(msg.data[5])
        self.metrics['turn_demand'].append(msg.data[6] if len(msg.data) > 6 else 0.0)

    @staticmethod
    def base_heart_point(theta):
        x = 16.0 * math.sin(theta) ** 3
        y = (
            13.0 * math.cos(theta)
            - 5.0 * math.cos(2.0 * theta)
            - 2.0 * math.cos(3.0 * theta)
            - math.cos(4.0 * theta)
        )
        return 0.22 * x, 0.22 * y + 0.55

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
        ripple = 0.13 * math.sin(9.0 * theta)
        return base[0] + ripple * normal[0], base[1] + ripple * normal[1]

    def heart_points(self, count=700):
        xs = []
        ys = []
        for i in range(count + 1):
            theta = 2.0 * math.pi * float(i) / float(count)
            x, y = self.rippled_heart_point(theta)
            xs.append(x)
            ys.append(y)
        return xs, ys

    def draw_obstacles(self, ax):
        for obstacle in OBSTACLES:
            circle = plt.Circle(
                (obstacle['x'], obstacle['y']),
                obstacle['radius'],
                color='dimgray',
                alpha=0.75,
            )
            halo = plt.Circle(
                (obstacle['x'], obstacle['y']),
                obstacle['radius'] + 0.95,
                color='gray',
                alpha=0.10,
                linestyle='--',
                fill=False,
            )
            ax.add_patch(circle)
            ax.add_patch(halo)

    def visualization(self):
        if len(self.samples['time']) < 2:
            rospy.logwarn('Not enough samples to save fig_complex.png')
            return

        ref_x, ref_y = self.heart_points()
        fig = plt.figure(figsize=(16, 10))
        ax_path = fig.add_subplot(2, 2, 1)
        ax_distance = fig.add_subplot(2, 2, 2)
        ax_cost = fig.add_subplot(2, 2, 3)
        ax_control = fig.add_subplot(2, 2, 4)

        ax_path.plot(ref_x, ref_y, 'k--', linewidth=1.1, label='heart reference')
        ax_path.plot(self.samples['pursuer_x'], self.samples['pursuer_y'], label='pursuer')
        ax_path.plot(self.samples['evader_x'], self.samples['evader_y'], label='evader')
        self.draw_obstacles(ax_path)
        ax_path.set_title('complex pursuit-evasion heart game')
        ax_path.set_xlabel('x')
        ax_path.set_ylabel('y')
        ax_path.set_aspect('equal', adjustable='box')
        ax_path.grid(True)
        ax_path.legend(loc='best')

        if self.metrics['time']:
            ax_distance.plot(self.metrics['time'], self.metrics['distance'], label='separation')
            ax_distance.plot(self.metrics['time'], self.metrics['obstacle_margin'], label='closest obstacle margin')
            ax_distance.set_title('safety signals')
            ax_distance.set_xlabel('time [s]')
            ax_distance.grid(True)
            ax_distance.legend(loc='best')

            ax_cost.plot(self.metrics['time'], self.metrics['pursuer_cost'], label='pursuer cost')
            ax_cost.plot(self.metrics['time'], self.metrics['evader_cost'], label='evader cost')
            ax_cost.set_title('noncooperative game costs')
            ax_cost.set_xlabel('time [s]')
            ax_cost.grid(True)
            ax_cost.legend(loc='best')

            ax_control.plot(self.metrics['time'], self.metrics['pursuer_u'], label='|u_p|')
            ax_control.plot(self.metrics['time'], self.metrics['evader_u'], label='|u_e|')
            ax_control.plot(self.metrics['time'], self.metrics['turn_demand'], label='turn demand')
            ax_control.set_title('control effort and direction changes')
            ax_control.set_xlabel('time [s]')
            ax_control.grid(True)
            ax_control.legend(loc='best')

        fig.tight_layout()
        fig_path = rospkg.RosPack().get_path('cylinder_robot') + '/fig_complex.png'
        fig.savefig(fig_path, dpi=130)
        plt.close(fig)
        rospy.loginfo('Saved complex pursuit-evasion plot to %s', fig_path)


if __name__ == '__main__':
    try:
        rospy.init_node('perception_complex', anonymous=True)
        ComplexGameRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
