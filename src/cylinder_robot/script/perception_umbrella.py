#!/usr/bin/env python3

import math

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import rospkg
import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray


class UmbrellaGameRecorder(object):
    def __init__(self):
        self.start_time = rospy.get_time()
        self.outer_radius = rospy.get_param('~outer_radius', 3.75)
        self.inner_radius = rospy.get_param('~inner_radius', 0.62)
        self.lobe_count = rospy.get_param('~lobe_count', 8)
        self.lobe_sharpness = rospy.get_param('~lobe_sharpness', 1.55)
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
            'turn_demand': [],
        }
        rospy.Subscriber('/umbrella/model_states', ModelStates, self.state_callback)
        rospy.Subscriber('/umbrella/game_metrics', Float64MultiArray, self.metrics_callback)
        rospy.on_shutdown(self.visualization)

    def umbrella_point(self, theta):
        wave = 0.5 + 0.5 * math.cos(float(self.lobe_count) * theta)
        radius = self.inner_radius + (self.outer_radius - self.inner_radius) * (max(0.0, wave) ** self.lobe_sharpness)
        return radius * math.cos(theta), radius * math.sin(theta)

    def umbrella_points(self, count=900):
        xs = []
        ys = []
        for i in range(count + 1):
            theta = 2.0 * math.pi * float(i) / float(count)
            x, y = self.umbrella_point(theta)
            xs.append(x)
            ys.append(y)
        return xs, ys

    def state_callback(self, msg):
        try:
            pursuer_index = msg.name.index('pursuer_umbrella')
            evader_index = msg.name.index('evader_umbrella')
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
        self.metrics['turn_demand'].append(msg.data[5])

    def draw_sector_guides(self, ax):
        for i in range(16):
            theta = 2.0 * math.pi * float(i) / 16.0
            x = self.outer_radius * math.cos(theta)
            y = self.outer_radius * math.sin(theta)
            ax.plot([0.0, x], [0.0, y], color='gray', alpha=0.12, linewidth=0.8)

    def visualization(self):
        if len(self.samples['time']) < 2:
            rospy.logwarn('Not enough samples to save fig_umbrella.png')
            return

        ref_x, ref_y = self.umbrella_points()
        fig = plt.figure(figsize=(16, 10))
        ax_path = fig.add_subplot(2, 2, 1)
        ax_distance = fig.add_subplot(2, 2, 2)
        ax_cost = fig.add_subplot(2, 2, 3)
        ax_control = fig.add_subplot(2, 2, 4)

        self.draw_sector_guides(ax_path)
        ax_path.plot(ref_x, ref_y, 'k--', linewidth=1.2, label='umbrella reference')
        ax_path.plot(self.samples['pursuer_x'], self.samples['pursuer_y'], label='pursuer')
        ax_path.plot(self.samples['evader_x'], self.samples['evader_y'], label='evader')
        ax_path.scatter([0.0], [0.0], s=35, color='black', label='center')
        ax_path.set_title('umbrella-like pursuit-evasion trajectory')
        ax_path.set_xlabel('x')
        ax_path.set_ylabel('y')
        ax_path.set_aspect('equal', adjustable='box')
        ax_path.grid(True)
        ax_path.legend(loc='best')

        if self.metrics['time']:
            ax_distance.plot(self.metrics['time'], self.metrics['distance'], label='separation')
            ax_distance.set_title('separation')
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
        fig_path = rospkg.RosPack().get_path('cylinder_robot') + '/fig_umbrella.png'
        fig.savefig(fig_path, dpi=130)
        plt.close(fig)
        rospy.loginfo('Saved umbrella pursuit-evasion plot to %s', fig_path)


if __name__ == '__main__':
    try:
        rospy.init_node('perception_umbrella', anonymous=True)
        UmbrellaGameRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
