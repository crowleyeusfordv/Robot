#!/usr/bin/env python3

import math

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import rospkg
import rospy
from gazebo_msgs.msg import ModelStates


class TrajectoryRecorder(object):
    def __init__(self):
        self.start_time = rospy.get_time()
        self.samples = {
            'time': [],
            'pursuer_x': [],
            'pursuer_y': [],
            'evader_x': [],
            'evader_y': [],
            'distance': [],
        }
        rospy.Subscriber('/pursuit_evasion/model_states', ModelStates, self.state_callback)
        rospy.on_shutdown(self.visualization)

    def state_callback(self, msg):
        try:
            pursuer_index = msg.name.index('pursuer_object')
            evader_index = msg.name.index('evader_object')
        except ValueError:
            return

        pursuer = msg.pose[pursuer_index].position
        evader = msg.pose[evader_index].position
        elapsed = rospy.get_time() - self.start_time
        distance = math.hypot(evader.x - pursuer.x, evader.y - pursuer.y)

        self.samples['time'].append(elapsed)
        self.samples['pursuer_x'].append(pursuer.x)
        self.samples['pursuer_y'].append(pursuer.y)
        self.samples['evader_x'].append(evader.x)
        self.samples['evader_y'].append(evader.y)
        self.samples['distance'].append(distance)

    @staticmethod
    def heart_points(count=400):
        xs = []
        ys = []
        for i in range(count + 1):
            theta = 2.0 * math.pi * float(i) / float(count)
            x = 16.0 * math.sin(theta) ** 3
            y = (
                13.0 * math.cos(theta)
                - 5.0 * math.cos(2.0 * theta)
                - 2.0 * math.cos(3.0 * theta)
                - math.cos(4.0 * theta)
            )
            xs.append(0.22 * x)
            ys.append(0.22 * y + 0.55)
        return xs, ys

    def visualization(self):
        if len(self.samples['time']) < 2:
            rospy.logwarn('Not enough trajectory samples to save fig_x.png')
            return

        ref_x, ref_y = self.heart_points()
        fig = plt.figure(figsize=(16, 9))
        ax_path = fig.add_subplot(1, 2, 1)
        ax_dist = fig.add_subplot(1, 2, 2)

        ax_path.plot(ref_x, ref_y, 'k--', linewidth=1.2, label='heart reference')
        ax_path.plot(self.samples['pursuer_x'], self.samples['pursuer_y'], label='pursuer')
        ax_path.plot(self.samples['evader_x'], self.samples['evader_y'], label='evader')
        ax_path.scatter(self.samples['pursuer_x'][0], self.samples['pursuer_y'][0], s=35, label='pursuer start')
        ax_path.scatter(self.samples['evader_x'][0], self.samples['evader_y'][0], s=35, label='evader start')
        ax_path.set_title('pursuit-evasion heart trajectory')
        ax_path.set_xlabel('x')
        ax_path.set_ylabel('y')
        ax_path.set_aspect('equal', adjustable='box')
        ax_path.grid(True)
        ax_path.legend(loc='best')

        ax_dist.plot(self.samples['time'], self.samples['distance'])
        ax_dist.set_title('separation over time')
        ax_dist.set_xlabel('time [s]')
        ax_dist.set_ylabel('distance')
        ax_dist.grid(True)

        fig.tight_layout()
        fig_path = rospkg.RosPack().get_path('cylinder_robot') + '/fig_x.png'
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        rospy.loginfo('Saved pursuit-evasion trajectory plot to %s', fig_path)


if __name__ == '__main__':
    try:
        rospy.init_node('perception', anonymous=True)
        TrajectoryRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
