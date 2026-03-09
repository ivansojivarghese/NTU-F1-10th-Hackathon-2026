#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import csv
import math
import numpy as np

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


def load_waypoints(csv_path):
    """
    Load waypoints from CSV file.
    Assumes x is column 0, y is column 1.
    """
    waypoints = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            x = float(row[0])
            y = float(row[1])
            waypoints.append([x, y])
    waypoints = np.array(waypoints)
    return waypoints[::-1]


class PurePursuit(Node):

    def __init__(self):
        super().__init__('pure_pursuit')

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter('waypoint_csv', '')
        self.declare_parameter('lookahead_distance', 2.5)
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('speed', 4.0)

        # Use hardcoded path if launch file doesn't provide one
        csv_from_launch = self.get_parameter('waypoint_csv').value
        self.waypoint_csv = csv_from_launch if csv_from_launch != '' else \
            'Nuerburgring_centerline.csv'

        self.wheelbase = self.get_parameter('wheelbase').value
        self.speed = 12.9
        self.lookahead_distance = 2.5
        

        # ----------------------------
        # Load waypoints
        # ----------------------------
        self.waypoints = load_waypoints(self.waypoint_csv)
        self.num_waypoints = len(self.waypoints)

        # self.get_logger().info(
        #     f'Loaded {self.num_waypoints} waypoints'
        # )

        # ----------------------------
        # ROS Interfaces
        # ----------------------------
        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )

    # ----------------------------
    # Helper functions
    # ----------------------------
    def get_yaw_from_quaternion(self, q):
        """
        Convert quaternion to yaw angle.
        """
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def find_lookahead_point(self, position,closest_idx, yaw):
        """
        Find the first waypoint at least lookahead_distance away.
        """

        for i in range(1, self.num_waypoints):
            idx = (closest_idx + i) % self.num_waypoints
            wp = self.waypoints[idx]

            dx = wp[0] - position[0]
            dy = wp[1] - position[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < self.lookahead_distance:
                continue

            # Check if waypoint is in front of the car (yaw_error < 90 deg)
            path_yaw = math.atan2(dy, dx)
            yaw_error = self.normalize_angle(path_yaw - yaw)

            if abs(yaw_error) < math.pi / 2:
                return wp

        # Fallback: return closest + some offset
        fallback_idx = (closest_idx + 5) % self.num_waypoints
        return self.waypoints[fallback_idx]
      


    def pure_pursuit_steering(self, position, yaw, target):
        """
        Compute steering angle using Pure Pursuit.
        """
        dx = target[0] - position[0]
        dy = target[1] - position[1]

        # Transform to vehicle frame
        local_x = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        local_y = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        if local_x <= 0.0:
            return 0.0

        curvature = 2.0 * local_y / (local_x**2 + local_y**2)
        steering_angle = math.atan(self.wheelbase * curvature)

        return steering_angle

    # ----------------------------
    # Odometry callback
    # ----------------------------
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        position = np.array([x, y])

        yaw = self.get_yaw_from_quaternion(msg.pose.pose.orientation)

        closest_idx = np.argmin(np.linalg.norm(self.waypoints - position, axis=1))
        if not hasattr(self, '_last_idx'):
            self._last_idx = 0
        # Handle wrap-around: only block backward jumps that aren't lap completions
        diff = closest_idx - self._last_idx
        wrap_threshold = self.num_waypoints // 2  # if jump > half track, it's a wrap

        if diff < 0 and abs(diff) < wrap_threshold:
            # Genuine backward jump (not a lap wrap) — reject it
            closest_idx = self._last_idx
        else:
            self._last_idx = closest_idx

        if np.linalg.norm(position) < 0.5:
            self._last_idx = 0
        # Compute track heading at closest waypoint to detect spawn misalignment
        next_idx = (closest_idx + 1) % self.num_waypoints
        track_vec = self.waypoints[next_idx] - self.waypoints[closest_idx]
        track_yaw = math.atan2(track_vec[1], track_vec[0])
        heading_error = self.normalize_angle(track_yaw - yaw)

        drive_msg = AckermannDriveStamped()

        if abs(heading_error) > math.radians(50):
            drive_msg.drive.steering_angle = math.copysign(math.radians(20), heading_error)
            drive_msg.drive.speed = 1.0  # crawl until roughly aligned
            
            # self.get_logger().info(
            #     f'ALIGNING: pos=({position[0]:.2f},{position[1]:.2f}) '
            #     f'yaw={math.degrees(yaw):.1f}deg '
            #     f'track_yaw={math.degrees(track_yaw):.1f}deg '
            #     f'heading_error={math.degrees(heading_error):.1f}deg'
            # )

        else:
            # Normal pure pursuit operation
            lookahead_point = self.find_lookahead_point(position, closest_idx, yaw)
            steering = self.pure_pursuit_steering(position, yaw, lookahead_point)

            # Adaptive speed: slow down proportional to steering angle
            # helps prevent spinouts on tight corners
            max_steer = math.radians(5)  # typical ackermann max steer
            steer_ratio = min(abs(steering) / max_steer, 1.0)
            adaptive_speed = self.speed * (1.0 - 0.58 * steer_ratio)
            adaptive_speed = max(adaptive_speed, 4.0)

            drive_msg.drive.steering_angle = steering
            drive_msg.drive.speed = adaptive_speed

            # self.get_logger().info(
            #     f'pos=({position[0]:.2f},{position[1]:.2f}) '
            #     f'target=({lookahead_point[0]:.2f},{lookahead_point[1]:.2f}) '
            #     f'yaw={yaw:.2f} '
            #     f'steer={math.degrees(steering):.1f}deg '
            #     f'speed={adaptive_speed:.2f} '
            #     f'closest_idx={closest_idx}'
            # )
        # Temporarily add to odom_callback in YOUR node (pure_pursuit):
        finish_mid = np.array([-1.33, -0.62])
        dist_to_finish = np.linalg.norm(position - finish_mid)
        #if dist_to_finish < 0.5:
            #self.get_logger().info(f'NEAR FINISH LINE: dist={dist_to_finish:.3f}')
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
