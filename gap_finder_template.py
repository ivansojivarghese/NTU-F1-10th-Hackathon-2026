#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker, MarkerArray

class GapFollower(Node):
    def __init__(self):
        super().__init__('gap_follower')
 
        self.declare_parameter('lookahead_distance', 5.0) 
        self.declare_parameter('robot_width', 0.2032) # 0.2032
        self.declare_parameter('obstacle_bubble_radius', 0.30) 
        self.declare_parameter('disparity_threshold', 0.2) 
        self.declare_parameter('max_speed', 8.0) #10.0 
        self.declare_parameter('min_speed', 1.0)
        self.declare_parameter('max_steering', 0.34) 
        self.declare_parameter('consecutive_valid_gap', 5)
        self.declare_parameter('steering_gain', 0.4) 
        self.declare_parameter('speed_gain', 1.0) # 1.0
        self.declare_parameter('field_of_vision', np.pi/2)
        self.declare_parameter('lidarscan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')
            
        self.lookahead_distance     = self.get_parameter('lookahead_distance').value
        self.robot_width            = self.get_parameter('robot_width').value
        self.obstacle_bubble_radius = self.get_parameter('obstacle_bubble_radius').value
        self.disparity_threshold    = self.get_parameter('disparity_threshold').value
        self.max_speed              = self.get_parameter('max_speed').value
        self.min_speed              = self.get_parameter('min_speed').value
        self.max_steering           = self.get_parameter('max_steering').value
        self.consecutive_valid_gap  = self.get_parameter('consecutive_valid_gap').value
        self.steering_gain          = self.get_parameter('steering_gain').value
        self.speed_gain             = self.get_parameter('speed_gain').value
        self.field_of_vision        = self.get_parameter('field_of_vision').value
        self.lidar_scan_topic       = self.get_parameter('lidarscan_topic').value
        self.drive_topic            = self.get_parameter('drive_topic').value
            
        self.subscriber = self.create_subscription(LaserScan, 
                                                   self.lidar_scan_topic,
                                                   self.lidar_callback,
                                                   10)
        
        # Subscribe to odometry for position and velocity tracking
        self.odom_subscriber = self.create_subscription(Odometry,
                                                        '/ego_racecar/odom',
                                                        self.odom_callback,
                                                        10)
        
        self.publisher = self.create_publisher(AckermannDriveStamped,
                                                self.drive_topic,
                                                10)
        
        # Variables to store odometry data
        self.current_position = None  # (x, y) tuple
        self.prev_position = None     # Previous (x, y) for distance calculation
        self.actual_velocity = 0.0    # m/s (magnitude)
        self.velocity_x = 0.0         # m/s (forward velocity)
        self.velocity_y = 0.0         # m/s (lateral velocity)
        self.total_distance = 0.0     # Odometer: distance since last lap reset (m)
        
        # Lap tracking
        self.start_position = None    # Recorded on first odom reading
        self.lap_count = 0            # Number of completed laps
        self.last_lap_distance = None # ODO distance of the most recently completed lap
        self.lap_start_time = None    # Timestamp when current lap started
        self.LAP_TRIGGER_RADIUS = 1.5 # metres - how close to start counts as crossing
        self.MIN_LAP_DISTANCE = 30.0  # metres - minimum travel before lap can trigger (avoids false reset at start)
        
        # DRS zone detection variables
        self.position_history = []    # Store (x, y, steering, timestamp)
        self.current_steering = 0.0
        self.drs_zones_detected = []  # Store detected DRS zones
        self.in_straight = False
        self.straight_start_idx = None
        self.STEERING_THRESHOLD     = 0.08  # radians (~4.6 degrees) - DRS straight detection
        self.TURN_DETECT_THRESHOLD   = 0.03  # radians (~1.7 degrees) - turn/bend logging (more sensitive)
        self.MIN_STRAIGHT_READINGS = 30  # ~1.5 seconds at 20Hz
        self.MIN_STRAIGHT_LENGTH = self.min_speed * 3  # min_speed * 3 meters
        
        # DRS System - ON/OFF Switch
        self.DRS_ENABLED = True  # Set to True to enable DRS speed boost, False for normal operation
        
        self.DRS_ZONES = []  # Populated dynamically by detect_drs_zones() after first lap
        self.DRS_BOOST_SPEED = 15.0      # Speed limit applied inside auto-detected DRS zones (m/s)
        self.drs_braking_distance = 8.0  # Default braking distance applied when exiting a DRS zone (m)
        self.drs_spool_distance = self.max_speed   # Distance over which speed ramps up on zone entry (m)
        self.base_lookahead = self.lookahead_distance  # Store original lookahead for DRS scaling
        self.base_fov = self.field_of_vision              # Store original FOV half-angle
        self.drs_fov  = min(2*np.pi/3, self.base_fov * 1.33)  # DRS FOV half-angle capped at 2π/3
        self._drs_log_counter = 0  # Throttle DRS lookahead log to ~1Hz

        # Turn logging (warm-up lap only)
        self.turn_log       = []    # List of dicts: one entry per turn detected in lap 1
        self.in_turn        = False
        self.turn_start_idx = None

        # Predictive corner braking (lap 2+)
        self.CORNER_MU    = 0.7   # Friction coefficient (indoor carpet/sim)
        self.CORNER_DECEL = 5.0   # Assumed deceleration capability (m/s²)

        # DRS zone transition tracking (entry/exit log events)
        self._prev_drs_zone_idx = None
        self._drs_active_steer_buf = []  # Steering history for current DRS zone pass
        self._drs_zone_incidents   = {}  # {zone_name: bool} — incident flag from last pass
        self._drs_zone_frozen      = set()  # Zone names permanently frozen after an incident

        # Base race speed — mirrors max_speed (master control)
        self._base_race_speed = self.max_speed

        self.bubble_viz_publisher = self.create_publisher(MarkerArray, "/safety_bubble", 10)
        self.scan_viz_publisher = self.create_publisher(MarkerArray, "/scan_msg", 10)
        self.gap_viz_publisher = self.create_publisher(Marker, "/goal_point", 10)
        self.dispa_viz_publisher = self.create_publisher(MarkerArray, "/disparity_points", 10)

    def odom_callback(self, msg):
        """ Process odometry data to track actual position and velocity
        """
        # Extract position (x, y coordinates on the map)
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        # Extract velocity components
        self.velocity_x = msg.twist.twist.linear.x  # Forward velocity
        self.velocity_y = msg.twist.twist.linear.y  # Lateral velocity
        
        # Calculate actual speed (magnitude of velocity vector)
        self.actual_velocity = np.sqrt(self.velocity_x**2 + self.velocity_y**2)
        
        # Record starting position on first reading
        if self.start_position is None:
            self.start_position = self.current_position
            self.lap_start_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(
                f'START POSITION recorded: ({self.start_position[0]:.2f}, {self.start_position[1]:.2f})'
            )
        
        # Odometer: accumulate distance travelled
        if self.prev_position is not None:
            dx = self.current_position[0] - self.prev_position[0]
            dy = self.current_position[1] - self.prev_position[1]
            self.total_distance += np.sqrt(dx**2 + dy**2)
        self.prev_position = self.current_position
        
        # Lap detection: check if car has returned near start position
        if self.start_position is not None and self.total_distance > self.MIN_LAP_DISTANCE:
            dist_to_start = np.sqrt(
                (self.current_position[0] - self.start_position[0])**2 +
                (self.current_position[1] - self.start_position[1])**2
            )
            if dist_to_start < self.LAP_TRIGGER_RADIUS:
                self.lap_count += 1
                self.last_lap_distance = self.total_distance  # Save before reset
                now = self.get_clock().now().nanoseconds / 1e9
                lap_time = now - self.lap_start_time if self.lap_start_time is not None else 0.0
                self.lap_start_time = now  # Reset lap timer for next lap
                self.get_logger().info(
                    f'LAP {self.lap_count} COMPLETE | '
                    f'Lap time: {lap_time:.2f} s | '
                    f'Lap distance: {self.total_distance:.2f} m | '
                    f'Speed: {self.actual_velocity:.2f} m/s'
                )
                # Refresh DRS speed limits using this lap's steering data
                # (collected at actual race speed → more accurate turn radii)
                self._refresh_drs_speed_limits(self.lap_count - 1)
                self.total_distance = 0.0  # Reset odometer for next lap
        
        # Record position and steering for DRS zone detection
        if self.current_position is not None:
            timestamp = self.get_clock().now().nanoseconds / 1e9
            self.position_history.append({
                'x': self.current_position[0],
                'y': self.current_position[1],
                'steering': abs(self.current_steering),
                'timestamp': timestamp,
                'index': len(self.position_history),
                'odo': self.total_distance,
                'lap': self.lap_count,  # tag for per-lap DRS refresh
            })
            
            # Detect DRS zones
            self.detect_drs_zones()
    
    def _check_drs_zone_incident(self, zone, steer_history):
        """Detect a wiggle/incident during a DRS zone pass.

        Two criteria — either triggers an incident:
          • >4 normal reversals  (|steer| ≥ 0.05 rad) — persistent oscillation
          • >2 big reversals     (|steer| ≥ 0.15 rad) — violent wall contact

        On detection: immediately cuts zone speed_limit -1 m/s (floor 8.0 m/s)
        and flags it so _refresh_drs_speed_limits skips the +1 m/s ramp this lap.
        """
        WIGGLE_STEER_MIN = 0.05   # rad — noise floor
        BIG_STEER_MIN    = 0.15   # rad — large correction threshold
        NORMAL_REV_MIN   = 4      # >4 normal reversals
        BIG_REV_MIN      = 2      # >2 big reversals

        normal_sig = [s for s in steer_history if abs(s) >= WIGGLE_STEER_MIN]
        big_sig    = [s for s in steer_history if abs(s) >= BIG_STEER_MIN]

        normal_reversals = sum(
            1 for i in range(1, len(normal_sig))
            if normal_sig[i] * normal_sig[i - 1] < 0
        )
        big_reversals = sum(
            1 for i in range(1, len(big_sig))
            if big_sig[i] * big_sig[i - 1] < 0
        )

        incident = normal_reversals > NORMAL_REV_MIN or big_reversals > BIG_REV_MIN
        reversals = normal_reversals  # for logging
        if incident:
            zone_name = zone['name']
            self._drs_zone_incidents[zone_name] = True
            # Immediately cut speed for the remainder of the current pass
            old_limit = zone['speed_limit']
            new_limit = round(max(old_limit - 0.2, self._base_race_speed), 1)
            zone['speed_limit'] = new_limit
            self.get_logger().info(
                f'DRS INCIDENT [{zone_name}]: '
                f'normal_rev={normal_reversals} big_rev={big_reversals} — '
                f'speed cut immediately {old_limit:.1f} → {new_limit:.1f} m/s'
            )

    def _refresh_drs_speed_limits(self, completed_lap):
        """Update each DRS zone's speed limit after a completed lap.

        - Incident detected (wiggle/near-collision in the zone): -1 m/s, floor 8.0 m/s, freeze.
        - No incident: +1 m/s ramp up to DRS_BOOST_SPEED.

        Initial speed limits are set by calculate_drs_params (physics-derived).
        """
        if not self.DRS_ZONES:
            return

        STEP = 0.2  # m/s increment per clean lap

        for zone in self.DRS_ZONES:
            old_limit = zone['speed_limit']
            had_incident = self._drs_zone_incidents.pop(zone['name'], False)

            if had_incident:
                # Penalise: reduce by 1 m/s, never below base race speed, freeze from future ramps
                new_limit = round(max(old_limit - 0.2, self._base_race_speed), 1)
                self._drs_zone_frozen.add(zone['name'])
                if new_limit < old_limit:
                    zone['speed_limit'] = new_limit
                self.get_logger().info(
                    f'DRS PENALISE [{zone["name"]}] lap {completed_lap + 1}: '
                    f'{old_limit:.1f} → {new_limit:.1f} m/s (incident — frozen)'
                )
            elif zone['name'] in self._drs_zone_frozen:
                pass  # Frozen after prior incident — no further increments
            else:
                # Clean lap: ramp up +1 m/s toward DRS_BOOST_SPEED
                new_limit = round(min(old_limit + STEP, self.DRS_BOOST_SPEED), 1)
                if new_limit > old_limit:
                    zone['speed_limit'] = new_limit
                    self.get_logger().info(
                        f'DRS RAMP [{zone["name"]}] lap {completed_lap + 1}: '
                        f'{old_limit:.1f} → {new_limit:.1f} m/s'
                        + (' (BOOST CAP)' if new_limit >= self.DRS_BOOST_SPEED else '')
                    )

    def check_drs_active(self):
        """Check if current ODO distance is inside any DRS zone and return (zone_index, zone_info)"""
        if not self.DRS_ENABLED:
            return None, None  # DRS disabled
        if self.lap_count < 1:
            return None, None  # Warm-up lap: detect zones but don't boost yet
        
        for idx, zone in enumerate(self.DRS_ZONES):
            if zone['from_m'] <= self.total_distance <= zone['to_m']:
                return idx, zone
        return None, None
    
    def calculate_distance_to_zone_exit(self, zone):
        """Calculate remaining ODO distance until end of DRS zone"""
        if zone is None:
            return float('inf')
        
        return max(0.0, zone['to_m'] - self.total_distance)

    def detect_drs_zones(self):
        """ Detect DRS zones by analyzing straight sections """
        if len(self.position_history) < self.MIN_STRAIGHT_READINGS:
            return
        
        current_idx = len(self.position_history) - 1
        current_data = self.position_history[current_idx]
        
        # DRS straight detection (uses STEERING_THRESHOLD = 0.08 rad)
        if current_data['steering'] < self.STEERING_THRESHOLD:
            if not self.in_straight:
                self.in_straight = True
                self.straight_start_idx = current_idx
        else:
            if self.in_straight:
                self.in_straight = False
                self.analyze_straight_section(self.straight_start_idx, current_idx - 1)
                self.straight_start_idx = None

        # Turn logging (uses TURN_DETECT_THRESHOLD = 0.03 rad - catches bends too)
        if current_data['steering'] >= self.TURN_DETECT_THRESHOLD:
            if not self.in_turn:
                self.in_turn = True
                self.turn_start_idx = current_idx
        else:
            if self.in_turn:
                self.in_turn = False
                self._log_turn(self.turn_start_idx, current_idx - 1)
                self.turn_start_idx = None
    
    def _log_turn(self, start_idx, end_idx):
        """Record turn data during the warm-up lap. Skipped on lap 2+."""
        if self.lap_count >= 1:
            return  # Only log during warm-up
        if start_idx is None or end_idx is None or end_idx <= start_idx:
            return

        turn_data = self.position_history[start_idx:end_idx + 1]
        if len(turn_data) < 3:
            return

        steerings   = [d['steering'] for d in turn_data]
        peak_steer  = max(steerings)
        mean_steer  = float(np.mean(steerings))
        from_m      = turn_data[0].get('odo', 0.0)
        to_m        = turn_data[-1].get('odo', 0.0)

        # Ignore micro-wiggles: require at least 0.5 m and peak steer above turn detect threshold
        if (to_m - from_m) < 0.5 or peak_steer < self.TURN_DETECT_THRESHOLD:
            return

        length_m    = round(to_m - from_m, 2)
        mid_m       = round((from_m + to_m) / 2.0, 2)  # ODO midpoint of entire turn

        # Apex: index of peak steering within this turn
        apex_local  = int(np.argmax(steerings))
        apex_data   = turn_data[apex_local]
        apex_m      = round(apex_data.get('odo', mid_m), 2)  # ODO at tightest point

        # XY positions
        entry_x     = round(turn_data[0].get('x', 0.0), 3)
        entry_y     = round(turn_data[0].get('y', 0.0), 3)
        exit_x      = round(turn_data[-1].get('x', 0.0), 3)
        exit_y      = round(turn_data[-1].get('y', 0.0), 3)
        apex_x      = round(apex_data.get('x', 0.0), 3)
        apex_y      = round(apex_data.get('y', 0.0), 3)

        WHEELBASE   = 0.3302
        turn_radius = round(WHEELBASE / np.tan(max(peak_steer, 0.01)), 2)
        turn_number = len(self.turn_log) + 1

        turn = {
            'turn_number':       turn_number,
            # ODO span
            'from_m':            round(from_m, 2),
            'to_m':              round(to_m, 2),
            'length_m':          length_m,
            'mid_m':             mid_m,   # ODO midpoint
            'apex_m':            apex_m,  # ODO at tightest steering point
            # XY positions
            'entry_xy':          (entry_x, entry_y),
            'exit_xy':           (exit_x,  exit_y),
            'apex_xy':           (apex_x,  apex_y),
            # Steering
            'peak_steering_rad': round(peak_steer, 4),
            'mean_steering_rad': round(mean_steer, 4),
            'turn_radius_m':     turn_radius,
            'readings':          len(turn_data),
        }
        self.turn_log.append(turn)
        self.get_logger().info(
            f'TURN {turn_number} LOGGED | '
            f'ODO {from_m:.1f}m -> {to_m:.1f}m  length={length_m:.1f}m  '
            f'mid={mid_m:.1f}m  apex={apex_m:.1f}m | '
            f'entry=({entry_x:.2f},{entry_y:.2f})  apex=({apex_x:.2f},{apex_y:.2f})  exit=({exit_x:.2f},{exit_y:.2f}) | '
            f'peak steer: {peak_steer:.3f} rad  mean: {mean_steer:.3f} rad  radius: {turn_radius:.1f}m | '
            f'readings: {len(turn_data)}'
        )

    def calculate_drs_params(self, end_idx, straight_length):
        """Dynamically calculate safe DRS speed limit and braking distance.

        Two-branch decision:
          1. Guaranteed straight (strict gate) → full DRS_BOOST_SPEED, no cap.
          2. Bending DRS zone               → physics cap: v = sqrt(μgR) × SAFETY.

        This prevents the physics formula from incorrectly capping true straights
        (where peak_exit_steering ≈ 0 → tiny radius → very low v_corner).
        """
        # Sample the next 25 readings (~1.25s at 20Hz) after end of straight
        CORNER_WINDOW = 25
        post_straight = self.position_history[end_idx + 1 : end_idx + 1 + CORNER_WINDOW]

        if len(post_straight) == 0:
            # No data yet (straight ends at current position) - use safe defaults
            return self.DRS_BOOST_SPEED * 0.75, self.drs_braking_distance

        # Peak absolute steering in the exit corner window
        peak_exit_steering = max(d['steering'] for d in post_straight)

        # --- Guaranteed straight gate (strict) ---
        # Both conditions must hold: steering tiny AND sustained for long enough.
        # Only real straights pass; sweepers and chicanes are caught.
        STRAIGHT_STEER_THRESH = 0.04   # rad ≈ 2.3° — strict cutoff
        STRAIGHT_CONFIRM_N    = 30     # ~1.5 s @ 20 Hz — must persist, not just a spike
        is_guaranteed_straight = (
            peak_exit_steering < STRAIGHT_STEER_THRESH
            and len(post_straight) >= STRAIGHT_CONFIRM_N
        )

        # Confidence metric (diagnostic / logging only)
        straight_confidence = float(np.exp(-peak_exit_steering / 0.03))

        # --- Turn radius: geometric estimate from exit corner steering ---
        WHEELBASE   = 0.3302
        turn_radius = round(WHEELBASE / np.tan(max(peak_exit_steering, 0.01)), 2)

        # --- Speed limit ---
        if is_guaranteed_straight:
            # Full DRS — no physics cap. The zone is verified flat.
            speed_limit = self.DRS_BOOST_SPEED
        else:
            # Bending DRS zone — physics-limited.
            # v_corner = sqrt(mu * g * R); 15% safety margin for latency/model error.
            SAFETY      = 0.85
            v_corner    = np.sqrt(self.CORNER_MU * 9.81 * turn_radius)
            speed_limit = min(self.DRS_BOOST_SPEED, SAFETY * v_corner)

        # Never below current base race speed
        speed_limit = round(max(speed_limit, self._base_race_speed), 1)

        # --- Braking distance: proportional to speed delta and corner sharpness ---
        speed_delta      = speed_limit - self._base_race_speed
        sharpness_scale  = 1.0 + (peak_exit_steering / self.max_steering) * 2.0
        braking_distance = round(max(0.0, speed_delta * sharpness_scale * 2.0), 1)

        # Safe lookahead cap for this corner
        corner_safe_lookahead = round(max(self.lookahead_distance, turn_radius * 4.0), 2)

        drs_type = 'STRAIGHT (full)' if is_guaranteed_straight else f'BENDING (capped)'
        self.get_logger().info(
            f'  DRS params [{drs_type}]: '
            f'peak_exit_steer={peak_exit_steering:.3f} rad | '
            f'confidence={straight_confidence:.2f} | '
            f'turn_radius={turn_radius:.1f} m | '
            f'speed_limit={speed_limit:.1f} m/s | '
            f'braking_dist={braking_distance:.1f} m | '
            f'corner_safe_la={corner_safe_lookahead:.1f} m'
        )
        return speed_limit, braking_distance, turn_radius, corner_safe_lookahead

    def analyze_straight_section(self, start_idx, end_idx):
        """ Analyze a detected straight section and log if it's a DRS zone """
        num_readings = end_idx - start_idx + 1
        
        if num_readings < self.MIN_STRAIGHT_READINGS:
            return  # Too short in time
        
        straight_data = self.position_history[start_idx:end_idx+1]
        x_coords = [d['x'] for d in straight_data]
        y_coords = [d['y'] for d in straight_data]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        start_pos = (straight_data[0]['x'], straight_data[0]['y'])
        end_pos = (straight_data[-1]['x'], straight_data[-1]['y'])
        length = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        if length < self.MIN_STRAIGHT_LENGTH:
            return  # Too short in distance
        
        new_from = straight_data[0].get('odo', 0.0)
        new_to   = straight_data[-1].get('odo', 0.0)

        # Detect ODO wrap: lap reset happened mid-straight
        if new_to < new_from:
            split_idx = next(
                (i for i in range(len(straight_data) - 1)
                 if straight_data[i]['odo'] > straight_data[i + 1]['odo']),
                None
            )
            if split_idx is not None:
                self.analyze_straight_section(start_idx, start_idx + split_idx)
                self.analyze_straight_section(start_idx + split_idx + 1, end_idx)
            return

        # Check if this is a start/finish tail
        START_FINISH_THRESHOLD = 15.0
        is_start_finish_tail = (
            self.last_lap_distance is not None
            and len(self.drs_zones_detected) >= 1
            and self.drs_zones_detected[0]['from_m'] < START_FINISH_THRESHOLD
            and new_to > self.last_lap_distance - START_FINISH_THRESHOLD
        )

        if (new_to - new_from) < 30.0 and not is_start_finish_tail:
            return  # ODO length < 30 m - not long enough to be a DRS zone

        for zone in self.drs_zones_detected:
            if new_from <= zone['to_m'] and zone['from_m'] <= new_to:
                return  # ODO ranges overlap - already detected this zone
        
        # New DRS zone detected!
        from_m = new_from
        to_m   = new_to
        zone_number = len(self.drs_zones_detected) + 1
        speed_limit, braking_distance, turn_radius, corner_safe_lookahead = self.calculate_drs_params(end_idx, length)
        zone = {
            'name': f'Auto-Zone {zone_number}',
            'from_m': from_m,
            'to_m': to_m,
            'dist_m': to_m - from_m,
            'speed_limit': speed_limit,
            'braking_distance': braking_distance,
            'turn_radius': turn_radius,
            'corner_safe_lookahead': corner_safe_lookahead,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'length': length,
            'start': start_pos,
            'end': end_pos,
            'readings': num_readings
        }
        self.drs_zones_detected.append(zone)
        self.DRS_ZONES.append(zone)  # Wire into active DRS activation system

        # Check if first and last zones are two halves of a single straight crossing start/finish
        if is_start_finish_tail:
            zone1 = self.drs_zones_detected[0]
            combined_speed = max(zone1['speed_limit'], zone['speed_limit'])
            zone1['speed_limit']        = combined_speed
            zone['speed_limit']         = combined_speed
            zone1['from_m']             = 0.0
            zone['braking_distance']    = 0.0
            zone1['start_finish_pair']  = True
            zone['start_finish_pair']   = True
            self.get_logger().info(
                f'START/FINISH PAIR: "{zone1["name"]}" (ODO 0.0m -> {zone1["to_m"]:.1f}m) + '
                f'"{zone["name"]}" (ODO {from_m:.1f}m -> {to_m:.1f}m) | '
                f'Combined speed limit: {combined_speed:.1f} m/s | No braking between halves'
            )

        self.get_logger().info(
            f'DRS ZONE {zone_number} ACTIVATED: "{zone["name"]}" | '
            f'ODO {from_m:.1f}m -> {to_m:.1f}m  ({zone["dist_m"]:.1f}m ODO  {zone["length"]:.1f}m physical) | '
            f'speed_limit={zone["speed_limit"]:.1f} m/s  braking_dist={zone["braking_distance"]:.1f}m | '
            f'turn_radius={zone["turn_radius"]:.1f}m  corner_safe_la={zone["corner_safe_lookahead"]:.1f}m | '
            f'start=({zone["start"][0]:.2f},{zone["start"][1]:.2f})  '
            f'end=({zone["end"][0]:.2f},{zone["end"][1]:.2f}) | '
            f'readings={zone["readings"]}'
        )

    def preprocess_lidar(self,data):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        # convert the FOV into indices
        ranges = np.array(data.ranges)
        fov_indices = int(self.field_of_vision / data.angle_increment)
        center_index = int(abs(data.angle_min) / data.angle_increment)  # // will return float, e.g. 540.0 if input is float value
        fov_start_i = center_index - fov_indices
        fov_end_i = center_index + fov_indices

        # handle edge cases
        fov_start_i = max(0, fov_start_i)
        fov_end_i = min(len(ranges)-1, fov_end_i)

        limited_ranges = np.copy(ranges)  # Alt: Truncate the ranges array
        limited_ranges[:fov_start_i] = 0
        limited_ranges[fov_end_i+1:] = 0

        # find nearest obstacle distance & index
        nearest_obstacle_distance = np.min(limited_ranges[limited_ranges > 0])
        nearest_obstacle_distance_indices = np.where(limited_ranges == nearest_obstacle_distance)

        # find the angle and indices span of the obstacle bubble 
        obs_bubble_half_angle = np.arctan(self.obstacle_bubble_radius / nearest_obstacle_distance)
        obs_bubble_index_extension = int(np.ceil(obs_bubble_half_angle / data.angle_increment))

        obs_bubble_start_i = np.maximum(0, nearest_obstacle_distance_indices[0] - obs_bubble_index_extension)
        obs_bubble_end_i = np.minimum(len(limited_ranges)-1, nearest_obstacle_distance_indices[0] + obs_bubble_index_extension)
        valid_limited_ranges = np.copy(limited_ranges)

        # draw safety bubble
        for start, end in zip(obs_bubble_start_i, obs_bubble_end_i):
            valid_limited_ranges[start:end] = 0

        proc_ranges = np.copy(valid_limited_ranges)

        # find bubble_coord
        nearest_obstacle_angle = data.angle_min + nearest_obstacle_distance_indices[0] * data.angle_increment
        obs_x = nearest_obstacle_distance * np.cos(nearest_obstacle_angle)
        obs_y = nearest_obstacle_distance * np.sin(nearest_obstacle_angle)

        # Turns [[x1, x2], [y1, y2]] into [[x1, y1], [x2, y2]]
        obs_bubble_coord = np.array([obs_x,obs_y]).T

        return proc_ranges, obs_bubble_coord, fov_start_i, fov_end_i

    def find_max_gap(self, data, proc_ranges):
        """ Return the start index & end index of the max gap in proc_ranges
        """
        # find the furthest point in the processed ranges array
        max_dist = np.max(proc_ranges)
        max_dist_indices = np.where(proc_ranges == max_dist)[0]

        if len(max_dist_indices) > 1:
            # Center-weighted average: biases toward straight ahead (scan midpoint)
            # prevents the midpoint jumping left/right when a symmetric plateau is
            # nibbled from alternating sides by disparity bubbles (root cause of wiggle)
            center = len(proc_ranges) // 2
            weights = 1.0 / (np.abs(max_dist_indices - center) + 1)
            max_dist_index = int(np.round(np.average(max_dist_indices, weights=weights)))
        else:
            max_dist_index = max_dist_indices[0]
        
        # find goal coord
        goal_distance = proc_ranges[max_dist_index]
        goal_angle = data.angle_min + max_dist_index * data.angle_increment
        goal_x = goal_distance * np.cos(goal_angle)
        goal_y = goal_distance * np.sin(goal_angle)
        goal_coord = np.array([goal_x, goal_y])
    
        return max_dist_index, goal_coord

    def disparity_extender(self, data):
        proc_ranges, obs_bubble_coord, fov_start_i, fov_end_i = self.preprocess_lidar(data)

        # Zero out diffs where either neighbour is artificial (FOV boundary, obstacle bubble).
        # This eliminates fake disparities at the FOV edge and obstacle bubble edges in one pass —
        # both neighbours must be real readings (> 0.01 m) for a diff to count as a disparity.
        valid = proc_ranges > 0.01
        range_diffs = np.abs(np.diff(proc_ranges))
        range_diffs[~(valid[:-1] & valid[1:])] = 0
        disparity_indices = np.where(range_diffs > self.disparity_threshold)[0]

        if len(disparity_indices) == 0:
            proc_ranges = np.where(proc_ranges > self.lookahead_distance, self.lookahead_distance, proc_ranges)
            return proc_ranges, obs_bubble_coord, fov_start_i, fov_end_i, []

        # find disparity coord (for visualisation)
        disparity_angles = data.angle_min + disparity_indices * data.angle_increment
        disparity_x = proc_ranges[disparity_indices] * np.cos(disparity_angles)
        disparity_y = proc_ranges[disparity_indices] * np.sin(disparity_angles)
        disparity_coord = np.column_stack((disparity_x, disparity_y))

        # Article-correct disparity extension:
        # For each disparity, identify closer vs farther point.
        # Extend ONLY toward the farther side, overwriting with the closer distance.
        # "Do not overwrite any points that are already closer."
        # This keeps the wall visible at its true distance instead of zeroing it out.
        for i in disparity_indices:
            p1 = proc_ranges[i]      # left of disparity
            p2 = proc_ranges[i + 1]  # right of disparity

            if p1 == 0.0 or p2 == 0.0:
                continue

            closer_dist = min(p1, p2)

            # Lookahead-scaled width: at base lookahead (5m) → robot_width unchanged.
            # At longer DRS lookahead the goal point is further away, so the same angular
            # bubble covers less lateral distance — scale up to compensate.
            # Capped at ×1.8 to prevent consuming all available gap at very long lookaheads.
            width_scale     = float(np.clip(self.lookahead_distance / self.base_lookahead, 1.0, 1.8))
            effective_width = self.robot_width * width_scale

            # Speed-aware extra samples: lateral drift = v × reaction_time.
            # Converts that physical margin into angular indices at the obstacle's distance.
            # Clamp [3, 12]: never zero at low speed; never collapses gap at high speed.
            REACTION_TIME = 0.2  # seconds (ROS2 + Python node latency estimate)
            lateral_margin = abs(self.actual_velocity) * REACTION_TIME
            extra_angle    = np.arctan(lateral_margin / closer_dist)
            extra_samples  = int(extra_angle / data.angle_increment)
            EXTRA_SAMPLES  = int(np.clip(extra_samples, 3, 12))

            num_samples = int(np.arctan(effective_width / closer_dist)
                              / data.angle_increment) + EXTRA_SAMPLES

            if p1 < p2:
                # Closer is on left (index i), extend rightward from i+1
                end = min(len(proc_ranges), i + 1 + num_samples)
                for idx in range(i + 1, end):
                    if proc_ranges[idx] > closer_dist:
                        proc_ranges[idx] = closer_dist
            else:
                # Closer is on right (index i+1), extend leftward from i
                start = max(0, i - num_samples + 1)
                for idx in range(start, i + 1):
                    if proc_ranges[idx] > closer_dist:
                        proc_ranges[idx] = closer_dist

        # cap vehicle vision
        proc_ranges = np.where(proc_ranges > self.lookahead_distance, self.lookahead_distance, proc_ranges)

        return proc_ranges, obs_bubble_coord, fov_start_i, fov_end_i, disparity_coord

    def get_ranges_coord(self, data, proc_ranges, fov_start_i ,fov_end_i):
        indices = np.arange(len(proc_ranges[fov_start_i:fov_end_i+1]))
        angles = -np.pi/2 + (indices * data.angle_increment)

        x_coords = proc_ranges[fov_start_i:fov_end_i+1] * np.cos(angles)
        y_coords = proc_ranges[fov_start_i:fov_end_i+1] * np.sin(angles)

        ranges_coord = np.column_stack((x_coords, y_coords))
        # both .T and .column_stack work        
        return ranges_coord

    
    def visualisation_marker(self, bubble_coord, goal_coord, scan_msg_coord, dispa_coord):
        bubble_array_viz_msg = MarkerArray()
        for i, coord in enumerate(bubble_coord):
            self.bubble_viz_msg = Marker()
            self.bubble_viz_msg.header.frame_id = "ego_racecar/base_link"
            self.bubble_viz_msg.color.a = 1.0
            self.bubble_viz_msg.color.r = 1.0
            self.bubble_viz_msg.scale.x = self.obstacle_bubble_radius
            self.bubble_viz_msg.scale.y = self.obstacle_bubble_radius
            self.bubble_viz_msg.scale.z = self.obstacle_bubble_radius
            self.bubble_viz_msg.type = Marker.SPHERE
            self.bubble_viz_msg.action = Marker.ADD
            self.bubble_viz_msg.id = i
            self.bubble_viz_msg.pose.position.x = coord[0]
            self.bubble_viz_msg.pose.position.y = coord[1]
            bubble_array_viz_msg.markers.append(self.bubble_viz_msg)

        scan_array_viz_msg = MarkerArray()
        for i, coord in enumerate(scan_msg_coord):
            self.scan_viz_msg = Marker()
            self.scan_viz_msg.header.frame_id = "ego_racecar/base_link"
            self.scan_viz_msg.color.b = 0.95
            self.scan_viz_msg.color.a = 0.5
            self.scan_viz_msg.scale.x = 0.2
            self.scan_viz_msg.scale.y = 0.2
            self.scan_viz_msg.scale.z = 0.2
            self.scan_viz_msg.type = Marker.CYLINDER
            self.scan_viz_msg.action = Marker.ADD
            self.scan_viz_msg.id = i
            self.scan_viz_msg.pose.position.x = coord[0]
            self.scan_viz_msg.pose.position.y = coord[1]
            scan_array_viz_msg.markers.append(self.scan_viz_msg)

        dispa_array_viz_msg = MarkerArray()
        for i, coord in enumerate(dispa_coord):
            self.dispa_viz_msg = Marker()
            self.dispa_viz_msg.header.frame_id = "ego_racecar/base_link"
            self.dispa_viz_msg.color.r = 0.95
            self.dispa_viz_msg.color.g = 0.7
            self.dispa_viz_msg.color.b = 0.83
            self.dispa_viz_msg.color.a = 0.8
            self.dispa_viz_msg.scale.x = self.robot_width / 2
            self.dispa_viz_msg.scale.y = self.robot_width / 2
            self.dispa_viz_msg.scale.z = self.robot_width / 2
            self.dispa_viz_msg.type = Marker.SPHERE
            self.dispa_viz_msg.action = Marker.ADD
            self.dispa_viz_msg.id = i
            self.dispa_viz_msg.pose.position.x = coord[0]
            self.dispa_viz_msg.pose.position.y = coord[1]
            dispa_array_viz_msg.markers.append(self.dispa_viz_msg)

        self.goal_viz_msg = Marker()
        self.goal_viz_msg.header.frame_id = "ego_racecar/base_link"
        self.goal_viz_msg.color.r = 0.3
        self.goal_viz_msg.color.g = 0.7
        self.goal_viz_msg.color.b = 0.0
        self.goal_viz_msg.color.a = 0.8
        self.goal_viz_msg.scale.x = 0.3
        self.goal_viz_msg.scale.y = 0.3
        self.goal_viz_msg.scale.z = 0.3
        self.goal_viz_msg.type = Marker.CYLINDER
        self.goal_viz_msg.action = Marker.ADD
        self.goal_viz_msg.pose.position.x = float(goal_coord[0])
        self.goal_viz_msg.pose.position.y = float(goal_coord[1])
        
        self.bubble_viz_publisher.publish(bubble_array_viz_msg)
        self.gap_viz_publisher.publish(self.goal_viz_msg)
        self.scan_viz_publisher.publish(scan_array_viz_msg)
        self.dispa_viz_publisher.publish(dispa_array_viz_msg)
 
    def _predictive_corner_speed(self, effective_max_speed):
        """Return a predictive speed limit based on the next logged turn.

        Physics:
          v_corner  = sqrt(mu * g * turn_radius)          [max safe cornering speed]
          brake_dist = (v_now² - v_corner²) / (2 * decel)

        All turns are floored at _base_race_speed.
        Returns effective_max_speed unchanged if no braking needed.
        """
        if not self.turn_log:
            return effective_max_speed

        LOOKAHEAD_M = 25.0  # look this far ahead for upcoming turns
        candidates = [
            t for t in self.turn_log
            if t['from_m'] > self.total_distance
            and (t['from_m'] - self.total_distance) <= LOOKAHEAD_M
        ]
        # Also include the turn we're currently inside
        current_turn = next(
            (t for t in self.turn_log
             if t['from_m'] <= self.total_distance <= t['to_m']),
            None
        )
        if current_turn:
            candidates.append(current_turn)

        if not candidates:
            return effective_max_speed

        target_turn  = min(candidates, key=lambda t: t['from_m'])
        turn_radius  = target_turn['turn_radius_m']
        dist_to_turn = max(0.0, target_turn['from_m'] - self.total_distance)

        # Safe cornering speed — floor at current base race speed
        v_corner = np.sqrt(self.CORNER_MU * 9.81 * turn_radius)
        v_corner = max(self._base_race_speed, round(v_corner, 2))

        v_now = abs(self.actual_velocity) if self.actual_velocity > 0.01 else effective_max_speed

        if v_now <= v_corner:
            return effective_max_speed  # Already slow enough

        brake_dist = (v_now**2 - v_corner**2) / (2.0 * self.CORNER_DECEL)

        if dist_to_turn > brake_dist:
            return effective_max_speed  # Too far away to start braking yet

        # Linear blend: full speed at brake start → v_corner at turn entry
        blend            = 1.0 - (dist_to_turn / brake_dist) if brake_dist > 0 else 0.0
        predictive_limit = v_now - (v_now - v_corner) * blend
        predictive_limit = max(v_corner, predictive_limit)

        self._brake_target_turn  = target_turn
        self._brake_dist_to_turn = dist_to_turn
        self._brake_v_corner     = v_corner
        self._brake_limit        = predictive_limit
        return predictive_limit

    def _turn_aware_lookahead(self, effective_lookahead):
        """Scale lookahead down for sharper turns from the warm-up turn_log.

        Physics intuition:
          - Sharp turn  (small radius) → lookahead → base_lookahead (5 m)  @ 8 m/s
          - Gentle turn (large radius) → lookahead unchanged

        A proximity blend starts APPROACH_M metres before the turn entry so the
        reduction is gradual rather than a sudden step.
        Never increases lookahead - only restricts it.
        """
        if not self.turn_log or self.lap_count < 1:
            return effective_lookahead

        APPROACH_M    = 20.0  # metres before turn to begin blending
        SHARP_RADIUS  = 1.5   # radius (m) at/below which → minimum lookahead (base_lookahead)
        GENTLE_RADIUS = 6.0   # radius (m) at/above which → no lookahead reduction

        # Find the nearest upcoming turn, plus any we're currently inside
        candidates = [
            t for t in self.turn_log
            if t['from_m'] > self.total_distance
            and (t['from_m'] - self.total_distance) <= APPROACH_M
        ]
        current_turn = next(
            (t for t in self.turn_log
             if t['from_m'] <= self.total_distance <= t['to_m']),
            None
        )
        if current_turn:
            candidates.append(current_turn)
        if not candidates:
            return effective_lookahead

        target_turn  = min(candidates, key=lambda t: t['from_m'])
        turn_radius  = target_turn['turn_radius_m']
        dist_to_turn = max(0.0, target_turn['from_m'] - self.total_distance)

        # 0.0 = sharpest (use base_lookahead), 1.0 = gentlest (no reduction)
        sharpness_factor = float(np.clip(
            (turn_radius - SHARP_RADIUS) / (GENTLE_RADIUS - SHARP_RADIUS), 0.0, 1.0
        ))
        # Lookahead at turn entry for this corner's sharpness
        corner_la = self.base_lookahead + (effective_lookahead - self.base_lookahead) * sharpness_factor

        # Proximity blend: 1.0 far away (no change yet), 0.0 at turn entry (full corner_la)
        proximity_blend = float(np.clip(dist_to_turn / APPROACH_M, 0.0, 1.0))
        blended_la = corner_la + (effective_lookahead - corner_la) * proximity_blend

        return min(effective_lookahead, blended_la)  # only restrict, never expand

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        # --- DRS zone state (computed first so lookahead/FOV can be set before disparity_extender) ---
        # DRS only activates from lap 2 onwards (lap_count >= 1); lap 1 is a warm-up detection run
        zone_idx, current_drs_zone = self.check_drs_active()

        # Log zone entry and exit events once per transition
        if zone_idx != self._prev_drs_zone_idx and self.lap_count >= 1:
            if current_drs_zone is not None:
                # Zone entry — reset steering buffer for this pass
                self._drs_active_steer_buf = []
                '''
                self.get_logger().info(
                    f'>>> DRS ENTER [{current_drs_zone["name"]}] | '
                    f'ODO {current_drs_zone["from_m"]:.1f}m -> {current_drs_zone["to_m"]:.1f}m '
                    f'({current_drs_zone["dist_m"]:.1f}m) | '
                    f'target={current_drs_zone["speed_limit"]:.1f} m/s | '
                    f'braking_dist={current_drs_zone["braking_distance"]:.1f}m | '
                    f'actual={self.actual_velocity:.1f} m/s'
                )
                '''
            elif self._prev_drs_zone_idx is not None:
                prev_zone = self.DRS_ZONES[self._prev_drs_zone_idx]
                # Zone exit — analyse buffer for wiggle/incident
                self._check_drs_zone_incident(prev_zone, self._drs_active_steer_buf)
                self._drs_active_steer_buf = []
                '''
                self.get_logger().info(
                    f'<<< DRS EXIT  [{prev_zone["name"]}] | '
                    f'ODO {self.total_distance:.1f}m | '
                    f'actual={self.actual_velocity:.1f} m/s'
                )
                '''
            self._prev_drs_zone_idx = zone_idx

        if current_drs_zone is not None:
            distance_to_exit    = self.calculate_distance_to_zone_exit(current_drs_zone)
            distance_from_entry = self.total_distance - current_drs_zone['from_m']
            drs_boost_speed     = current_drs_zone.get('speed_limit', self.max_speed)
            zone_braking_distance = current_drs_zone.get('braking_distance', self.drs_braking_distance)
            # Race base speed: perturbed [8.0–9.0] m/s from lap 2 onwards, 5.0 m/s on warm-up lap
            base_speed    = self._base_race_speed if self.lap_count >= 1 else self.max_speed
            # Lookahead scales proportionally with DRS speed limit
            drs_lookahead = self.base_lookahead * (drs_boost_speed / base_speed)

            if distance_to_exit < zone_braking_distance:
                # Approaching exit - quadratic ease-out (speed returns to base_speed, not 5 m/s)
                # FOV ease-out is linear so corner walls stay visible during braking
                blend_factor    = distance_to_exit / zone_braking_distance  # 1.0 -> 0.0
                smooth_factor   = 1 - (1 - blend_factor) ** 2
                fov_factor      = blend_factor  # Linear: FOV stays wider longer than speed
                effective_max_speed  = base_speed          + (drs_boost_speed  - base_speed)          * smooth_factor
                effective_lookahead  = self.base_lookahead + (drs_lookahead    - self.base_lookahead)  * smooth_factor
                effective_fov        = self.base_fov       + (self.drs_fov     - self.base_fov)        * fov_factor
            elif distance_from_entry < self.drs_spool_distance:
                # Just entered zone - quadratic ease-in
                is_sf_zone1 = (
                    len(self.DRS_ZONES) >= 2
                    and self.DRS_ZONES[0] is current_drs_zone
                    and self.DRS_ZONES[0].get('start_finish_pair')
                    and self.DRS_ZONES[-1].get('start_finish_pair')
                )
                if is_sf_zone1:
                    # Start/finish zone 1: already at speed crossing the line, no spool needed
                    effective_max_speed = drs_boost_speed
                    effective_lookahead = drs_lookahead
                    effective_fov       = self.drs_fov
                else:
                    blend_factor    = distance_from_entry / self.drs_spool_distance  # 0.0 -> 1.0
                    smooth_factor   = blend_factor ** 2
                    effective_max_speed  = base_speed          + (drs_boost_speed  - base_speed)          * smooth_factor
                    effective_lookahead  = self.base_lookahead + (drs_lookahead    - self.base_lookahead)  * smooth_factor
                    effective_fov        = self.base_fov       + (self.drs_fov     - self.base_fov)        * smooth_factor
            else:
                # Full DRS boost
                effective_max_speed = drs_boost_speed
                effective_lookahead = drs_lookahead
                effective_fov       = self.drs_fov

            # Independent lookahead + FOV exit taper:
            # Both fully back to base by the time distance_to_exit == drs_lookahead.
            # Uses min() so it only caps downward - never fights the ease-in spool.
            la_exit_blend   = np.clip((distance_to_exit - drs_lookahead) / max(drs_lookahead, 0.01), 0.0, 1.0)
            taper_lookahead = self.base_lookahead + (drs_lookahead - self.base_lookahead) * la_exit_blend
            taper_fov       = self.base_fov       + (self.drs_fov  - self.base_fov)       * la_exit_blend
            effective_lookahead = min(effective_lookahead, taper_lookahead)
            effective_fov       = min(effective_fov,       taper_fov)
        else:
            # Not in DRS zone - lap 1 warm-up uses base speed, lap 2+ uses race speed
            effective_max_speed = self._base_race_speed if self.lap_count >= 1 else self.max_speed
            effective_lookahead = self.base_lookahead
            effective_fov       = self.base_fov

        # Dynamic lookahead: in DRS zone, extend lookahead on straights only
        # Cap at 2 × current speed (physically: distance covered in 2 s) instead of
        # range_max — prevents the goal point jumping past the corner entry at high speed
        if current_drs_zone is not None:
            if abs(self.current_steering) < self.STEERING_THRESHOLD:
                dist_to_exit   = self.calculate_distance_to_zone_exit(current_drs_zone)
                corner_safe_la = current_drs_zone.get('corner_safe_lookahead', self.base_lookahead)
                speed_cap      = max(self.base_lookahead, 2.0 * abs(self.actual_velocity))
                if dist_to_exit < corner_safe_la:
                    # Blend from speed_cap → base_lookahead over the final corner_safe_la metres
                    blend = dist_to_exit / corner_safe_la  # 1.0 far → 0.0 at exit
                    effective_lookahead = self.base_lookahead + (speed_cap - self.base_lookahead) * blend
                else:
                    effective_lookahead = speed_cap
            else:
                effective_lookahead = self.base_lookahead

        # Restrict lookahead for sharp turns: scales from base_lookahead (5 m) at sharpest
        # up to the full effective_lookahead for gentle/straight sections.
        effective_lookahead = self._turn_aware_lookahead(effective_lookahead)

        # Apply effective lookahead and FOV before disparity_extender reads them
        self.lookahead_distance = effective_lookahead
        self.field_of_vision    = effective_fov

        proc_ranges, obs_bubble_coord, fov_start_i, fov_end_i, dispa_coord = self.disparity_extender(data)
        max_dist_index, goal_coord = self.find_max_gap(data, proc_ranges)
        ranges_coord = self.get_ranges_coord(data, proc_ranges, fov_start_i, fov_end_i)
        
        msg = AckermannDriveStamped()
        steering_angle = 0.0
        # Dynamic steering gain: scales linearly with speed above race base (8 m/s).
        # At 8 m/s → steering_gain (0.4).  At 20 m/s → steering_gain × (20/8) = 1.0.
        # Floored at steering_gain (never reduce below baseline).
        # Capped at steering_gain × 2.5 to prevent instability at very high speeds.
        #
        # DRS straights: keep gain flat (stable) — amplified gain causes oversteer on straights.
        # Pre-turn / post-turn (and all non-DRS driving): apply speed-scaled gain for sharper
        # turn-in and correction response at elevated speeds.
        _speed_scale    = np.clip(abs(self.actual_velocity) / 8.0, 1.0, 2.5)
        _on_drs_straight = (
            current_drs_zone is not None
            and abs(self.current_steering) < self.STEERING_THRESHOLD
        )
        active_gain = self.steering_gain if _on_drs_straight else self.steering_gain * _speed_scale
        if max_dist_index is not None:
            steering_angle = data.angle_min + max_dist_index * data.angle_increment
            steering_angle = np.clip(steering_angle * active_gain, -self.max_steering, self.max_steering)
            msg.drive.steering_angle = steering_angle
        
        # Store current steering for DRS zone detection
        self.current_steering = steering_angle
        # Accumulate steering for per-zone incident detection
        if current_drs_zone is not None:
            self._drs_active_steer_buf.append(steering_angle)

        raw_speed = effective_max_speed - abs(steering_angle) * self.speed_gain
        # Predictive braking: slow before known turns using warm-up turn data
        self._brake_target_turn = None
        self._brake_limit       = None
        predictive_limit = self._predictive_corner_speed(effective_max_speed)
        msg.drive.speed = max(self.min_speed, min(raw_speed, predictive_limit))

        self.visualisation_marker(obs_bubble_coord, goal_coord, ranges_coord, dispa_coord)

        # Status log
        if current_drs_zone is not None:
            distance_to_exit = self.calculate_distance_to_zone_exit(current_drs_zone)
            zone_name = current_drs_zone.get('name', f'Zone {zone_idx+1}')
            drs_status = f"DRS ON [{zone_name}] (exit in {distance_to_exit:.1f}m)"
        else:
            drs_status = f"DRS OFF (lap {self.lap_count})"

        self._drs_log_counter += 1
        if self._drs_log_counter >= 2:
            self._drs_log_counter = 0
            if self.lap_count >= 1:
                # Turn position indicator: show current or next upcoming turn
                turn_info = ''
                if self.turn_log:
                    current_turn = next(
                        (t for t in self.turn_log
                         if t['from_m'] <= self.total_distance <= t['to_m']),
                        None
                    )
                    if current_turn:
                        turn_info = f' | >> Turn {current_turn["turn_number"]} (r={current_turn["turn_radius_m"]:.1f}m)'
                    else:
                        upcoming = [t for t in self.turn_log if t['from_m'] > self.total_distance]
                        if upcoming:
                            nxt = min(upcoming, key=lambda t: t['from_m'])
                            turn_info = f' | Turn {nxt["turn_number"]} in {nxt["from_m"] - self.total_distance:.1f}m'

                brake_info = ''
                if self._brake_target_turn is not None:
                    brake_info = (
                        f' | BRAKE T{self._brake_target_turn["turn_number"]} '
                        f'{self._brake_dist_to_turn:.1f}m away '
                        f'-> {self._brake_v_corner:.1f}m/s '
                        f'(lim={self._brake_limit:.1f})'
                    )
                '''
                self.get_logger().info(
                    f'{drs_status} | '
                    f'Speed: {msg.drive.speed:.2f} m/s | '
                    f'Steer: {msg.drive.steering_angle:.2f} rad | '
                    f'Lookahead: {self.lookahead_distance:.1f} m | '
                    f'ODO: {self.total_distance:.1f} m'
                    f'{turn_info}'
                    f'{brake_info}'
                )
                '''
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    reactive_node = GapFollower()
    print("WARM UP PHASE: Detecting DRS zones... (drive around the track at least once)")
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
