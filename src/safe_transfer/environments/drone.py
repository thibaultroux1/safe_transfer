import pymunk
import pygame
import numpy as np


class Drone():
    """
    Represents a simulated drone composed of a central frame and two motors
    within a Pymunk physics space.

    The drone is constructed from three dynamic bodies (frame, left motor, right motor)
    connected by multiple PivotJoints to maintain structural integrity while allowing
    independent physical properties.

    Attributes:
        drone_radius (float): The distance from the center of the frame to the center of each motor.
        frame_shape (pymunk.Poly): The Pymunk shape representing the drone's central frame.
        left_motor_shape (pymunk.Poly): The Pymunk shape representing the left motor.
        right_motor_shape (pymunk.Poly): The Pymunk shape representing the right motor.
        left_1, left_2, left_3 (pymunk.PivotJoint): Joints connecting the left motor to the frame.
        right_1, right_2, right_3 (pymunk.PivotJoint): Joints connecting the right motor to the frame.
    """

    def __init__(self, x: float, y: float, angle: float, height: float, width: float,
                 mass_f: float, mass_l: float, mass_r: float, space: pymunk.Space):
        """
        Initializes the Drone object, creating its physical components and adding them
        to the provided Pymunk space.

        Args:
            x (float): Initial x-coordinate of the drone's center.
            y (float): Initial y-coordinate of the drone's center.
            angle (float): Initial angle of the drone in radians.
            height (float): The height of the motor boxes and half-height of the central frame.
            width (float): The total width of the drone's central frame.
            mass_f (float): Mass of the central frame.
            mass_l (float): Mass of the left motor.
            mass_r (float): Mass of the right motor.
            space (pymunk.Space): The Pymunk space to add the drone components to.
        """
        distance_between_joints = height / 2 - 3
        self.drone_radius = width / 2 - height / 2 # Distance from frame center to motor center

        # --- Drone's frame properties ---
        self.frame_shape = pymunk.Poly.create_box(None, size=(width, height / 2))
        frame_moment_of_inertia = pymunk.moment_for_poly(mass_f, self.frame_shape.get_vertices())

        frame_body = pymunk.Body(mass_f, frame_moment_of_inertia, body_type=pymunk.Body.DYNAMIC)
        frame_body.position = x, y
        frame_body.angle = angle

        self.frame_shape.body = frame_body
        # Sensor=True means it detects collisions but doesn't produce contact points/forces itself
        # Useful for composite objects where sub-shapes handle collisions.
        self.frame_shape.sensor = True # Set to False if frame should collide directly
        self.frame_shape.color = pygame.Color((66, 135, 245)) # Blue

        space.add(frame_body, self.frame_shape)

        # --- Drone's left motor properties ---
        self.left_motor_shape = pymunk.Poly.create_box(None, size=(height, height))
        left_motor_moment_of_inertia = pymunk.moment_for_poly(mass_l, self.left_motor_shape.get_vertices())

        left_motor_body = pymunk.Body(mass_l, left_motor_moment_of_inertia, body_type=pymunk.Body.DYNAMIC)
        # Calculate initial position relative to frame center and angle
        left_motor_body.position = np.cos(angle + np.pi) * self.drone_radius + x, \
                                   np.sin(angle + np.pi) * self.drone_radius + y
        left_motor_body.angle = angle

        self.left_motor_shape.body = left_motor_body
        self.left_motor_shape.sensor = True # Set to False if motors should collide
        self.left_motor_shape.color = pygame.Color((33, 93, 191)) # Darker Blue

        space.add(left_motor_body, self.left_motor_shape)

        # --- Drone's right motor properties ---
        self.right_motor_shape = pymunk.Poly.create_box(None, size=(height, height))
        right_motor_moment_of_inertia = pymunk.moment_for_poly(mass_r, self.right_motor_shape.get_vertices())

        right_motor_body = pymunk.Body(mass_r, right_motor_moment_of_inertia, body_type=pymunk.Body.DYNAMIC)
        # Calculate initial position relative to frame center and angle
        right_motor_body.position = np.cos(angle) * self.drone_radius + x, \
                                    np.sin(angle) * self.drone_radius + y
        right_motor_body.angle = angle

        self.right_motor_shape.body = right_motor_body
        self.right_motor_shape.sensor = True
        self.right_motor_shape.color = pygame.Color((33, 93, 191)) # Darker Blue

        space.add(right_motor_body, self.right_motor_shape)

        # --- Properties of the joints ---
        # Multiple joints are used per motor to create a rigid connection.
        # Anchor points are defined relative to the body's center of gravity.

        # Left Motor Joints
        motor_point = (-distance_between_joints, 0) # Point on left motor
        frame_point = (-self.drone_radius - distance_between_joints, 0) # Corresponding point on frame
        self.left_1 = pymunk.PivotJoint(self.left_motor_shape.body, self.frame_shape.body, motor_point, frame_point)
        self.left_1.error_bias = 0 # Disable joint error correction bias
        space.add(self.left_1)

        motor_point = (0, 0)
        frame_point = (-self.drone_radius, 0)
        self.left_2 = pymunk.PivotJoint(self.left_motor_shape.body, self.frame_shape.body, motor_point, frame_point)
        self.left_2.error_bias = 0
        space.add(self.left_2)

        motor_point = (distance_between_joints, 0)
        frame_point = (-self.drone_radius + distance_between_joints, 0)
        self.left_3 = pymunk.PivotJoint(self.left_motor_shape.body, self.frame_shape.body, motor_point, frame_point)
        self.left_3.error_bias = 0
        space.add(self.left_3)

        # Right Motor Joints
        motor_point = (-distance_between_joints, 0) # Point on right motor
        frame_point = (self.drone_radius - distance_between_joints, 0) # Corresponding point on frame
        self.right_1 = pymunk.PivotJoint(self.right_motor_shape.body, self.frame_shape.body, motor_point, frame_point)
        self.right_1.error_bias = 0
        space.add(self.right_1)

        motor_point = (0, 0)
        frame_point = (self.drone_radius, 0)
        self.right_2 = pymunk.PivotJoint(self.right_motor_shape.body, self.frame_shape.body, motor_point, frame_point)
        self.right_2.error_bias = 0
        space.add(self.right_2)

        motor_point = (distance_between_joints, 0)
        frame_point = (self.drone_radius + distance_between_joints, 0)
        self.right_3 = pymunk.PivotJoint(self.right_motor_shape.body, self.frame_shape.body, motor_point, frame_point)
        self.right_3.error_bias = 0
        space.add(self.right_3)

    def change_positions(self, x: float, y: float, space: pymunk.Space):
        """
        Manually sets the position of the drone's components and reindexes them
        in the physics space.

        Note: This directly manipulates positions and might break physical constraints
              if not used carefully (e.g., during environment reset). It maintains
              the relative positions based on the current frame angle.

        Args:
            x (float): The new x-coordinate for the drone's central frame.
            y (float): The new y-coordinate for the drone's central frame.
            space (pymunk.Space): The Pymunk space containing the drone.
        """
        # Set frame position and reindex
        self.frame_shape.body.position = x, y
        space.reindex_shapes_for_body(self.frame_shape.body)

        # Get current angle to maintain relative motor positions
        angle = self.frame_shape.body.angle

        # Set left motor position based on new frame position and angle, then reindex
        self.left_motor_shape.body.position = np.cos(angle + np.pi) * self.drone_radius + x, \
                                              np.sin(angle + np.pi) * self.drone_radius + y
        space.reindex_shapes_for_body(self.left_motor_shape.body)

        # Set right motor position based on new frame position and angle, then reindex
        self.right_motor_shape.body.position = np.cos(angle) * self.drone_radius + x, \
                                               np.sin(angle) * self.drone_radius + y
        space.reindex_shapes_for_body(self.right_motor_shape.body)