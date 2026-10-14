"""
coppelia_controller.py

Remote API client and controller for the 2-DOF planar robot in CoppeliaSim.
Manages object handles, joint control, collision masks, simulation stepping,
and pen-tip telemetry.
"""

import math
import time
from typing import Optional, Tuple

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from src.config import SimConfig, DEFAULT_SIM_CONFIG


class CoppeliaController:
    """
    Interface for controlling the 2-DOF planar robot inside CoppeliaSim
    via the ZeroMQ remote API.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        config: SimConfig = DEFAULT_SIM_CONFIG,
        verbose: bool = True,
    ):
        """
        Initialize the CoppeliaSim connection and resolve robot handles.

        Parameters:
            host: CoppeliaSim host (defaults to config.host).
            port: CoppeliaSim port (defaults to config.port).
            config: SimConfig instance containing paths and tuning parameters.
            verbose: Whether to log handle discovery details.
        """
        self.config = config
        target_host = host if host is not None else config.host
        target_port = port if port is not None else config.port

        self.client = RemoteAPIClient(host=target_host, port=target_port)
        self.sim = self.client.require("sim")

        # Resolve robot hierarchy object handles
        self.base = self.sim.getObject(config.base_path)
        self.joint1 = self.sim.getObject(config.joint1_path)
        self.link1 = self.sim.getObject(config.link1_path)
        self.joint2 = self.sim.getObject(config.joint2_path)
        self.link2 = self.sim.getObject(config.link2_path)

        # Resolve or create the pen Tip dummy handle
        self.tip = self._ensure_tip_dummy()

        # Configure dynamic parameters (non-colliding link masks and PID stiffness)
        self.configure_robot_dynamics()

        if verbose:
            print("Connected to CoppeliaSim successfully.")
            print(f"Joint 1 handle: {self.joint1} | {self.sim.getObjectAlias(self.joint1, 4)}")
            print(f"Joint 2 handle: {self.joint2} | {self.sim.getObjectAlias(self.joint2, 4)}")
            print(f"Tip dummy handle: {self.tip} | {self.sim.getObjectAlias(self.tip, 4)}")

    def _ensure_tip_dummy(self) -> int:
        """
        Verify that the Tip dummy exists on Link2, or create it at the
        distal end of Link2 (local position [0.05, 0, 0] m).
        """
        try:
            tip = self.sim.getObject(self.config.tip_path)
            return tip
        except Exception:
            tip = self.sim.createDummy(0.005)
            self.sim.setObjectAlias(tip, "Tip")
            self.sim.setObjectParent(tip, self.link2, False)
            self.sim.setObjectPosition(tip, [0.05, 0.0, 0.0], self.link2)
            self.sim.setObjectOrientation(tip, [0.0, 0.0, 0.0], self.link2)
            return tip

    def configure_robot_dynamics(self) -> None:
        """
        Apply physics fixes to prevent self-collision blow-out and ensure
        accurate trajectory tracking:
        1. Set Link1 and Link2 to disjoint respondable bitmasks so they do
           not dynamically collide with each other at the elbow.
        2. Set PID gains, velocity limits, and acceleration limits.
        """
        try:
            # Set disjoint respondable collision masks (1 & 2 == 0)
            self.sim.setObjectInt32Param(
                self.link1, self.sim.shapeintparam_respondable_mask, 1
            )
            self.sim.setObjectInt32Param(
                self.link2, self.sim.shapeintparam_respondable_mask, 2
            )

            # Set stiff PID gains and velocity/acceleration limits on both joints
            for joint in (self.joint1, self.joint2):
                self.sim.setObjectFloatParam(
                    joint, self.sim.jointfloatparam_pid_p, self.config.pid_p
                )
                self.sim.setObjectFloatParam(
                    joint, self.sim.jointfloatparam_pid_d, self.config.pid_d
                )
                self.sim.setObjectFloatParam(
                    joint, self.sim.jointfloatparam_maxvel, self.config.max_vel
                )
                self.sim.setObjectFloatParam(
                    joint, self.sim.jointfloatparam_maxaccel, self.config.max_accel
                )
        except Exception:
            pass

    def start_simulation(self) -> None:
        """Start simulation if stopped."""
        if self.sim.getSimulationState() == self.sim.simulation_stopped:
            self.sim.startSimulation()

    def stop_simulation(self) -> None:
        """Stop simulation if running or paused."""
        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.sim.stopSimulation()
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                time.sleep(0.01)

    def is_running(self) -> bool:
        """Return True if simulation is actively running."""
        return self.sim.getSimulationState() != self.sim.simulation_stopped

    def set_stepping(self, enabled: bool) -> None:
        """Enable or disable synchronous stepping mode."""
        self.client.setStepping(enabled)

    def step(self) -> None:
        """Advance the simulation by one step in synchronous mode."""
        self.client.step()

    def get_joint_angles(self) -> Tuple[float, float]:
        """Return current joint angles (theta1, theta2) in radians."""
        return (
            self.sim.getJointPosition(self.joint1),
            self.sim.getJointPosition(self.joint2),
        )

    def get_joint_angles_deg(self) -> Tuple[float, float]:
        """Return current joint angles (theta1, theta2) in degrees."""
        t1, t2 = self.get_joint_angles()
        return math.degrees(t1), math.degrees(t2)

    def set_joint_targets(self, theta1: float, theta2: float) -> None:
        """Set target positions for Joint 1 and Joint 2 in radians."""
        self.sim.setJointTargetPosition(self.joint1, theta1)
        self.sim.setJointTargetPosition(self.joint2, theta2)

    def set_joint_targets_deg(self, theta1_deg: float, theta2_deg: float) -> None:
        """Set target positions for Joint 1 and Joint 2 in degrees."""
        self.set_joint_targets(math.radians(theta1_deg), math.radians(theta2_deg))

    def get_tip_position(self, relative_to: int = -1) -> Tuple[float, float, float]:
        """
        Get the 3D position of the pen tip in centimetres.

        Parameters:
            relative_to: Object handle to measure position relative to.
                         Default -1 represents the world frame (identical to
                         RobotBase XY frame).

        Returns:
            (x_cm, y_cm, z_cm) in centimetres.
        """
        pos_m = self.sim.getObjectPosition(self.tip, relative_to)
        return (pos_m[0] * 100.0, pos_m[1] * 100.0, pos_m[2] * 100.0)

    def get_tip_xy_cm(self, relative_to: int = -1) -> Tuple[float, float]:
        """Get (x, y) coordinates of the pen tip in centimetres."""
        pos = self.get_tip_position(relative_to=relative_to)
        return (pos[0], pos[1])

    def wait_for_settle(
        self,
        target1_rad: float,
        target2_rad: float,
        timeout_s: float = 3.0,
        tolerance_deg: Optional[float] = None,
        step_simulation: bool = False,
    ) -> bool:
        """
        Wait until both joints reach their commanded targets within tolerance.

        Parameters:
            target1_rad: Target angle for Joint 1 in radians.
            target2_rad: Target angle for Joint 2 in radians.
            timeout_s: Maximum duration to wait in seconds.
            tolerance_deg: Maximum allowable absolute error per joint in degrees.
            step_simulation: If True, invoke client.step() on each iteration.

        Returns:
            True if settled within tolerance, False if timed out.
        """
        tol = tolerance_deg if tolerance_deg is not None else self.config.position_tolerance_deg
        tol_rad = math.radians(tol)
        start_time = time.time()

        while (time.time() - start_time) < timeout_s:
            if step_simulation:
                self.step()
            else:
                time.sleep(0.01)

            t1, t2 = self.get_joint_angles()
            err1 = abs(t1 - target1_rad)
            err2 = abs(t2 - target2_rad)

            if err1 <= tol_rad and err2 <= tol_rad:
                return True

        return False

    def create_drawing_object(
        self,
        line_width: float = 3.0,
        color: Tuple[float, float, float] = (0.0, 0.2, 0.8),
        max_lines: int = 20000,
    ) -> int:
        """
        Create a visible ink drawing object in CoppeliaSim.

        Parameters:
            line_width: Rendered width of the ink line.
            color: Normalized RGB tuple in [0.0, 1.0].
            max_lines: Maximum line segment buffer capacity.

        Returns:
            Drawing object handle.
        """
        self.clear_drawing_object()
        self.drawing_object_handle = self.sim.addDrawingObject(
            self.sim.drawing_lines + self.sim.drawing_painttag,
            line_width,
            0.0,
            -1,
            max_lines,
            list(color),
        )
        return self.drawing_object_handle

    def add_ink_segment(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        in_meters: bool = False,
    ) -> None:
        """
        Add a visible line segment between 3D points p1 and p2.

        Parameters:
            p1: Starting (x, y, z) coordinates.
            p2: Ending (x, y, z) coordinates.
            in_meters: If False (default), inputs are in centimetres and converted to metres.
        """
        if self.drawing_object_handle is None:
            self.create_drawing_object()

        scale = 1.0 if in_meters else 0.01
        line_data = [
            p1[0] * scale, p1[1] * scale, p1[2] * scale,
            p2[0] * scale, p2[1] * scale, p2[2] * scale,
        ]
        self.sim.addDrawingObjectItem(self.drawing_object_handle, line_data)

    def clear_drawing_object(self) -> None:
        """Remove and reset all visible ink drawing objects in the scene."""
        try:
            self.sim.removeDrawingObject(self.sim.handle_all)
        except Exception:
            if hasattr(self, "drawing_object_handle") and self.drawing_object_handle is not None:
                try:
                    self.sim.removeDrawingObject(self.drawing_object_handle)
                except Exception:
                    pass
        self.drawing_object_handle = None

    def get_vision_sensor_image(
        self,
        camera_path: str = "/TopDownVisionSensor",
    ) -> np.ndarray:
        """
        Capture and return an RGB image from the specified vision sensor.

        Returns:
            NumPy array of shape (H, W, 3) with uint8 RGB values.
        """
        cam = self.sim.getObject(camera_path)
        img_bytes, [res_x, res_y] = self.sim.getVisionSensorImg(cam)
        raw_img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((res_y, res_x, 3))
        # CoppeliaSim stores sensor images bottom-to-top; flip vertically
        return np.flipud(raw_img)
