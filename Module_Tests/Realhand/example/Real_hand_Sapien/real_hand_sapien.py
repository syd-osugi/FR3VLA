import sapien.core as sapien

import numpy as np

class RealHandController:
    def __init__(self, engine, scene, urdf_path,camera):
        self.engine = engine
        self.scene = scene
        self.urdf_path = urdf_path
        self.camera = camera
        self.robot = None  # Articulation object
        self.joint_names = []  # Joint name list
        self.joint_indices = {}  # Mapping from joint name to index

        # Load URDF file and initialize drives
        self.load_urdf()

    def load_urdf(self):
        """Load URDF file and initialize robot"""
        #loader = URDFLoader(self.engine, self.scene)
        scene = self.engine.create_scene()
        loader = scene.create_urdf_loader()
        self.robot = loader.load(self.urdf_path)
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        # Render in simulation loop
        scene.update_render()
        self.camera.take_picture()
        rgb = self.camera.get_color_rgba()  # Get image data
        # Get names and indices of all active joints
        active_joints = self.robot.get_active_joints()
        self.joint_names = [joint.get_name() for joint in active_joints]
        self.joint_indices = {name: idx for idx, name in enumerate(self.joint_names)}

        # Initialize joint drive parameters (PD control)
        for joint in active_joints:
            joint.set_drive_property(stiffness=100.0, damping=10.0)

    def set_joint_position(self, joint_name, target_position):
        """Set target position of the specified joint (via PD control)"""
        if joint_name in self.joint_indices:
            idx = self.joint_indices[joint_name]
            #self.robot.set_q_target(target_position, idx)
        else:
            print(f"Joint {joint_name} not found!")

    def get_joint_positions(self):
        """Get current positions of all joints"""
        return self.robot.get_qpos()

    def step(self):
        """Advance simulation"""
        self.scene.step()


# Example usage
def main():
    # Initialize Sapien engine and scene
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    scene = engine.create_scene()
    scene.set_timestep(1 / 240.0)  # Set simulation time step
    camera = scene.add_camera("camera", width=1280, height=720, fovy=1.0, near=0.1, far=100.0)
    camera.set_local_pose(sapien.Pose([-0.5, 0, 0.5], [0.707, 0, 0.707, 0]))
    # Add ground
    scene.add_ground(altitude=0.0)

    # Create RealHandController instance
    urdf_path = "urdf/real_hand_l20_8_right.urdf"  # Replace with your URDF file path
    hand_controller = RealHandController(engine, scene, urdf_path,camera)

    # Set joint target positions
    hand_controller.set_joint_position("thumb_joint0", 0.5)  # Set thumb joint position
    hand_controller.set_joint_position("index_joint0", 0.3)  # Set index finger joint position

    # Simulation loop
    for _ in range(1000):
        hand_controller.step()  # Advance simulation
        positions = hand_controller.get_joint_positions()  # Get all joint positions
        print(positions)  # Print joint positions (as an array)


if __name__ == "__main__":
    main()
