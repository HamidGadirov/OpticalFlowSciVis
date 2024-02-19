import taichi as ti
from ..basic_op.type import *

class Gui3d:
    def __init__(self):
        self.canvas = None
        self.scene = None
        self.camera = None

        self.frame_rate = 60
        self.camera_fov = 55
        self.res = (1920, 1080)
        self.camera_pos = (0.0, 0.0, 10.0)
        self.camera_lookat = (0.0, 0.0, 0.0)
        self.canvas_color = (0.2, 0.2, 0.6)
        self.ambient_color = (0.7, 0.7, 0.7)
        self.point_light_pos = (2, 1.5, -1.5)
        self.point_light_color = (0.8, 0.8, 0.8)

        # Toggles
        self.show_bound = True
        self.show_help = True
        self.show_run_info = True
        self.op_system_run = False
        self.op_write_file = False
        self.op_refresh_window = True
        self.op_save_img = False

        self.env_set_up()


    def env_set_up(self):
        self.window = ti.ui.Window(
            "Fluid Simulation",
            self.res,
            vsync=True,
        )
        
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()

        self.canvas.set_background_color(self.canvas_color)
        self.camera.fov(self.camera_fov)
        self.camera.position(
            self.camera_pos[0],
            self.camera_pos[1],
            self.camera_pos[2],
        )
        self.camera.lookat(
            self.camera_lookat[0],
            self.camera_lookat[1],
            self.camera_lookat[2],
        )

        
    def monitor_listen(self):
        self.camera.track_user_inputs(
            self.window, movement_speed=0.03, hold_key=ti.ui.RMB
        )

        if self.show_help:
            self.window.GUI.begin("options", 0.05, 0.3, 0.2, 0.2)
            self.window.GUI.text("w: front")
            self.window.GUI.text("s: back")
            self.window.GUI.text("a: left")
            self.window.GUI.text("d: right")
            self.window.GUI.text("RMB: rotate")
            self.window.GUI.text("r: run system")
            self.window.GUI.text("f: write file")
            # self.window.GUI.text("c: refresh window")
            self.window.GUI.text("i: save image to the 'output' folder")
            self.window.GUI.end()

        if self.window.get_event(ti.ui.PRESS):
            # run
            if self.window.event.key == "r":
                self.op_system_run = not self.op_system_run
                print("start to run:", self.op_system_run)

            if self.window.event.key == "f":
                self.op_write_file = not self.op_write_file
                print("write file:", self.op_write_file)
            
            if self.window.event.key == "i":
                self.op_save_img = not self.op_save_img
                print("save image:", self.op_save_img)

            if self.window.event.key == "b":
                self.show_bound = not self.show_bound
                print("show boundary:", self.show_bound)

            # if self.window.event.key == "i":
            #     self.show_run_info = not self.show_run_info
            #     print("show run information:", self.show_run_info)

            # if self.window.event.key == "h":
            #     self.show_help = not self.show_help
            #     print("show help:", self.show_help)
            
            # if self.window.event.key == "c":
            #     self.op_refresh_window = not self.op_refresh_window
            #     print("refresh window:", self.op_refresh_window)

    

    def scene_setup(self):
        self.scene.set_camera(self.camera)
        self.scene.ambient_light(self.ambient_color)
        self.scene.point_light(pos=self.point_light_pos, color=self.point_light_color)

    def scene_add_parts_colorful(self, obj_pos: ti.template(), obj_color:ti.template(), index_count:ti.i32, size:ti.f32):
        self.scene.particles(centers=obj_pos, radius=size/2, per_vertex_color=obj_color, index_offset=0, index_count=index_count)

    def scene_add_parts(self, obj_pos: ti.template(), index_count:ti.i32, size:ti.f32, obj_color=(0.5,0.5,0.5)):
        self.scene.particles(centers=obj_pos, radius=size/2, color=obj_color, index_offset=0, index_count=index_count)

    def scene_render(self):
        self.canvas.scene(self.scene)  # Render the scene
        self.window.show()