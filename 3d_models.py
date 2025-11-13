import vtk
import numpy as np
import cv2
from vtk.util import numpy_support


class Medical3DModelLoader:
    def __init__(self):
        self.models = {}
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 600)
        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetPosition(0, 0, 500)
        self.camera.SetFocalPoint(0, 0, 0)
        self.renderer.SetBackground(0.1, 0.1, 0.1)

    def load_stl_model(self, filepath, name, color=(1.0, 0.8, 0.8)):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filepath)
        reader.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(0.8)
        self.models[name] = {
            'actor': actor,
            'reader': reader,
            'visible': True
        }
        self.renderer.AddActor(actor)

        return actor

    def load_obj_model(self, filepath, name, color=(1.0, 0.8, 0.8)):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filepath)
        reader.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(0.8)

        self.models[name] = {
            'actor': actor,
            'reader': reader,
            'visible': True
        }
        self.renderer.AddActor(actor)

        return actor

    def create_heart_model(self):
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(50)
        sphere.SetPhiResolution(30)
        sphere.SetThetaResolution(30)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.0, 0.0)  # Red

        self.models['heart'] = {
            'actor': actor,
            'visible': True
        }
        self.renderer.AddActor(actor)

        return actor

    def create_lung_model(self, position=(0, 0, 0)):
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(40)
        cylinder.SetHeight(120)
        cylinder.SetResolution(20)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cylinder.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.7, 0.7)  # Pink
        actor.SetPosition(position)

        name = f'lung_{position[0]}'
        self.models[name] = {
            'actor': actor,
            'visible': True
        }
        self.renderer.AddActor(actor)

        return actor

    def set_model_visibility(self, name, visible):
        if name in self.models:
            self.models[name]['visible'] = visible
            self.models[name]['actor'].SetVisibility(visible)

    def set_model_opacity(self, name, opacity):
        if name in self.models:
            self.models[name]['actor'].GetProperty().SetOpacity(opacity)

    def rotate_model(self, name, x_angle, y_angle, z_angle):
        if name in self.models:
            actor = self.models[name]['actor']
            actor.RotateX(x_angle)
            actor.RotateY(y_angle)
            actor.RotateZ(z_angle)

    def rotate_all(self, x_angle, y_angle, z_angle):
        for model_data in self.models.values():
            actor = model_data['actor']
            actor.RotateX(x_angle)
            actor.RotateY(y_angle)
            actor.RotateZ(z_angle)

    def render_to_image(self):
        self.render_window.Render()

        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.render_window)
        w2if.Update()

        # Convert to numpy
        vtk_image = w2if.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()

        arr = numpy_support.vtk_to_numpy(vtk_array)
        arr = arr.reshape(height, width, components)

        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        arr = cv2.flip(arr, 0)  # VTK image is upside down

        return arr

    def add_text_annotation(self, text, position, color=(1, 1, 1)):
        text_actor = vtk.vtkTextActor3D()
        text_actor.SetInput(text)
        text_actor.SetPosition(position)
        text_actor.GetTextProperty().SetFontSize(20)
        text_actor.GetTextProperty().SetColor(color)

        self.renderer.AddActor(text_actor)
        return text_actor


class VTKGestureController:
    def __init__(self, model_loader):
        self.loader = model_loader
        self.rotation_speed = 2.0

    def handle_rotation_gesture(self, dx, dy):
        """Gesture se rotation control karo"""
        self.loader.rotate_all(dy * self.rotation_speed,
                               dx * self.rotation_speed, 0)

    def handle_zoom_gesture(self, zoom_factor):
        """Zoom control karo"""
        camera = self.loader.camera
        camera.Zoom(zoom_factor)


def demo_medical_viewer():
    """Complete medical viewer demo"""
    loader = Medical3DModelLoader()
    print("Loading anatomical models...")
    loader.create_heart_model()
    loader.create_lung_model((-60, 0, 0))  # Left lung
    loader.create_lung_model((60, 0, 0))  # Right lung
    loader.add_text_annotation("Heart", (0, 60, 0), (1, 0, 0))
    loader.add_text_annotation("Left Lung", (-60, 60, 0), (1, 1, 1))
    loader.add_text_annotation("Right Lung", (60, 60, 0), (1, 1, 1))
    controller = VTKGestureController(loader)

    print("Medical 3D Viewer Started!")
    print("Press 'q' to quit")
    print("Use mouse to interact in VTK window")
    rotation = 0
    while True:
        loader.rotate_all(0, 1, 0)

        image = loader.render_to_image()
        cv2.imshow('Medical 3D Viewer', image)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            # Toggle heart visibility
            visible = loader.models['heart']['visible']
            loader.set_model_visibility('heart', not visible)
        elif key == ord('2'):
            # Toggle lungs
            for name in loader.models:
                if 'lung' in name:
                    visible = loader.models[name]['visible']
                    loader.set_model_visibility(name, not visible)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_medical_viewer()