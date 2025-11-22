# [file name]: unified_3d_anatomy.py
import vtk
import numpy as np
import cv2
from vtk.util import numpy_support
import math


class UnifiedAnatomySystem:
    def __init__(self):
        self.models = {}
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 600)

        # Camera setup
        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetPosition(0, -300, 300)
        self.camera.SetFocalPoint(0, 0, 0)
        self.camera.SetViewUp(0, 0, 1)
        self.renderer.SetBackground(0.0, 0.0, 0.0)  # Black background for AR overlay

        # Layer system
        self.layers = {
            'skeleton': {'visible': False, 'opacity': 0.9},
            'organs': {'visible': True, 'opacity': 0.8},
            'muscles': {'visible': False, 'opacity': 0.7},
            'nerves': {'visible': False, 'opacity': 0.6},
            'skin': {'visible': False, 'opacity': 0.3}
        }

        self.current_layer = 'organs'

    def load_complete_anatomy(self):
        """Placeholder for loading all models."""
        self.create_complete_skeleton()
        self.create_detailed_heart()
        self.create_detailed_lungs()
        self.create_brain_model()
        self.create_digestive_system()
        self.create_muscular_system()
        self.create_nervous_system()
        # Placeholder for 'appendix' model for the procedure
        self.create_appendix_model()

    def create_appendix_model(self):
        """Simplified appendix model for the procedure."""
        appendix = vtk.vtkCylinderSource()
        appendix.SetRadius(3)
        appendix.SetHeight(30)
        appendix.SetResolution(8)

        appendix_mapper = vtk.vtkPolyDataMapper()
        appendix_mapper.SetInputConnection(appendix.GetOutputPort())

        appendix_actor = vtk.vtkActor()
        appendix_actor.SetMapper(appendix_mapper)
        appendix_actor.GetProperty().SetColor(0.0, 0.8, 0.0) # Green
        appendix_actor.SetPosition(50, 100, 50) # Placeholder position

        self.models['appendix'] = {
            'actor': appendix_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(appendix_actor)

    def create_complete_skeleton(self):
        """Create complete skeletal system"""
        skeleton_parts = {}

        # Spine
        spine = vtk.vtkCylinderSource()
        spine.SetRadius(8)
        spine.SetHeight(180)
        spine.SetResolution(12)

        spine_mapper = vtk.vtkPolyDataMapper()
        spine_mapper.SetInputConnection(spine.GetOutputPort())

        spine_actor = vtk.vtkActor()
        spine_actor.SetMapper(spine_mapper)
        spine_actor.GetProperty().SetColor(0.9, 0.9, 0.8)
        spine_actor.SetPosition(0, 0, 90)
        spine_actor.RotateX(90)

        self.models['spine'] = {
            'actor': spine_actor,
            'layer': 'skeleton',
            'visible': True
        }
        self.renderer.AddActor(spine_actor)

        # Skull
        skull = vtk.vtkSphereSource()
        skull.SetRadius(25)
        skull.SetPhiResolution(20)
        skull.SetThetaResolution(20)

        skull_mapper = vtk.vtkPolyDataMapper()
        skull_mapper.SetInputConnection(skull.GetOutputPort())

        skull_actor = vtk.vtkActor()
        skull_actor.SetMapper(skull_mapper)
        skull_actor.GetProperty().SetColor(0.9, 0.9, 0.8)
        skull_actor.SetPosition(0, 0, 200)

        self.models['skull'] = {
            'actor': skull_actor,
            'layer': 'skeleton',
            'visible': True
        }
        self.renderer.AddActor(skull_actor)

        # Ribs (simplified)
        for i in range(12):
            rib = vtk.vtkCylinderSource()
            rib.SetRadius(1)
            rib.SetHeight(40 + i * 2)
            rib.SetResolution(8)

            rib_mapper = vtk.vtkPolyDataMapper()
            rib_mapper.SetInputConnection(rib.GetOutputPort())

            rib_actor = vtk.vtkActor()
            rib_actor.SetMapper(rib_mapper)
            rib_actor.GetProperty().SetColor(0.9, 0.9, 0.8)
            rib_actor.SetPosition(0, 20 + i * 3, 100 - i * 5)
            rib_actor.RotateX(90)
            rib_actor.RotateZ(30 + i * 5)

            name = f'rib_{i}'
            self.models[name] = {
                'actor': rib_actor,
                'layer': 'skeleton',
                'visible': True
            }
            self.renderer.AddActor(rib_actor)

        return skeleton_parts

    def create_detailed_heart(self):
        """Create detailed heart model with chambers"""
        # Main heart body
        heart_main = vtk.vtkSphereSource()
        heart_main.SetRadius(25)
        heart_main.SetPhiResolution(20)
        heart_main.SetThetaResolution(20)

        heart_mapper = vtk.vtkPolyDataMapper()
        heart_mapper.SetInputConnection(heart_main.GetOutputPort())

        heart_actor = vtk.vtkActor()
        heart_actor.SetMapper(heart_mapper)
        heart_actor.GetProperty().SetColor(0.8, 0.2, 0.2)
        heart_actor.SetPosition(0, -40, 120)

        self.models['heart_main'] = {
            'actor': heart_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(heart_actor)

        # Aorta
        aorta = vtk.vtkCylinderSource()
        aorta.SetRadius(5)
        aorta.SetHeight(30)
        aorta.SetResolution(12)

        aorta_mapper = vtk.vtkPolyDataMapper()
        aorta_mapper.SetInputConnection(aorta.GetOutputPort())

        aorta_actor = vtk.vtkActor()
        aorta_actor.SetMapper(aorta_mapper)
        aorta_actor.GetProperty().SetColor(0.8, 0.3, 0.3)
        aorta_actor.SetPosition(0, -55, 140)
        aorta_actor.RotateX(90)

        self.models['aorta'] = {
            'actor': aorta_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(aorta_actor)

        # Ventricles
        left_ventricle = vtk.vtkSphereSource()
        left_ventricle.SetRadius(12)
        left_ventricle.SetPhiResolution(15)
        left_ventricle.SetThetaResolution(15)

        ventricle_mapper = vtk.vtkPolyDataMapper()
        ventricle_mapper.SetInputConnection(left_ventricle.GetOutputPort())

        left_ventricle_actor = vtk.vtkActor()
        left_ventricle_actor.SetMapper(ventricle_mapper)
        left_ventricle_actor.GetProperty().SetColor(0.7, 0.1, 0.1)
        left_ventricle_actor.SetPosition(-10, -35, 115)

        self.models['left_ventricle'] = {
            'actor': left_ventricle_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(left_ventricle_actor)

        right_ventricle_actor = vtk.vtkActor()
        right_ventricle_actor.SetMapper(ventricle_mapper)
        right_ventricle_actor.GetProperty().SetColor(0.7, 0.1, 0.1)
        right_ventricle_actor.SetPosition(10, -35, 115)

        self.models['right_ventricle'] = {
            'actor': right_ventricle_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(right_ventricle_actor)

    def create_detailed_lungs(self):
        """Create detailed lung models with lobes"""
        # Left lung (2 lobes)
        left_lung_upper = vtk.vtkSphereSource()
        left_lung_upper.SetRadius(20)
        left_lung_upper.SetPhiResolution(15)
        left_lung_upper.SetThetaResolution(15)

        lung_mapper = vtk.vtkPolyDataMapper()
        lung_mapper.SetInputConnection(left_lung_upper.GetOutputPort())

        left_upper_actor = vtk.vtkActor()
        left_upper_actor.SetMapper(lung_mapper)
        left_upper_actor.GetProperty().SetColor(0.9, 0.7, 0.7)
        left_upper_actor.SetPosition(-40, -20, 130)

        self.models['left_lung_upper'] = {
            'actor': left_upper_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(left_upper_actor)

        left_lower_actor = vtk.vtkActor()
        left_lower_actor.SetMapper(lung_mapper)
        left_lower_actor.GetProperty().SetColor(0.85, 0.65, 0.65)
        left_lower_actor.SetPosition(-40, 10, 110)

        self.models['left_lung_lower'] = {
            'actor': left_lower_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(left_lower_actor)

        # Right lung (3 lobes)
        right_upper_actor = vtk.vtkActor()
        right_upper_actor.SetMapper(lung_mapper)
        right_upper_actor.GetProperty().SetColor(0.9, 0.7, 0.7)
        right_upper_actor.SetPosition(40, -20, 130)

        self.models['right_lung_upper'] = {
            'actor': right_upper_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(right_upper_actor)

        right_middle_actor = vtk.vtkActor()
        right_middle_actor.SetMapper(lung_mapper)
        right_middle_actor.GetProperty().SetColor(0.87, 0.67, 0.67)
        right_middle_actor.SetPosition(40, 5, 120)

        self.models['right_lung_middle'] = {
            'actor': right_middle_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(right_middle_actor)

        right_lower_actor = vtk.vtkActor()
        right_lower_actor.SetMapper(lung_mapper)
        right_lower_actor.GetProperty().SetColor(0.85, 0.65, 0.65)
        right_lower_actor.SetPosition(40, 20, 100)

        self.models['right_lung_lower'] = {
            'actor': right_lower_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(right_lower_actor)

    def create_brain_model(self):
        """Create detailed brain model"""
        # Cerebrum
        cerebrum = vtk.vtkSphereSource()
        cerebrum.SetRadius(22)
        cerebrum.SetPhiResolution(20)
        cerebrum.SetThetaResolution(20)

        brain_mapper = vtk.vtkPolyDataMapper()
        brain_mapper.SetInputConnection(cerebrum.GetOutputPort())

        brain_actor = vtk.vtkActor()
        brain_actor.SetMapper(brain_mapper)
        brain_actor.GetProperty().SetColor(0.9, 0.7, 0.8)
        brain_actor.SetPosition(0, 0, 200)

        self.models['brain'] = {
            'actor': brain_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(brain_actor)

        # Cerebellum
        cerebellum = vtk.vtkSphereSource()
        cerebellum.SetRadius(12)
        cerebellum.SetPhiResolution(15)
        cerebellum.SetThetaResolution(15)

        cerebellum_mapper = vtk.vtkPolyDataMapper()
        cerebellum_mapper.SetInputConnection(cerebellum.GetOutputPort())

        cerebellum_actor = vtk.vtkActor()
        cerebellum_actor.SetMapper(cerebellum_mapper)
        cerebellum_actor.GetProperty().SetColor(0.8, 0.6, 0.7)
        cerebellum_actor.SetPosition(0, -15, 190)

        self.models['cerebellum'] = {
            'actor': cerebellum_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(cerebellum_actor)

    def create_digestive_system(self):
        """Create digestive system organs"""
        # Stomach
        stomach = vtk.vtkSphereSource()
        stomach.SetRadius(15)
        stomach.SetPhiResolution(15)
        stomach.SetThetaResolution(15)

        stomach_mapper = vtk.vtkPolyDataMapper()
        stomach_mapper.SetInputConnection(stomach.GetOutputPort())

        stomach_actor = vtk.vtkActor()
        stomach_actor.SetMapper(stomach_mapper)
        stomach_actor.GetProperty().SetColor(0.7, 0.5, 0.5)
        stomach_actor.SetPosition(-25, 40, 100)

        self.models['stomach'] = {
            'actor': stomach_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(stomach_actor)

        # Liver
        liver = vtk.vtkSphereSource()
        liver.SetRadius(18)
        liver.SetPhiResolution(15)
        liver.SetThetaResolution(15)

        liver_mapper = vtk.vtkPolyDataMapper()
        liver_mapper.SetInputConnection(liver.GetOutputPort())

        liver_actor = vtk.vtkActor()
        liver_actor.SetMapper(liver_mapper)
        liver_actor.GetProperty().SetColor(0.5, 0.3, 0.3)
        liver_actor.SetPosition(30, 50, 95)

        self.models['liver'] = {
            'actor': liver_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(liver_actor)

        # Kidneys
        kidney = vtk.vtkSphereSource()
        kidney.SetRadius(8)
        kidney.SetPhiResolution(12)
        kidney.SetThetaResolution(12)

        kidney_mapper = vtk.vtkPolyDataMapper()
        kidney_mapper.SetInputConnection(kidney.GetOutputPort())

        left_kidney_actor = vtk.vtkActor()
        left_kidney_actor.SetMapper(kidney_mapper)
        left_kidney_actor.GetProperty().SetColor(0.6, 0.4, 0.4)
        left_kidney_actor.SetPosition(-35, 70, 90)

        self.models['left_kidney'] = {
            'actor': left_kidney_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(left_kidney_actor)

        right_kidney_actor = vtk.vtkActor()
        right_kidney_actor.SetMapper(kidney_mapper)
        right_kidney_actor.GetProperty().SetColor(0.6, 0.4, 0.4)
        right_kidney_actor.SetPosition(35, 70, 90)

        self.models['right_kidney'] = {
            'actor': right_kidney_actor,
            'layer': 'organs',
            'visible': True
        }
        self.renderer.AddActor(right_kidney_actor)

    def create_muscular_system(self):
        """Create simplified muscular system"""
        # Pectoral muscles
        pectoral = vtk.vtkSphereSource()
        pectoral.SetRadius(35)
        pectoral.SetPhiResolution(15)
        pectoral.SetThetaResolution(15)

        muscle_mapper = vtk.vtkPolyDataMapper()
        muscle_mapper.SetInputConnection(pectoral.GetOutputPort())

        pectoral_actor = vtk.vtkActor()
        pectoral_actor.SetMapper(muscle_mapper)
        pectoral_actor.GetProperty().SetColor(0.8, 0.6, 0.5)
        pectoral_actor.SetPosition(0, -10, 120)

        self.models['pectoral_muscles'] = {
            'actor': pectoral_actor,
            'layer': 'muscles',
            'visible': False
        }
        self.renderer.AddActor(pectoral_actor)

    def create_nervous_system(self):
        """Create simplified nervous system"""
        # Spinal cord
        spinal_cord = vtk.vtkCylinderSource()
        spinal_cord.SetRadius(3)
        spinal_cord.SetHeight(160)
        spinal_cord.SetResolution(12)

        nerve_mapper = vtk.vtkPolyDataMapper()
        nerve_mapper.SetInputConnection(spinal_cord.GetOutputPort())

        spinal_cord_actor = vtk.vtkActor()
        spinal_cord_actor.SetMapper(nerve_mapper)
        spinal_cord_actor.GetProperty().SetColor(0.9, 0.9, 0.2)
        spinal_cord_actor.SetPosition(0, 0, 80)
        spinal_cord_actor.RotateX(90)

        self.models['spinal_cord'] = {
            'actor': spinal_cord_actor,
            'layer': 'nerves',
            'visible': False
        }
        self.renderer.AddActor(spinal_cord_actor)

    def set_layer_visibility(self, layer_name, visible):
        """Set visibility for entire layer"""
        if layer_name in self.layers:
            self.layers[layer_name]['visible'] = visible

            # Update all models in this layer
            for model_name, model_data in self.models.items():
                if model_data['layer'] == layer_name:
                    model_data['actor'].SetVisibility(visible)
                    model_data['visible'] = visible

    def set_layer_opacity(self, layer_name, opacity):
        """Set opacity for entire layer"""
        if layer_name in self.layers:
            self.layers[layer_name]['opacity'] = opacity

            # Update all models in this layer
            for model_name, model_data in self.models.items():
                if model_data['layer'] == layer_name:
                    model_data['actor'].GetProperty().SetOpacity(opacity)

    def set_model_visibility(self, model_name, visible):
        """Set visibility for specific model"""
        if model_name in self.models:
            self.models[model_name]['visible'] = visible
            self.models[model_name]['actor'].SetVisibility(visible)

    def rotate_all(self, x_angle, y_angle, z_angle):
        """Rotate all models"""
        for model_data in self.models.values():
            actor = model_data['actor']
            actor.RotateX(x_angle)
            actor.RotateY(y_angle)
            actor.RotateZ(z_angle)

    def reset_view(self):
        """Reset camera to default position"""
        self.camera.SetPosition(0, -300, 300)
        self.camera.SetFocalPoint(0, 0, 0)
        self.camera.SetViewUp(0, 0, 1)

        # Reset rotations
        for model_data in self.models.values():
            model_data['actor'].SetOrientation(0, 0, 0)

    def focus_on_organ(self, organ_name):
        """Dynamically focus camera on a specific organ"""
        if organ_name in self.models:
            actor = self.models[organ_name]['actor']
            pos = actor.GetPosition()
            
            # Smoothly move camera (simplified for now, just set)
            self.camera.SetFocalPoint(pos)
            
            # Adjust camera position to be in front of the organ
            # Keep the same relative direction but move closer/further
            current_pos = np.array(self.camera.GetPosition())
            focal_point = np.array(pos)
            direction = current_pos - focal_point
            direction = direction / np.linalg.norm(direction)
            
            # Set new position at a fixed distance
            new_pos = focal_point + direction * 200  # 200 units distance
            self.camera.SetPosition(new_pos)
            self.camera.SetViewUp(0, 0, 1)
            self.render_window.Render()

    def render_to_image(self):
        """Render 3D scene to numpy image"""
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
        
        # VTK returns RGB, OpenCV expects BGR
        if components == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        
        # Flip image vertically (VTK origin is bottom-left)
        arr = np.flipud(arr)
        
        return arr

class UnifiedAnatomyController:
    def __init__(self, anatomy_system):
        self.anatomy = anatomy_system

    def handle_rotation(self, dx, dy):
        """Handle rotation based on mouse/gesture movement."""
        self.anatomy.renderer.GetActiveCamera().Azimuth(dx * 0.1)
        self.anatomy.renderer.GetActiveCamera().Elevation(dy * 0.1)
        self.anatomy.render_window.Render()

    def handle_zoom(self, factor):
        """Handle zoom."""
        self.anatomy.renderer.GetActiveCamera().Dolly(factor)
        self.anatomy.render_window.Render()

    def switch_layer(self, layer_name):
        """Switch the current visible layer."""
        if layer_name in self.anatomy.layers:
            self.anatomy.current_layer = layer_name
            # Update visibility for all models
            for model_name, model_data in self.anatomy.models.items():
                is_visible = model_data['layer'] == layer_name
                model_data['actor'].SetVisibility(is_visible)
            self.anatomy.render_window.Render()

        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        arr = cv2.flip(arr, 0)  # VTK image is upside down

        return arr

    def add_anatomical_labels(self):
        """Add anatomical labels to the model"""
        labels = {
            'heart_main': (0, -40, 120, "Heart"),
            'brain': (0, 0, 200, "Brain"),
            'left_lung_upper': (-40, -20, 130, "Left Lung"),
            'right_lung_upper': (40, -20, 130, "Right Lung"),
            'liver': (30, 50, 95, "Liver"),
            'stomach': (-25, 40, 100, "Stomach"),
            'left_kidney': (-35, 70, 90, "Left Kidney"),
            'right_kidney': (35, 70, 90, "Right Kidney")
        }

        for model_name, (x, y, z, text) in labels.items():
            if model_name in self.models:
                text_actor = vtk.vtkTextActor3D()
                text_actor.SetInput(text)
                text_actor.SetPosition(x, y, z + 30)
                text_actor.GetTextProperty().SetFontSize(15)
                text_actor.GetTextProperty().SetColor(1, 1, 1)

                self.models[f'label_{model_name}'] = {
                    'actor': text_actor,
                    'layer': 'organs',
                    'visible': True
                }
                self.renderer.AddActor(text_actor)

    def load_complete_anatomy(self):
        """Load complete anatomical system"""
        print("Loading complete anatomical system...")
        self.create_complete_skeleton()
        self.create_detailed_heart()
        self.create_detailed_lungs()
        self.create_brain_model()
        self.create_digestive_system()
        self.create_muscular_system()
        self.create_nervous_system()
        self.add_anatomical_labels()

        # Set initial layer visibility
        self.set_layer_visibility('skeleton', False)
        self.set_layer_visibility('organs', True)
        self.set_layer_visibility('muscles', False)
        self.set_layer_visibility('nerves', False)
        self.set_layer_visibility('skin', False)

        print("Anatomical system loaded successfully!")
        print(f"Total models: {len(self.models)}")


class UnifiedAnatomyController:
    def __init__(self, anatomy_system):
        self.anatomy = anatomy_system
        self.rotation_speed = 2.0
        self.zoom_speed = 1.1

    def handle_rotation(self, dx, dy):
        """Handle rotation gestures"""
        self.anatomy.rotate_all(dy * self.rotation_speed,
                                dx * self.rotation_speed, 0)

    def handle_zoom(self, zoom_factor):
        """Handle zoom gestures"""
        camera = self.anatomy.camera
        camera.Zoom(zoom_factor)

    def switch_layer(self, layer_name):
        """Switch between anatomical layers"""
        if layer_name in self.anatomy.layers:
            # Hide all layers first
            for layer in self.anatomy.layers:
                self.anatomy.set_layer_visibility(layer, False)

            # Show selected layer
            self.anatomy.set_layer_visibility(layer_name, True)
            self.anatomy.current_layer = layer_name

            # Adjust opacity based on layer
            if layer_name == 'skin':
                self.anatomy.set_layer_opacity(layer_name, 0.3)
            elif layer_name == 'muscles':
                self.anatomy.set_layer_opacity(layer_name, 0.7)
            elif layer_name == 'organs':
                self.anatomy.set_layer_opacity(layer_name, 0.8)
            elif layer_name == 'skeleton':
                self.anatomy.set_layer_opacity(layer_name, 0.9)
            elif layer_name == 'nerves':
                self.anatomy.set_layer_opacity(layer_name, 0.6)


def demo_unified_anatomy():
    """Demo the unified anatomy system"""
    anatomy = UnifiedAnatomySystem()
    controller = UnifiedAnatomyController(anatomy)

    # Load complete anatomy
    anatomy.load_complete_anatomy()

    print("\n=== Unified Anatomy System ===")
    print("Controls:")
    print("  1 - Skeleton Layer")
    print("  2 - Organs Layer")
    print("  3 - Muscles Layer")
    print("  4 - Nerves Layer")
    print("  5 - Skin Layer")
    print("  R - Reset View")
    print("  Q - Quit")
    print("  Mouse/gesture - Rotate and zoom")
    print("==============================\n")

    # Auto-rotation for demo
    auto_rotate = True

    while True:
        if auto_rotate:
            anatomy.rotate_all(0, 1, 0)

        # Render to image
        image = anatomy.render_to_image()

        # Add UI overlay
        cv2.putText(image, f"Current Layer: {anatomy.current_layer.upper()}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Press 1-5 to switch layers, R to reset, Q to quit",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Unified Anatomy System', image)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            controller.switch_layer('skeleton')
        elif key == ord('2'):
            controller.switch_layer('organs')
        elif key == ord('3'):
            controller.switch_layer('muscles')
        elif key == ord('4'):
            controller.switch_layer('nerves')
        elif key == ord('5'):
            controller.switch_layer('skin')
        elif key == ord('r'):
            anatomy.reset_view()
        elif key == ord(' '):
            auto_rotate = not auto_rotate

    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo_unified_anatomy()