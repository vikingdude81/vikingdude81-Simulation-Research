
import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'osmesa'

import numpy as np
import pyvista as pv
from math import pi, sqrt, sin, cos
import random

pv.OFF_SCREEN = True

def bloch_vector(theta, phi):
    """Convert spherical angles to Bloch vector"""
    return np.array([
        sin(theta) * cos(phi),
        sin(theta) * sin(phi),
        cos(theta)
    ])

def rotation_axis_angle(axis, angle, vector):
    """Rotate vector around axis by angle (Rodrigues' formula)"""
    axis = axis / np.linalg.norm(axis)
    return (vector * cos(angle) + 
            np.cross(axis, vector) * sin(angle) +
            axis * np.dot(axis, vector) * (1 - cos(angle)))

def create_bloch_sphere():
    """Create the Bloch sphere with axes"""
    sphere = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
    
    # Create axes
    x_axis = pv.Line((-1.2, 0, 0), (1.2, 0, 0))
    y_axis = pv.Line((0, -1.2, 0), (0, 1.2, 0))
    z_axis = pv.Line((0, 0, -1.2), (0, 0, 1.2))
    
    return sphere, x_axis, y_axis, z_axis

def create_state_vector_arrow(bloch_vec):
    """Create arrow representing quantum state"""
    start = np.array([0, 0, 0])
    direction = bloch_vec * 1.0
    
    arrow = pv.Arrow(start=start, direction=direction, 
                     tip_length=0.2, tip_radius=0.08, 
                     shaft_radius=0.03, scale=1.0)
    return arrow

def plot_static_qubit_state():
    """Create static visualization of random qubit state"""
    theta = random.uniform(0, pi)
    phi = random.uniform(0, 2*pi)
    
    bloch_vec = bloch_vector(theta, phi)
    
    state_name = ""
    if abs(theta) < 0.1:
        state_name = "|0⟩ (North pole)"
    elif abs(theta - pi) < 0.1:
        state_name = "|1⟩ (South pole)"
    elif abs(theta - pi/2) < 0.1:
        if abs(phi) < 0.1 or abs(phi - 2*pi) < 0.1:
            state_name = "|+⟩ = (|0⟩+|1⟩)/√2"
        elif abs(phi - pi) < 0.1:
            state_name = "|−⟩ = (|0⟩−|1⟩)/√2"
        elif abs(phi - pi/2) < 0.1:
            state_name = "|+i⟩ = (|0⟩+i|1⟩)/√2"
        elif abs(phi - 3*pi/2) < 0.1:
            state_name = "|−i⟩ = (|0⟩−i|1⟩)/√2"
    
    print("=" * 60)
    print("  BLOCH SPHERE - QUBIT STATE VISUALIZATION")
    print("=" * 60)
    print(f"\n🎲 Random qubit state:")
    print(f"   θ = {theta:.3f} rad ({theta*180/pi:.1f}°)")
    print(f"   φ = {phi:.3f} rad ({phi*180/pi:.1f}°)")
    if state_name:
        print(f"   Special state: {state_name}")
    print(f"   Bloch vector: [{bloch_vec[0]:.3f}, {bloch_vec[1]:.3f}, {bloch_vec[2]:.3f}]")
    print()
    
    print("🎨 Creating Bloch sphere visualization...")
    
    pl = pv.Plotter(window_size=[800, 800], off_screen=True)
    pl.set_background('#0a0a1a')
    
    # Add sphere
    sphere, x_axis, y_axis, z_axis = create_bloch_sphere()
    pl.add_mesh(sphere, color='lightblue', opacity=0.2, show_edges=False)
    
    # Add axes
    pl.add_mesh(x_axis, color='red', line_width=3, label='X')
    pl.add_mesh(y_axis, color='green', line_width=3, label='Y')
    pl.add_mesh(z_axis, color='blue', line_width=3, label='Z')
    
    # Add labels
    pl.add_point_labels([[1.3, 0, 0]], ['|+⟩'], text_color='red', font_size=20)
    pl.add_point_labels([[-1.3, 0, 0]], ['|−⟩'], text_color='red', font_size=20)
    pl.add_point_labels([[0, 0, 1.3]], ['|0⟩'], text_color='blue', font_size=24, bold=True)
    pl.add_point_labels([[0, 0, -1.3]], ['|1⟩'], text_color='blue', font_size=24, bold=True)
    
    # Add state vector
    arrow = create_state_vector_arrow(bloch_vec)
    pl.add_mesh(arrow, color='yellow', lighting=True)
    
    # Add state point
    pl.add_mesh(pv.Sphere(radius=0.05, center=bloch_vec), 
                color='yellow', lighting=True)
    
    title = f"Bloch Sphere: θ={theta*180/pi:.1f}°, φ={phi*180/pi:.1f}°"
    pl.add_text(title, position='upper_edge', color='white', font_size=12)
    
    pl.camera.position = (2, 2, 2)
    pl.camera.focal_point = (0, 0, 0)
    
    filename = f'bloch_sphere_theta_{theta*180/pi:.0f}_phi_{phi*180/pi:.0f}.png'
    pl.screenshot(filename)
    print(f"✓ Saved: {filename}")
    pl.close()

def create_rotation_animation():
    """Animate qubit state rotation around random axis"""
    # Random initial state
    theta0 = random.uniform(0, pi)
    phi0 = random.uniform(0, 2*pi)
    initial_vec = bloch_vector(theta0, phi0)
    
    # Random rotation axis
    axis_theta = random.uniform(0, pi)
    axis_phi = random.uniform(0, 2*pi)
    rotation_axis = bloch_vector(axis_theta, axis_phi)
    
    print("🎬 Creating rotation animation...")
    print(f"   Initial state: θ={theta0*180/pi:.1f}°, φ={phi0*180/pi:.1f}°")
    print(f"   Rotation axis: [{rotation_axis[0]:.2f}, {rotation_axis[1]:.2f}, {rotation_axis[2]:.2f}]")
    
    pl = pv.Plotter(window_size=[800, 800], off_screen=True)
    pl.set_background('#0a0a1a')
    
    filename = 'bloch_rotation_animation.gif'
    pl.open_gif(filename, fps=20)
    
    n_frames = 60
    for i in range(n_frames):
        angle = 2 * pi * i / n_frames
        
        # Rotate state vector
        current_vec = rotation_axis_angle(rotation_axis, angle, initial_vec)
        
        pl.clear()
        
        # Add sphere and axes
        sphere, x_axis, y_axis, z_axis = create_bloch_sphere()
        pl.add_mesh(sphere, color='lightblue', opacity=0.15, show_edges=False)
        pl.add_mesh(x_axis, color='red', line_width=2)
        pl.add_mesh(y_axis, color='green', line_width=2)
        pl.add_mesh(z_axis, color='blue', line_width=2)
        
        # Add rotation axis
        axis_arrow = pv.Arrow(start=-rotation_axis*1.5, direction=rotation_axis*3,
                             tip_length=0.15, shaft_radius=0.02)
        pl.add_mesh(axis_arrow, color='magenta', opacity=0.5)
        
        # Add state vector
        arrow = create_state_vector_arrow(current_vec)
        pl.add_mesh(arrow, color='yellow', lighting=True)
        pl.add_mesh(pv.Sphere(radius=0.06, center=current_vec), 
                   color='yellow', lighting=True)
        
        # Add trace
        if i > 0:
            trace_angles = np.linspace(0, angle, min(i, 30))
            trace_points = [rotation_axis_angle(rotation_axis, a, initial_vec) 
                          for a in trace_angles]
            trace_points = np.array(trace_points)
            if len(trace_points) > 1:
                trace_line = pv.Spline(trace_points, n_points=100)
                pl.add_mesh(trace_line, color='cyan', line_width=3, opacity=0.7)
        
        title = f"Qubit Rotation: {angle*180/pi:.0f}°"
        pl.add_text(title, position='upper_edge', color='white', font_size=12)
        
        pl.camera.position = (2.5, 2.5, 2.5)
        pl.camera.focal_point = (0, 0, 0)
        
        pl.write_frame()
    
    pl.close()
    print(f"✓ Saved: {filename}")

def main():
    plot_static_qubit_state()
    create_rotation_animation()
    
    print("\n" + "=" * 60)
    print("Bloch sphere visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
