import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import deque
import colorsys
import json


@dataclass
class Neuron:
    """Digital neuron with physics properties"""
    x: float
    y: float
    vx: float = 0
    vy: float = 0
    radius: float = 5
    activation: float = 0.0
    target_activation: float = 0.0
    connections: List[int] = field(default_factory=list)
    color: Tuple[int, int, int] = (100, 150, 255)
    energy: float = 100.0
    age: int = 0
    pulse_phase: float = 0.0
    layer: int = 0

    def update(self, dt: float, width: float, height: float, friction: float = 0.98):
        """Update physics and activation"""
        # Physics
        self.vx *= friction
        self.vy *= friction
        self.x += self.vx * dt * 60
        self.y += self.vy * dt * 60

        # Boundary bounce with energy loss
        margin = self.radius
        if self.x < margin:
            self.x = margin
            self.vx *= -0.8
        elif self.x > width - margin:
            self.x = width - margin
            self.vx *= -0.8

        if self.y < margin:
            self.y = margin
            self.vy *= -0.8
        elif self.y > height - margin:
            self.y = height - margin
            self.vy *= -0.8

        # Smooth activation transition
        self.activation += (self.target_activation - self.activation) * 0.1
        self.pulse_phase += 0.05 + self.activation * 0.1
        self.age += 1

        # Energy decay/regeneration
        self.energy = max(0, min(100, self.energy - 0.1 + self.activation * 0.5))


class Synapse:
    """Connection between neurons with signal transmission"""

    def __init__(self, source_idx: int, target_idx: int, strength: float = 1.0):
        self.source = source_idx
        self.target = target_idx
        self.strength = strength
        self.active_signal = 0.0
        self.transmission_progress = 0.0
        self.history = deque(maxlen=50)
        self.learning_rate = 0.01

    def transmit(self, source_activation: float):
        """Transmit signal from source to target"""
        if source_activation > 0.3:
            self.active_signal = source_activation * self.strength
            self.transmission_progress = 0.0

    def update(self, dt: float):
        """Update signal transmission"""
        if self.active_signal > 0:
            self.transmission_progress += dt * 2
            if self.transmission_progress >= 1.0:
                self.active_signal *= 0.9
            self.history.append(self.active_signal)
        else:
            self.active_signal = 0

    def strengthen(self, amount: float):
        """Hebbian learning - strengthen connection"""
        self.strength = min(2.0, self.strength + amount * self.learning_rate)


class NeuralNetwork:
    """Manages the neural colony"""

    def __init__(self, canvas_width: float, canvas_height: float):
        self.neurons: List[Neuron] = []
        self.synapses: List[Synapse] = []
        self.width = canvas_width
        self.height = canvas_height
        self.next_id = 0
        self.activity_history = deque(maxlen=100)
        self.global_inhibition = 0.0

    def add_neuron(self, x: float, y: float, layer: int = 0) -> int:
        """Add new neuron to network"""
        hue = random.random()
        rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.7, 0.9))
        neuron = Neuron(x, y, layer=layer, color=rgb)
        self.neurons.append(neuron)
        idx = len(self.neurons) - 1

        # Auto-connect to nearby neurons
        self._form_connections(idx)
        return idx

    def _form_connections(self, idx: int, max_dist: float = 150):
        """Form synaptic connections based on proximity"""
        neuron = self.neurons[idx]
        for i, other in enumerate(self.neurons):
            if i != idx:
                dist = math.hypot(neuron.x - other.x, neuron.y - other.y)
                if dist < max_dist and random.random() < 0.3:
                    synapse = Synapse(idx, i, random.uniform(0.1, 1.0))
                    self.synapses.append(synapse)
                    neuron.connections.append(len(self.synapses) - 1)

                    # Reciprocal connection
                    rev_synapse = Synapse(i, idx, random.uniform(0.1, 1.0))
                    self.synapses.append(rev_synapse)
                    other.connections.append(len(self.synapses) - 1)

    def stimulate(self, x: float, y: float, radius: float = 100, intensity: float = 1.0):
        """Stimulate neurons in area"""
        for neuron in self.neurons:
            dist = math.hypot(neuron.x - x, neuron.y - y)
            if dist < radius:
                neuron.target_activation = intensity * (1 - dist / radius)
                neuron.energy = min(100, neuron.energy + 20)

    def update(self, dt: float):
        """Update entire network"""
        total_activity = 0

        # Update neurons
        for neuron in self.neurons:
            neuron.update(dt, self.width, self.height)
            total_activity += neuron.activation

            # Spontaneous activation based on energy
            if neuron.energy > 80 and random.random() < 0.001:
                neuron.target_activation = random.uniform(0.5, 1.0)

        # Update synapses and propagate signals
        for synapse in self.synapses:
            synapse.update(dt)
            if synapse.active_signal > 0.5 and synapse.transmission_progress >= 1.0:
                target = self.neurons[synapse.target]
                target.target_activation = min(1.0, target.target_activation + synapse.active_signal * 0.3)
                synapse.transmit(0)  # Reset

        # Global inhibition (homeostasis)
        avg_activity = total_activity / max(1, len(self.neurons))
        self.activity_history.append(avg_activity)
        if avg_activity > 0.5:
            self.global_inhibition = min(0.5, self.global_inhibition + 0.01)
        else:
            self.global_inhibition = max(0, self.global_inhibition - 0.01)

        # Apply inhibition
        for neuron in self.neurons:
            neuron.target_activation = max(0, neuron.target_activation - self.global_inhibition * 0.1)

    def get_stats(self) -> Dict:
        """Get network statistics"""
        return {
            'neurons': len(self.neurons),
            'synapses': len(self.synapses),
            'avg_activation': sum(n.activation for n in self.neurons) / max(1, len(self.neurons)),
            'avg_energy': sum(n.energy for n in self.neurons) / max(1, len(self.neurons))
        }


class AdvancedRenderer:
    """High-quality rendering engine"""

    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.particles = []
        self.trails = {}
        self.glow_effects = {}

    def draw_glow(self, x: float, y: float, radius: float, color: Tuple[int, int, int], intensity: float):
        """Draw glowing effect using multiple circles"""
        r, g, b = color
        for i in range(3):
            alpha = int(intensity * (100 - i * 30))
            size = radius * (1 + i * 0.5)
            self.canvas.create_oval(
                x - size, y - size, x + size, y + size,
                fill=f'#{r:02x}{g:02x}{b:02x}',
                outline='',
                stipple='gray50' if i > 0 else ''
            )

    def draw_connection(self, x1: float, y1: float, x2: float, y2: float,
                        strength: float, signal: float, color: Tuple[int, int, int]):
        """Draw synaptic connection with signal visualization"""
        if strength < 0.1:
            return

        r, g, b = color
        width = max(1, strength * 3)

        # Base connection
        self.canvas.create_line(
            x1, y1, x2, y2,
            fill=f'#{int(r * 0.5):02x}{int(g * 0.5):02x}{int(b * 0.5):02x}',
            width=width,
            stipple='gray25'
        )

        # Active signal
        if signal > 0.1:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            pulse_x = x1 + (x2 - x1) * (signal % 1.0)
            pulse_y = y1 + (y2 - y1) * (signal % 1.0)

            self.canvas.create_oval(
                pulse_x - 3, pulse_y - 3, pulse_x + 3, pulse_y + 3,
                fill=f'#{r:02x}{g:02x}{b:02x}',
                outline='white'
            )


class NeuralCanvasApp:
    """Main application with advanced UI"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Neural Canvas - Generative Art System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0a0a0a')

        # State
        self.running = False
        self.last_time = time.time()
        self.mouse_pos = (0, 0)
        self.selected_tool = "stimulate"
        self.show_connections = True
        self.show_glow = True
        self.particle_trails = True
        self.recording = False
        self.frame_count = 0

        # Network
        self.network = None
        self.renderer = None

        # UI Colors
        self.bg_color = '#0a0a0a'
        self.fg_color = '#00ff88'
        self.accent_color = '#ff0066'

        self._setup_ui()
        self._init_network()

    def _setup_ui(self):
        """Create sophisticated UI"""
        # Main container
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left sidebar - Controls
        sidebar = tk.Frame(main_container, bg='#1a1a1a', width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)

        # Title
        title = tk.Label(
            sidebar,
            text="⚡ NEURAL CANVAS",
            font=('Courier', 16, 'bold'),
            bg='#1a1a1a',
            fg=self.fg_color
        )
        title.pack(pady=20)

        # Control sections
        self._create_control_section(sidebar, "Network Controls", [
            ("Initialize", self._init_network),
            ("Add Random Neurons (10)", lambda: self._add_random_neurons(10)),
            ("Clear All", self._clear_network),
            ("Save State", self._save_state),
            ("Load State", self._load_state),
        ])

        self._create_control_section(sidebar, "Stimulation", [
            ("Random Stimulation", self._random_stimulation),
            ("Wave Pattern", self._wave_pattern),
            ("Chaos Mode", self._chaos_mode),
        ])

        # Parameters
        param_frame = tk.LabelFrame(sidebar, text="Parameters", bg='#1a1a1a', fg='white')
        param_frame.pack(fill=tk.X, padx=10, pady=10)

        self.connection_dist = tk.Scale(
            param_frame, from_=50, to=300, orient=tk.HORIZONTAL,
            label="Connection Distance", bg='#1a1a1a', fg='white',
            highlightthickness=0
        )
        self.connection_dist.set(150)
        self.connection_dist.pack(fill=tk.X, padx=5)

        self.friction_slider = tk.Scale(
            param_frame, from_=0.90, to=0.99, resolution=0.001, orient=tk.HORIZONTAL,
            label="Friction", bg='#1a1a1a', fg='white',
            highlightthickness=0
        )
        self.friction_slider.set(0.98)
        self.friction_slider.pack(fill=tk.X, padx=5)

        # Visualization options
        viz_frame = tk.LabelFrame(sidebar, text="Visualization", bg='#1a1a1a', fg='white')
        viz_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Checkbutton(
            viz_frame, text="Show Connections", variable=tk.BooleanVar(value=True),
            command=self._toggle_connections, bg='#1a1a1a', fg='white',
            selectcolor='#333333'
        ).pack(anchor=tk.W)

        tk.Checkbutton(
            viz_frame, text="Glow Effects", variable=tk.BooleanVar(value=True),
            command=self._toggle_glow, bg='#1a1a1a', fg='white',
            selectcolor='#333333'
        ).pack(anchor=tk.W)

        tk.Checkbutton(
            viz_frame, text="Particle Trails", variable=tk.BooleanVar(value=True),
            command=self._toggle_trails, bg='#1a1a1a', fg='white',
            selectcolor='#333333'
        ).pack(anchor=tk.W)

        # Stats display
        self.stats_label = tk.Label(
            sidebar, text="Neurons: 0\nSynapses: 0",
            font=('Courier', 10), bg='#1a1a1a', fg=self.fg_color,
            justify=tk.LEFT
        )
        self.stats_label.pack(pady=20)

        # Canvas area
        canvas_container = tk.Frame(main_container, bg=self.bg_color)
        canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_container, bg=self.bg_color, highlightthickness=0,
            width=1000, height=800
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind events
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Motion>", self._on_motion)
        self.root.bind("<space>", lambda e: self._toggle_pause())
        self.root.bind("s", lambda e: self._save_screenshot())

        # Control bar at bottom
        control_bar = tk.Frame(canvas_container, bg='#1a1a1a', height=40)
        control_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.play_btn = tk.Button(
            control_bar, text="▶ Play", command=self._toggle_pause,
            bg=self.fg_color, fg='black', font=('Courier', 10, 'bold')
        )
        self.play_btn.pack(side=tk.LEFT, padx=10, pady=5)

        tk.Button(
            control_bar, text="📷 Screenshot", command=self._save_screenshot,
            bg='#333333', fg='white'
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_bar, text="🎨 Export Art", command=self._export_art,
            bg=self.accent_color, fg='white'
        ).pack(side=tk.LEFT, padx=5)

        # Status
        self.status_label = tk.Label(
            control_bar, text="Ready", bg='#1a1a1a', fg='white'
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

    def _create_control_section(self, parent, title, buttons):
        """Create a section of control buttons"""
        frame = tk.LabelFrame(parent, text=title, bg='#1a1a1a', fg='white')
        frame.pack(fill=tk.X, padx=10, pady=5)

        for text, command in buttons:
            btn = tk.Button(
                frame, text=text, command=command,
                bg='#333333', fg='white', relief=tk.FLAT,
                activebackground=self.fg_color, activeforeground='black'
            )
            btn.pack(fill=tk.X, padx=5, pady=2)

    def _init_network(self):
        """Initialize neural network"""
        width = self.canvas.winfo_width() or 1000
        height = self.canvas.winfo_height() or 800
        self.network = NeuralNetwork(width, height)
        self.renderer = AdvancedRenderer(self.canvas)

        # Create initial structure
        center_x, center_y = width / 2, height / 2
        for i in range(5):
            angle = (i / 5) * 2 * math.pi
            x = center_x + math.cos(angle) * 200
            y = center_y + math.sin(angle) * 200
            self.network.add_neuron(x, y)

        self._start_loop()

    def _start_loop(self):
        """Start animation loop"""
        if not self.running:
            self.running = True
            self._animate()

    def _animate(self):
        """Main animation loop"""
        if not self.running:
            return

        current_time = time.time()
        dt = min(current_time - self.last_time, 0.1)
        self.last_time = current_time

        # Update network
        if self.network:
            self.network.update(dt)
            self._render()
            self._update_stats()

        self.frame_count += 1
        self.root.after(16, self._animate)  # ~60 FPS

    def _render(self):
        """Render the neural network"""
        self.canvas.delete("all")

        # Draw connections first (behind neurons)
        if self.show_connections:
            for synapse in self.network.synapses:
                source = self.network.neurons[synapse.source]
                target = self.network.neurons[synapse.target]

                # Calculate color based on activity
                intensity = synapse.active_signal
                color = (
                    int(source.color[0] * (1 - intensity) + 255 * intensity),
                    int(source.color[1] * (1 - intensity) + 255 * intensity),
                    int(source.color[2] * (1 - intensity) + 255 * intensity)
                )

                self.renderer.draw_connection(
                    source.x, source.y, target.x, target.y,
                    synapse.strength, synapse.transmission_progress, color
                )

        # Draw neurons
        for neuron in self.network.neurons:
            # Calculate display properties
            pulse = math.sin(neuron.pulse_phase) * 0.3 + 0.7
            radius = neuron.radius * (1 + neuron.activation * 0.5) * pulse

            # Color based on activation
            r = int(neuron.color[0] * (0.3 + neuron.activation * 0.7))
            g = int(neuron.color[1] * (0.3 + neuron.activation * 0.7))
            b = int(neuron.color[2] * (0.3 + neuron.activation * 0.7))

            # Glow effect
            if self.show_glow and neuron.activation > 0.1:
                glow_radius = radius * (2 + neuron.activation * 2)
                self.renderer.draw_glow(
                    neuron.x, neuron.y, glow_radius, (r, g, b),
                    neuron.activation * 0.5
                )

            # Core
            self.canvas.create_oval(
                neuron.x - radius, neuron.y - radius,
                neuron.x + radius, neuron.y + radius,
                fill=f'#{r:02x}{g:02x}{b:02x}',
                outline='white' if neuron.activation > 0.5 else '',
                width=2
            )

            # Energy indicator
            if neuron.energy < 30:
                self.canvas.create_arc(
                    neuron.x - radius - 2, neuron.y - radius - 2,
                    neuron.x + radius + 2, neuron.y + radius + 2,
                    start=0, extent=360 * (neuron.energy / 100),
                    outline='red', width=2
                )

    def _update_stats(self):
        """Update statistics display"""
        if self.network:
            stats = self.network.get_stats()
            self.stats_label.config(
                text=f"Neurons: {stats['neurons']}\n"
                     f"Synapses: {stats['synapses']}\n"
                     f"Avg Activation: {stats['avg_activation']:.2f}\n"
                     f"Avg Energy: {stats['avg_energy']:.1f}\n"
                     f"FPS: {int(1 / 0.016)}"
            )

    def _on_click(self, event):
        """Handle mouse click"""
        if self.selected_tool == "stimulate":
            self.network.stimulate(event.x, event.y, 100, 1.0)
        elif self.selected_tool == "add":
            self.network.add_neuron(event.x, event.y)

    def _on_drag(self, event):
        """Handle mouse drag"""
        if self.selected_tool == "stimulate":
            self.network.stimulate(event.x, event.y, 80, 0.5)

    def _on_motion(self, event):
        self.mouse_pos = (event.x, event.y)

    def _add_random_neurons(self, count: int):
        """Add random neurons"""
        for _ in range(count):
            x = random.uniform(50, self.network.width - 50)
            y = random.uniform(50, self.network.height - 50)
            self.network.add_neuron(x, y)

    def _clear_network(self):
        """Clear all neurons"""
        self.network = NeuralNetwork(self.network.width, self.network.height)

    def _random_stimulation(self):
        """Random stimulation pattern"""
        for _ in range(5):
            x = random.uniform(0, self.network.width)
            y = random.uniform(0, self.network.height)
            self.network.stimulate(x, y, 150, random.uniform(0.5, 1.0))

    def _wave_pattern(self):
        """Create wave stimulation"""
        cx, cy = self.network.width / 2, self.network.height / 2
        for i, neuron in enumerate(self.network.neurons):
            angle = (i / max(1, len(self.network.neurons))) * 2 * math.pi
            delay = i * 50
            self.root.after(delay, lambda n=neuron: setattr(n, 'target_activation', 1.0))

    def _chaos_mode(self):
        """Chaotic activation"""
        for neuron in self.network.neurons:
            neuron.target_activation = random.random()
            neuron.vx = random.uniform(-5, 5)
            neuron.vy = random.uniform(-5, 5)

    def _toggle_pause(self):
        """Toggle animation"""
        self.running = not self.running
        self.play_btn.config(text="⏸ Pause" if self.running else "▶ Play")
        if self.running:
            self._animate()

    def _toggle_connections(self):
        self.show_connections = not self.show_connections

    def _toggle_glow(self):
        self.show_glow = not self.show_glow

    def _toggle_trails(self):
        self.particle_trails = not self.particle_trails

    def _save_screenshot(self):
        """Save canvas as image"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".ps",
            filetypes=[("PostScript", "*.ps"), ("All files", "*.*")]
        )
        if filename:
            self.canvas.postscript(file=filename, colormode="color")
            self.status_label.config(text=f"Saved: {filename}")

    def _export_art(self):
        """Export high-resolution art"""
        # Create larger canvas for export
        export_window = tk.Toplevel(self.root)
        export_window.title("Export Neural Art")
        export_canvas = tk.Canvas(export_window, width=1920, height=1080, bg=self.bg_color)
        export_canvas.pack()

        # Render current state at higher resolution
        # (Simplified - would need proper scaling logic)
        messagebox.showinfo("Export", "High-res export feature - integrate with PIL for PNG export")

    def _save_state(self):
        """Save network state to JSON"""
        filename = filedialog.asksaveasfilename(defaultextension=".json")
        if filename and self.network:
            state = {
                'neurons': [
                    {'x': n.x, 'y': n.y, 'activation': n.activation, 'energy': n.energy}
                    for n in self.network.neurons
                ],
                'synapses': [
                    {'source': s.source, 'target': s.target, 'strength': s.strength}
                    for s in self.network.synapses
                ]
            }
            with open(filename, 'w') as f:
                json.dump(state, f)

    def _load_state(self):
        """Load network state"""
        filename = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if filename:
            with open(filename, 'r') as f:
                state = json.load(f)
            self._init_network()
            for n_data in state['neurons']:
                self.network.add_neuron(n_data['x'], n_data['y'])


def main():
    root = tk.Tk()
    app = NeuralCanvasApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()