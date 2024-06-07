import numpy as np

def plot_initial_graph(self):
    self.ax.clear()
    self.ax.set_xlim(-50, 50)
    self.ax.set_ylim(-50, 50)
    self.ax.axhline(0, color='black', linewidth=0.5)
    self.ax.axvline(0, color='black', linewidth=0.5)
    self.ax.grid(color='gray', linestyle='--', linewidth=0.5)
    self.ax.set_xlabel("X")
    self.ax.set_ylabel("Y")
    self.canvas.draw()

def setup_drag_feature(self):
    self.dragging = False
    self.start_event = None
    self.canvas.mpl_connect("button_press_event", self.on_press)
    self.canvas.mpl_connect("button_release_event", self.on_release)
    self.canvas.mpl_connect("motion_notify_event", self.on_motion)

def setup_zoom_feature(self):
    self.canvas.mpl_connect("scroll_event", self.on_scroll)

def zoom_to_fit(self):
    if not self.data:
        return
    x_vals = [p[0] for p in self.data]
    y_vals = [p[1] for p in self.data]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    margin_x = 0.15 * (max_x - min_x)
    margin_y = 0.15 * (max_y - min_y)
    self.ax.set_xlim(min_x - margin_x, max_x + margin_x)
    self.ax.set_ylim(min_y - margin_y, max_y + margin_y)
    self.canvas.draw()

def zoom_in(self):
    zoom(self, 1.1)

def zoom_out(self):
    zoom(self, 1 / 1.1)

def zoom(self, scale_factor):
    cur_xlim = self.ax.get_xlim()
    cur_ylim = self.ax.get_ylim()
    xdata = (cur_xlim[0] + cur_xlim[1]) / 2
    ydata = (cur_ylim[0] + cur_ylim[1]) / 2

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

    self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
    self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
    self.canvas.draw()
