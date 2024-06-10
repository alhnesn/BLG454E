import tkinter as tk
from tkinter import Tk, Frame, Button, Label, Entry, Text, Scrollbar, filedialog, messagebox, Toplevel, Listbox, SINGLE
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from plotting import plot_initial_graph, setup_zoom_feature, zoom_to_fit
from regression import linear_regression, polynomial_regression, remove_linear_regression, remove_polynomial_regression, get_polynomial_degree
from clustering import k_means_clustering, remove_kmeans_clustering, agglomerative_clustering, remove_agglomerative_clustering

class InteractiveTool:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive Regression and Clustering Tool")
        self.master.geometry("1050x650")
        self.master.resizable(False, False)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.data = []
        self.point_labels = []
        self.highlights = {}
        self.highlight_colors = cycle(['red', 'green', 'orange', 'purple'])
        self.lin_reg_line = None
        self.poly_reg_line = None
        self.kmeans = None
        self.kmeans_plot = None
        self.kmeans_centers_plot = None
        self.agglom = None
        self.agglom_plot = None

        self.example_datasets = [
            {"filename": "Project/Implementation/examples/concentric_circles_example.csv", "title": "Concentric Circles Example"},
            {"filename": "Project/Implementation/examples/polynomial_example1.csv", "title": "Polynomial Example 1"},
            {"filename": "Project/Implementation/examples/polynomial_example2.csv", "title": "Polynomial Example 2"},
            {"filename": "Project/Implementation/examples/tilted_blobs_example.csv", "title": "Tilted Blobs Example"},
            {"filename": "Project/Implementation/examples/two_moons_example.csv", "title": "Two Moons Example"}
        ]

        self.create_widgets()
        plot_initial_graph(self)
        setup_zoom_feature(self)

    def create_widgets(self):
        style = {
            "bg": "#f0f0f0",
            "fg": "#333",
            "font": ("Arial", 10),
            "bd": 2,
            "relief": "groove"
        }
        button_style = {
            "bg": "#4caf50",
            "fg": "white",
            "font": ("Arial", 10, "bold"),
            "bd": 2,
            "relief": "raised",
            "cursor": "hand2"
        }
        self.master.configure(bg=style["bg"])

        main_frame = Frame(self.master, bg=style["bg"], width=1050, height=650)
        main_frame.pack(fill="both", expand=False)

        left_frame = Frame(main_frame, bg=style["bg"], width=400, height=650)
        left_frame.pack_propagate(False)
        left_frame.pack(side="left", padx=10, pady=10)

        right_frame = Frame(main_frame, bg=style["bg"], width=650, height=650)
        right_frame.pack_propagate(False)
        right_frame.pack(side="right", padx=10, pady=10)

        add_frame = Frame(left_frame, bg=style["bg"])
        add_frame.pack(fill="x", pady=10)

        self.x_label = Label(add_frame, text="X:", **style)
        self.x_label.pack(side="left")
        self.x_entry = Entry(add_frame, width=10)
        self.x_entry.pack(side="left", padx=5)

        self.y_label = Label(add_frame, text="Y:", **style)
        self.y_label.pack(side="left")
        self.y_entry = Entry(add_frame, width=10)
        self.y_entry.pack(side="left", padx=5)

        self.add_button = Button(add_frame, text="Add Point", command=self.add_point_from_entry, **button_style)
        self.add_button.pack(side="left", padx=5)

        self.import_button = Button(add_frame, text="Import Data", command=self.open_import_window, **button_style)
        self.import_button.pack(side="left", padx=5)

        reg_title_label = Label(left_frame, text="Regression Tools", font=("Arial", 12, "bold"), bg=style["bg"], fg=style["fg"])
        reg_title_label.pack(fill="x", pady=5)

        reg_frame = Frame(left_frame, bg=style["bg"])
        reg_frame.pack(fill="x", pady=10)

        linreg_frame = Frame(reg_frame, bg=style["bg"])
        linreg_frame.pack(fill="x", pady=5)

        self.regression_button = Button(linreg_frame, text="Linear Regression", command=self.linear_regression, **button_style)
        self.regression_button.pack(side="left", fill="x", expand=True, padx=5)

        self.remove_linreg_button = Button(linreg_frame, text="Clear", command=self.remove_linear_regression, width=5, height=1, **button_style)
        self.remove_linreg_button.pack(side="left", padx=5)

        polyreg_frame = Frame(reg_frame, bg=style["bg"])
        polyreg_frame.pack(fill="x", pady=5)

        self.degree_label = Label(polyreg_frame, text="Degree:", **style)
        self.degree_label.pack(side="left")
        self.degree_entry = Entry(polyreg_frame, width=5)
        self.degree_entry.pack(side="left", padx=5)

        self.polynomial_button = Button(polyreg_frame, text="Polynomial Regression", command=self.polynomial_regression, **button_style)
        self.polynomial_button.pack(side="left", fill="x", expand=True, padx=5)

        self.remove_polyreg_button = Button(polyreg_frame, text="Clear", command=self.remove_polynomial_regression, width=5, height=1, **button_style)
        self.remove_polyreg_button.pack(side="left", padx=5)

        equation_frame = Frame(left_frame, bg=style["bg"])
        equation_frame.pack(fill="both", expand=True, pady=5)

        self.scrollbar_eq = Scrollbar(equation_frame)
        self.scrollbar_eq.pack(side="right", fill="y")

        self.equation_text = Text(equation_frame, height=6, yscrollcommand=self.scrollbar_eq.set, wrap="word", bg="#fff", fg=style["fg"], font=style["font"])
        self.equation_text.pack(side="left", fill="both", expand=True)
        self.scrollbar_eq.config(command=self.equation_text.yview)

        cluster_title_label = Label(left_frame, text="Clustering Tools", font=("Arial", 12, "bold"), bg=style["bg"], fg=style["fg"])
        cluster_title_label.pack(fill="x", pady=5)

        cluster_frame = Frame(left_frame, bg=style["bg"])
        cluster_frame.pack(fill="x", pady=10)

        kmeans_frame = Frame(cluster_frame, bg=style["bg"])
        kmeans_frame.pack(fill="x", pady=5)

        self.kmeans_label = Label(kmeans_frame, text="K:", **style)
        self.kmeans_label.pack(side="left")
        self.kmeans_entry = Entry(kmeans_frame, width=5)
        self.kmeans_entry.pack(side="left", padx=5)

        self.clustering_button = Button(kmeans_frame, text="K-Means Clustering", command=self.k_means_clustering, **button_style)
        self.clustering_button.pack(side="left", fill="x", expand=True, padx=5)

        self.remove_kmeans_button = Button(kmeans_frame, text="Clear", command=self.remove_kmeans_clustering, width=5, height=1, **button_style)
        self.remove_kmeans_button.pack(side="left", padx=5)

        agglom_frame = Frame(cluster_frame, bg=style["bg"])
        agglom_frame.pack(fill="x", pady=5)

        self.agglom_label = Label(agglom_frame, text="K:", **style)
        self.agglom_label.pack(side="left")
        self.agglom_entry = Entry(agglom_frame, width=5)
        self.agglom_entry.pack(side="left", padx=5)

        self.agglom_button = Button(agglom_frame, text="Agglomerative Clustering", command=self.agglomerative_clustering, **button_style)
        self.agglom_button.pack(side="left", fill="x", expand=True, padx=5)

        self.remove_agglom_button = Button(agglom_frame, text="Clear", command=self.remove_agglomerative_clustering, width=5, height=1, **button_style)
        self.remove_agglom_button.pack(side="left", padx=5)

        self.error_label = Label(left_frame, text="", fg="red", bg=style["bg"], font=("Arial", 10, "bold"), wraplength=380, justify="left")
        self.error_label.pack(pady=10)

        self.points_frame = Frame(left_frame, bg=style["bg"])
        self.points_frame.pack(fill="both", expand=True, pady=10)

        self.scrollbar = Scrollbar(self.points_frame)
        self.scrollbar.pack(side="right", fill="y")

        self.points_text = Text(self.points_frame, height=10, yscrollcommand=self.scrollbar.set, wrap="word", bg="#fff", fg=style["fg"], font=style["font"])
        self.points_text.pack(side="left", fill="both", expand=True)
        self.scrollbar.config(command=self.points_text.yview)

        self.figure, self.ax = plt.subplots()
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.ax.grid(color='gray', linestyle='--', linewidth=0.5)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, right_frame)
        self.toolbar.update()

        for button in self.toolbar.winfo_children():
            if 'Subplots' in str(button):
                button.pack_forget()

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        
        bottom_frame = Frame(right_frame, bg=style["bg"])
        bottom_frame.pack(fill="x", pady=10)

        self.clear_tools_button = Button(bottom_frame, text="Clear Tools", command=self.clear_regressions_and_clustering, **button_style)
        self.clear_tools_button.pack(side="left", padx=5)

        self.clear_button = Button(bottom_frame, text="Clear All", command=self.clear, **button_style)
        self.clear_button.pack(side="left", padx=5)

        self.help_button = Button(bottom_frame, text="?", command=self.create_help_window, **button_style)
        self.help_button.pack(side="right", padx=5)

    def open_import_window(self):
        self.import_window = Toplevel(self.master)
        self.import_window.title("Import Data")
        self.import_window.geometry("400x450")

        frame = Frame(self.import_window)
        frame.pack(fill="both", expand=True)

        label = Label(frame, text="Choose an option:", font=("Arial", 12))
        label.pack(pady=10)

        example_button = Button(frame, text="Import Example Data", command=self.show_example_datasets)
        example_button.pack(pady=10)

        own_button = Button(frame, text="Import Own Data", command=self.import_csv)
        own_button.pack(pady=10)

        self.example_listbox = Listbox(frame, selectmode=SINGLE, width=50, height=10)
        for dataset in self.example_datasets:
            self.example_listbox.insert(tk.END, dataset["title"])
        self.example_listbox.pack(pady=10)
        self.example_listbox.pack_forget()

        self.import_example_button = Button(frame, text="Import", command=self.import_selected_example_data)
        self.import_example_button.pack(pady=10)
        self.import_example_button.pack_forget()

    def show_example_datasets(self):
        self.example_listbox.pack(pady=10)
        self.import_example_button.pack(pady=10)

    def import_selected_example_data(self):
        selected_index = self.example_listbox.curselection()
        if not selected_index:
            messagebox.showerror("Error", "No dataset selected")
            return

        selected_dataset = self.example_datasets[selected_index[0]]["filename"]
        try:
            data = np.loadtxt(selected_dataset, delimiter=',', skiprows=1)
            if data.shape[1] != 2:
                raise ValueError("CSV file does not have the correct format")

            if len(data) > 300:
                if messagebox.askyesno("Reduce Points", "The CSV file contains more than 300 points. Do you want to reduce the number of points?"):
                    factor = len(data) // 300
                    data = data[::factor]

            self.clear()
            for point in data:
                x, y = point
                self.data.append((x, y))
                self.ax.plot(x, y, 'bo', label='Data Point' if len(self.data) == 1 else "")
            self.update_points_text()
            zoom_to_fit(self)
            self.canvas.draw()
            self.import_window.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def import_csv(self):
        self.import_window.destroy()

        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path.endswith(".csv"):
            messagebox.showerror("Error", "Selected file is not a CSV file")
            return

        try:
            data = np.loadtxt(file_path, delimiter=',', skiprows=1)
            if data.shape[1] != 2:
                raise ValueError("CSV file does not have the correct format")

            if len(data) > 300:
                if messagebox.askyesno("Reduce Points", "The CSV file contains more than 300 points. Do you want to reduce the number of points?"):
                    factor = len(data) // 300
                    data = data[::factor]

            self.clear()
            for point in data:
                x, y = point
                self.data.append((x, y))
                self.ax.plot(x, y, 'bo', label='Data Point' if len(self.data) == 1 else "")
            self.update_points_text()
            zoom_to_fit(self)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_help_window(self):
        help_window = Toplevel(self.master)
        help_window.title("Help")
        help_window.geometry("500x400")

        help_frame = Frame(help_window)
        help_frame.pack(fill="both", expand=True)

        scrollbar = Scrollbar(help_frame)
        scrollbar.pack(side="right", fill="y")

        help_text = Text(help_frame, wrap="word", yscrollcommand=scrollbar.set)
        help_text.pack(fill="both", expand=True)
        scrollbar.config(command=help_text.yview)

        help_content = (
            "How to Use the Program:\n\n"
            "1. Add Points:\n"
            "   - Use the X and Y entry fields to input coordinates.\n"
            "   - Click 'Add Point' to add the point to the graph.\n\n"
            "2. Import Data:\n"
            "   - Click 'Import Data' to load data.\n"
            "   - Choose 'Import Example Data' to select from predefined datasets.\n"
            "   - Choose 'Import Own Data' to load a CSV file with X, Y coordinates.\n"
            "   - The CSV file should have two columns: X and Y.\n\n"
            "3. Regression Tools:\n"
            "   - Linear Regression: Click 'Linear Regression' to fit a linear model.\n"
            "   - Polynomial Regression: Enter the degree and click 'Polynomial Regression'.\n"
            "   - Clear: Click 'Clear' to remove the regression.\n\n"
            "4. Clustering Tools:\n"
            "   - K-Means Clustering: Enter the number of clusters (K) and click 'K-Means Clustering'.\n"
            "   - Agglomerative Clustering: Enter the number of clusters (K) and click 'Agglomerative Clustering'.\n"
            "   - Clear: Click 'Clear' to remove the clustering.\n\n"
            "5. Other Features:\n"
            "   - Clear All: Click to clear all points and regressions/clustering.\n"
            "   - Highlight Points: Click 'Highlight' next to a point to change its color.\n"
            "   - Remove Points: Click 'Remove' next to a point to delete it.\n\n"
            "6. Navigation:\n"
            "   - Drag the graph: Use the pan button on the toolbar.\n"
            "   - Zoom: Use the scroll wheel to zoom in and out, or use the zoom button on the toolbar.\n\n"
            "Note:\n"
            "- Ensure the CSV file is in the correct format (two columns: X and Y).\n"
            "- If the number of points exceeds 300, you will be prompted to reduce the points.\n"
            "- Polynomial regression degree cannot exceed the number of data points."
        )

        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)

    def add_point_from_entry(self):
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            self.data.append((x, y))
            self.ax.plot(x, y, 'bo', label='Data Point' if len(self.data) == 1 else "")
            self.update_points_text()
            self.canvas.draw()
        except ValueError:
            messagebox.showerror("Error", "Invalid input")

    def on_press(self, event):
        if self.toolbar.mode != '':
            return
        if event.button == 1:  # Left click to add point
            self.add_point(event)

    def on_key_press(self, event):
        pass

    def add_point(self, event):
        if event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            self.data.append((x, y))
            self.ax.plot(x, y, 'bo', label='Data Point' if len(self.data) == 1 else "")
            self.update_points_text()
            self.canvas.draw()

    def on_scroll(self, event, scale_factor=None):
        base_scale = 1.1
        if scale_factor is None:
            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                scale_factor = 1

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2
        ydata = event.ydata if event.ydata is not None else (cur_ylim[0] + cur_ylim[1]) / 2

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
        self.canvas.draw()

    def linear_regression(self):
        linear_regression(self)

    def remove_linear_regression(self):
        remove_linear_regression(self)

    def polynomial_regression(self):
        polynomial_regression(self)

    def remove_polynomial_regression(self):
        remove_polynomial_regression(self)

    def k_means_clustering(self):
        k_means_clustering(self)

    def remove_kmeans_clustering(self):
        remove_kmeans_clustering(self)

    def agglomerative_clustering(self):
        agglomerative_clustering(self)

    def remove_agglomerative_clustering(self):
        remove_agglomerative_clustering(self)

    def clear(self):
        self.ax.clear()
        self.data = []
        plot_initial_graph(self)
        self.clear_regressions_and_clustering()
        self.error_label.config(text="")
        self.update_points_text()
        if self.ax.get_legend():
            self.ax.get_legend().remove()
        self.canvas.draw()

    def clear_regressions_and_clustering(self):
        self.remove_linear_regression()
        self.remove_polynomial_regression()
        self.remove_kmeans_clustering()
        self.remove_agglomerative_clustering()
        self.canvas.draw()

    def update_points_text(self):
        self.points_text.config(state=tk.NORMAL)
        self.points_text.delete(1.0, tk.END)

        for i, (x, y) in enumerate(self.data):
            point_text = f"Point {i+1}: ({x:.2f}, {y:.2f})"
            self.points_text.insert("end", point_text)

            edit_button = Button(self.points_frame, text="Edit", command=lambda i=i: self.edit_point(i))
            remove_button = Button(self.points_frame, text="Remove", command=lambda i=i: self.remove_point(i))
            highlight_button = Button(self.points_frame, text="Highlight", command=lambda i=i: self.highlight_point(i))

            self.points_text.window_create("end", window=edit_button)
            self.points_text.insert("end", " ")
            self.points_text.window_create("end", window=remove_button)
            self.points_text.insert("end", " ")
            self.points_text.window_create("end", window=highlight_button)
            self.points_text.insert("end", "\n")

        self.points_text.config(state=tk.DISABLED)

    def edit_point(self, index):
        x, y = self.data[index]
        self.x_entry.delete(0, "end")
        self.x_entry.insert(0, f"{x:.2f}")
        self.y_entry.delete(0, "end")
        self.y_entry.insert(0, f"{y:.2f}")

        self.add_button.config(text="Edit Point", command=lambda: self.update_point(index))

    def update_point(self, index):
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            self.data[index] = (x, y)
            self.ax.clear()
            plot_initial_graph(self)
            for point in self.data:
                self.ax.plot(point[0], point[1], 'bo')
            self.update_points_text()
            self.canvas.draw()
            self.add_button.config(text="Add Point", command=self.add_point_from_entry)
        except ValueError:
            messagebox.showerror("Error", "Invalid input")

    def remove_point(self, index):
        del self.data[index]
        self.ax.clear()
        plot_initial_graph(self)
        for point in self.data:
            self.ax.plot(point[0], point[1], 'bo')
        self.update_points_text()
        self.clear_regressions_and_clustering()
        self.canvas.draw()

    def highlight_point(self, index):
        if index in self.highlights:
            self.highlights[index].remove()
            del self.highlights[index]
            self.update_points_text()
        else:
            x, y = self.data[index]
            color = next(self.highlight_colors)
            highlight, = self.ax.plot(x, y, 'o', color=color, label=f'P{index+1}')
            self.highlights[index] = highlight
            self.update_points_text()
        self.update_legend()
        self.canvas.draw()

    def update_legend(self):
        handles, labels = self.ax.get_legend_handles_labels()
        new_handles_labels = [(h, l) for h, l in zip(handles, labels) if not l.startswith('P')]
        for index in self.highlights:
            new_handles_labels.append((self.highlights[index], f'P{index+1}'))
        if new_handles_labels:
            handles, labels = zip(*new_handles_labels)
            self.ax.legend(handles=handles, labels=labels)
        else:
            if self.ax.get_legend():
                self.ax.get_legend().remove()

    def update_equation_text(self, text):
        self.equation_text.config(state=tk.NORMAL)
        self.equation_text.delete(1.0, "end")
        self.equation_text.insert("end", text)
        self.equation_text.config(state=tk.DISABLED)

    def on_closing(self):
        self.master.quit()
        self.master.destroy()
