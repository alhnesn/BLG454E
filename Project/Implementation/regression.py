import numpy as np

def linear_regression(self):
    self.error_label.config(text="")
    if len(self.data) < 2:
        self.error_label.config(text="Error: Not enough data points")
        return

    try:
        x = np.array([p[0] for p in self.data])
        y = np.array([p[1] for p in self.data])
        
        # Create the design matrix for linear regression
        X = np.vstack([x, np.ones(len(x))]).T
        
        # Solve for coefficients using the normal equation
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        b1, b0 = theta[0], theta[1]

        # Predict values
        min_x, max_x = min(x), max(x)
        margin_x = 0.15 * (max_x - min_x)
        cur_xlim = self.ax.get_xlim()
        x_fit = np.linspace(min(min_x - margin_x, cur_xlim[0]), max(max_x + margin_x, cur_xlim[1]), 100)
        y_fit = b0 + b1 * x_fit
        
        # Calculate Mean Squared Error
        y_pred = b0 + b1 * x
        mse = np.mean((y - y_pred) ** 2)

        if self.lin_reg_line:
            self.lin_reg_line.remove()
        self.lin_reg_line, = self.ax.plot(x_fit, y_fit, color='red', linewidth=2, label='Linear Regression')
        equation = f"Linear Regression\n\nEquation: y = {b1:.2f}x + {b0:.2f}\n\nMSE: {mse:.6e}"
        self.update_equation_text(equation)
        self.ax.legend()
        self.canvas.draw()
    except Exception as e:
        self.update_equation_text(f"Error: {str(e)}")

def remove_linear_regression(self):
    if self.lin_reg_line:
        self.lin_reg_line.remove()
        self.lin_reg_line = None
        self.update_equation_text("")
        handles, labels = self.ax.get_legend_handles_labels()
        new_handles_labels = [(h, l) for h, l in zip(handles, labels) if l != 'Linear Regression']
        if new_handles_labels:
            handles, labels = zip(*new_handles_labels)
            self.ax.legend(handles=handles, labels=labels)
        else:
            self.ax.legend().remove()
        self.canvas.draw()

def polynomial_regression(self):
    self.error_label.config(text="")
    if len(self.data) < 2:
        self.error_label.config(text="Error: Not enough data points")
        return

    try:
        degree = get_polynomial_degree(self)
        if degree > len(self.data):
            self.error_label.config(text="Error: Degree greater than number of data points")
            return
        x = np.array([p[0] for p in self.data])
        y = np.array([p[1] for p in self.data])
        
        # Create the design matrix for polynomial regression
        X_poly = np.vander(x, degree + 1)

        # Solve for coefficients using the normal equation
        coef = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
        
        # Predict values
        min_x, max_x = min(x), max(x)
        margin_x = 0.15 * (max_x - min_x)
        cur_xlim = self.ax.get_xlim()
        x_fit = np.linspace(min(min_x - margin_x, cur_xlim[0]), max(max_x + margin_x, cur_xlim[1]), 100)
        X_fit_poly = np.vander(x_fit, degree + 1)
        y_fit = X_fit_poly @ coef
        
        # Calculate Mean Squared Error
        y_pred = X_poly @ coef
        mse = np.mean((y - y_pred) ** 2)

        if self.poly_reg_line:
            self.poly_reg_line.remove()
        self.poly_reg_line, = self.ax.plot(x_fit, y_fit, color='green', linewidth=2, label='Polynomial Regression')
        equation = "Polynomial Regression\n\nEquation: y = " + " + ".join([f"{coef[i]:.2f}x^{degree - i}" for i in range(degree) if coef[i] != 0]) + f" + {coef[-1]:.2f}\n\nMSE: {mse:.6e}"
        self.update_equation_text(equation)
        self.ax.legend()
        self.canvas.draw()
    except Exception as e:
        self.update_equation_text(f"Error: {str(e)}")


def remove_polynomial_regression(self):
    if self.poly_reg_line:
        self.poly_reg_line.remove()
        self.poly_reg_line = None
        self.update_equation_text("")
        handles, labels = self.ax.get_legend_handles_labels()
        new_handles_labels = [(h, l) for h, l in zip(handles, labels) if l != 'Polynomial Regression']
        if new_handles_labels:
            handles, labels = zip(*new_handles_labels)
            self.ax.legend(handles=handles, labels=labels)
        else:
            self.ax.legend().remove()
        self.canvas.draw()

def get_polynomial_degree(self):
    try:
        degree = int(self.degree_entry.get())
    except ValueError:
        degree = None

    if degree is None or degree <= 0:
        degree = find_optimal_degree(self)

    return degree

def find_optimal_degree(self):
    x = np.array([p[0] for p in self.data])
    y = np.array([p[1] for p in self.data])
    max_degree = min(len(self.data) - 1, 10)
    bic_scores = []
    for d in range(1, max_degree + 1):
        X_poly = np.vander(x, d + 1)
        coef = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
        y_poly_pred = X_poly @ coef
        mse = np.mean((y - y_poly_pred) ** 2)
        n = len(y)
        p = len(coef)
        bic_score = n * np.log(mse) + p * np.log(n)
        bic_scores.append(bic_score)
    optimal_degree = bic_scores.index(min(bic_scores)) + 1
    return optimal_degree
