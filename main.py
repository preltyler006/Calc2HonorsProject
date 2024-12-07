import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.misc import derivative
import sympy as sp



# Define the symbolic variable and the symbolic function
x = sp.symbols('x')



symbolic_f = x**2  # !!!! Define the function/expression here !!!!
a, b = 0, 2  # Define the range of integration, (a, b)
axis = 'x'  # Choose the axis of rotation ('x' or 'y')



f = sp.lambdify(x, symbolic_f, "numpy") # Used to convert common math functions from sympy library to numpy


'''
DIFFERENT FUNCTIONS:

FOR POWERS:
    ex: f(x) = x^2 

    ==  x**2


FOR CONSTANTS:
    ex: f(x) = 2x

    ==  (x*2), where constant is multiplied after x


FOR ABSOLUTE VALUE:
    ex: f(x) = |x^2 - 4|

    == 1) symbolic_f = sp.Abs(x**2 - 4)
       2) f = sp.lambdify(x, symbolic_f, "numpy")


FOR e^x AND LOG:
    ex: f(x) = e^x^2 - ln(x+1)

    == 1) symbolic_f = sp.exp(x**2) - sp.log(x + 1)
       2) f = sp.lambdify(x, symbolic_f, "numpy")
       --> make sure domains line up, ex x > -1 to avoid complex numbers


FOR NESTED FUNCTIONS AND COMPOSITIONS:
    ex: f(x) = sin(e^x) + cos(ln(x+2))

    == 1) symbolic_f = sp.sin(sp.exp(x)) + sp.cos(sp.log(x + 2))
       2) f = sp.lambdify(x, symbolic_f, "numpy")
       --> make sure domains line up, ex x > -2 to avoid complex numbers

       
FOR RATIONAL FUNCTIONS:
    ex: f(x) = (x^2 + 1) / (x-2)

    == 1) symbolic_f = (x**2 + 1) / (x - 2)
       2) f = sp.lambdify(x, symbolic_f, "numpy")
 


'''

# Automatically convert symbolic function to a lambda function, taking in any number of x values and 
# outputting the corresponding y values based on the function
f = sp.lambdify(x, symbolic_f, "numpy")

def main():


    # Calculate arc length, int a->b sqrt(1+(f'x)^2)dx
    arc_length, _ = quad(lambda x: np.sqrt(1 + derivative(f, x, dx=1e-6)**2), a, b)
    '''
    Quad is a function that takes in the function, and two other parameters for the bounds. Numerically calculates 
    the integral with a bunch of little squares using "adaptive quadrature". Quad automatically changes the 
    size of these intervals where the function changes quickly to guarantee more accurate values
    SAMPLE RATE
    
    lambda x takes in the inner part of the function, the sqrt() and spits out a value to be used for the integration
    
    dx=1e-6 gives accuracy for decimal by incrementation
    '''


    # Calculate surface area based on axis of rotation
    print("help")
    print((derivative(f, x, dx=1e-6)))
    if axis == 'x': # Rotation around x-axis
        surface_area, _ = quad(lambda x: 2 * np.pi * f(x) * np.sqrt(1 + derivative(f, x, dx=1e-6)**2), a, b)
    
    elif axis == 'y': # Rotation around y-axis
        surface_area, _ = quad(lambda x: 2 * np.pi * x * np.sqrt(1 + derivative(f, x, dx=1e-6)**2), a, b)

    else:
        raise ValueError("Axis must be 'x' or 'y'.")
    
    '''
    Similar to the quad function being used above for the arc length, it is used here to get the surface area

    If we have the x axis being rotated, we use f(x) similar to how we use y in manually integrating
    Opposite for y axis being rotated, we use x similar to using x in manual integration

    2 * np.pi * [f(x) or x] == 2*pi*r
    '''

    # Print results in terminal
    print(f"Arc Length: {arc_length}")
    print(f"Surface Area (rotated around {axis}-axis): {surface_area}")



    # ___________________________ 2D Plot of the function ___________________________

    x_vals = np.linspace(a, b, 1000) # List of x values going to be graphed on 2d surface
    y_vals = f(x_vals) # List of y values obtained from putting in all x values into function f(x)

    plt.figure(figsize=(10, 6))
    plt.axis([a, b, f(a), f(b)]) # Sets bounds for 2d space to graph


    # Setting up space for graph and readability for graph
    plt.plot(x_vals, y_vals, label=f"f(x) = {symbolic_f}", color="blue")
    plt.title(f"Function f(x) on a 2D Graph")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()

    # Graph f(x) as arc length
    plt.plot(x_vals, y_vals, color="blue", label="Arc Length")

    # Display results for calculations of arc length
    plt.figtext(0.45, 0.95, f"Arc Length: {arc_length:.3f}")

    plt.grid(True)
    plt.show(block=False)





    # ___________________________ 3D Plot for the rotated surface ___________________________

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    theta = np.linspace(0, 2 * np.pi, 100)
    # theta used for calculations of y and z coordinates in x axis rotation,
    # x and z coordinates in y axis rotation
    x_vals = np.linspace(a, b, 100)
    
    if axis == 'x':
        # Similar to 2D graph, creates 3 parallel arrays of values for x, y and z dimensions based on 
        # x values and trig functions of theta        
        X, Theta = np.meshgrid(x_vals, theta)
        Y = f(X) * np.cos(Theta)
        Z = f(X) * np.sin(Theta)

    elif axis == 'y':
        # Rotation around y-axis
        Y, Theta = np.meshgrid(x_vals, theta)
        X = Y * np.cos(Theta)
        Z = Y * np.sin(Theta)

    else:
        raise ValueError("Axis must be 'x' or 'y'.")

    ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.6, edgecolor='none')
    

    x_axis_length = np.max(X) - np.min(X)  # Length of X-axis
    y_axis_length = np.max(Y) - np.min(Y)  # Length of Y-axis
    z_axis_length = np.max(Z) - np.min(Z) # Length of Z-axis

    # Scale each axis length for visualization
    x_center = (np.max(X) + np.min(X)) / 2
    y_center = (np.max(Y) + np.min(Y)) / 2
    z_center = (np.max(Z) + np.min(Z)) / 2

    # Plot axis lines with appropriate lengths
    ax.plot([x_center - x_axis_length / 2, x_center + x_axis_length / 2], [0, 0], [0, 0], 
            color='red', linewidth=2, label="X-axis")  # X-axis

    ax.plot([0, 0], [y_center - y_axis_length / 2, y_center + y_axis_length / 2], [0, 0], 
            color='green', linewidth=2, label="Y-axis")  # Y-axis

    ax.plot([0, 0], [0, 0], [z_center - z_axis_length / 2, z_center + z_axis_length / 2], 
            color='blue', linewidth=2, label="Z-axis")  # Z-axis


    # Settings for proper viewability
    ax.set_title(f"Surface of Rotation around {axis}-axis")
    ax.set_xlabel("X axis", fontweight="bold")
    ax.set_ylabel("Y axis", fontweight="bold")
    ax.set_zlabel("Z axis", fontweight="bold")
    ax.legend(loc="upper left")
    plt.figtext(0.40, 0.95, f"Surface Area (around {axis}-axis): {surface_area:.3f}")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
