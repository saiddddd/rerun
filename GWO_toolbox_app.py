import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from GWO import GWO

# Title of the application
st.title("GWO Toolbox")

# Sidebar for parameter input
st.sidebar.header("Parameters")

search_agents = st.sidebar.number_input("Search Agents", min_value=1, value=30, step=1)
max_iterations = st.sidebar.number_input("Max Iterations", min_value=1, value=500, step=1)
lower_bound = st.sidebar.number_input("Lower Bound", value=-10.0)
upper_bound = st.sidebar.number_input("Upper Bound", value=10.0)
dimension = st.sidebar.number_input("Dimension", min_value=1, value=30, step=1)

# Text area for custom objective function
st.sidebar.header("Objective Function")
objective_function_code = st.sidebar.text_area(
    "Define your objective function here (use 'position' as the input variable):",
    value="sum(x**2 for x in position)", 
    height=200
)

# Try to compile the custom objective function
try:
    exec(f"def custom_objective_function(position): return {objective_function_code}", globals())
    objective_function_valid = True
except Exception as e:
    st.sidebar.error(f"Error in objective function: {e}")
    objective_function_valid = False

if st.sidebar.button("Run GWO") and objective_function_valid:
    try:
        # Run the GWO algorithm with the custom objective function
        alpha_score, alpha_pos, convergence_curve = GWO(
            search_agents, max_iterations, lower_bound, upper_bound, dimension, custom_objective_function
        )

        # Display the results
        st.write(f"Best Score: {alpha_score}")
        st.write(f"Best Position: {alpha_pos}")

        # Plot the convergence curve
        fig, ax = plt.subplots()
        ax.plot(convergence_curve)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best score obtained so far')
        ax.set_title('Convergence curve')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {str(e)}")
