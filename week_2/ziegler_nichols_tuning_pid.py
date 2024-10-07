import os 
import numpy as np
from numpy.fft import fft, fftfreq
import time
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin


# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext = cur_dir, use_gui=False)  # Initialize simulation interface

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False,0,cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")

# single joint tuning
#episode_duration is specified in seconds
def simulate_with_given_pid_values(sim_, kp, joints_id, regulation_displacement=0.1, episode_duration=10, kd=0, plot=False):
    
    # here we reset the simulator each time we start a new test
    sim_.ResetPose()
    
    # updating the kp value for the joint we want to tune
    kp_vec = np.array([1000]*dyn_model.getNumberofActuatedJoints())
    kp_vec[joints_id] = kp

    kd_vec = np.array([0]*dyn_model.getNumberofActuatedJoints())
    kd_vec[joints_id] = kd
    # IMPORTANT: to ensure that no side effect happens, we need to copy the initial joint angles
    q_des = init_joint_angles.copy()
    qd_des = np.array([0]*dyn_model.getNumberofActuatedJoints())

    q_des[joints_id] = q_des[joints_id] + regulation_displacement 

    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    
    steps = int(episode_duration/time_step)
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_des, qd_des, kp, kd)  # Zero torque command
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        #simulation_time = sim.GetTimeSinceReset()

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_des)
        qd_d_all.append(qd_des)
        #cur_regressor = dyn_model.ComputeDyanmicRegressor(q_mes,qd_mes, qdd_est)
        #regressor_all = np.vstack((regressor_all, cur_regressor))

        #time.sleep(0.01)  # Slow down the loop for better visualization
        # get real time
        current_time += time_step
        #print("current time in seconds",current_time)

    
    # TODO make the plot for the current joint
    # Plotting joint 0 actual vs desired angles if plot is True
    if plot:
        q_mes_all = np.array(q_mes_all)
        q_d_all = np.array(q_d_all)

        plt.figure()
        plt.plot(np.arange(steps) * time_step, q_mes_all[:, joints_id], label="Actual Angle (Joint 0)")
        plt.plot(np.arange(steps) * time_step, q_d_all[:, joints_id], label="Desired Angle (Joint 0)", linestyle='--')
        plt.xlabel("Time [s]")
        plt.ylabel("Joint Angle [rad]")
        plt.title(f"Joint 0: Actual vs Desired Angles (Kp = {kp})")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return q_mes_all

def perform_frequency_analysis(data, dt, plot=False):
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])

    # Find the dominant frequency (ignoring the zero-frequency component)
    dominant_frequency_idx = np.argmax(power[1:]) + 1  # Skip the zero-frequency component
    dominant_frequency = xf[dominant_frequency_idx]

    # Optional: Plot the spectrum
    if plot:
        plt.figure()
        plt.plot(xf, power)
        plt.title("FFT of the signal")
        plt.xlabel("Frequency in Hz")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    return dominant_frequency

def tune_kp_values_visual_analysis(sim_, joint_id, regulation_displacement, init_gain, gain_step, max_gain, test_duration):
    # Initialize list to store results for each gain
    kp_values = []
    q_mes_all_values = []
    q_des_all_values = []

    kp = init_gain
    while kp <= max_gain:
        print(f"Testing Kp = {kp}")

        # Simulate with current Kp value and store the result
        q_mes_all = simulate_with_given_pid_values(sim_, kp, joint_id, regulation_displacement, test_duration, plot=False)
        q_mes_all = np.array(q_mes_all)  # Convert to numpy array

        # Store gain and corresponding actual and desired angles
        kp_values.append(kp)
        q_mes_all_values.append(q_mes_all)
        q_des_all_values.append(init_joint_angles[joint_id] + regulation_displacement)

        # Increase the gain for the next iteration
        kp += gain_step

    # Plot each Kp value in its own subplot
    time_step = sim_.GetTimeStep()
    steps = int(test_duration / time_step)
    time_axis = np.arange(steps) * time_step

    num_plots = len(kp_values)
    cols = 2  # Number of columns in subplot
    rows = (num_plots + 1) // cols  # Calculate rows needed

    fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    fig.suptitle("Joint 0: Actual vs Desired Angles for Different Kp Values", fontsize=16)

    axs = axs.flatten()  # Flatten the axis array to easily iterate

    for i, (ax, kp) in enumerate(zip(axs, kp_values)):
        # Plot the desired trajectory for reference
        ax.plot(time_axis, [init_joint_angles[joint_id] + regulation_displacement] * steps, 'k--', label='Desired Angle', linewidth=2)

        # Plot actual trajectory for the current Kp value
        ax.plot(time_axis, q_mes_all_values[i][:, joint_id], label=f'Actual Angle (Kp={kp})')

        # Label the axes
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Joint Angle [rad]")
        ax.set_title(f"Kp = {kp}")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots if the number of Kp values isn't a multiple of columns
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title properly
    plt.show()


# TODO Implement the table in thi function
def calculate_zn_values(ku, tu, controller_type="P"):

    # Coefficients table based on Ziegler-Nichols rules
    zn_coefficients = {
        "P": [0.5, 0, 0],
        "PI": [0.45, 0.83, 0],
        "PD": [0.8, 0, 0.125],
        "PID": [0.6, 0.5, 0.125]
    }

    if controller_type not in zn_coefficients:
        raise ValueError("Invalid controller type. Choose from 'P', 'PI', 'PD', or 'PID'.")

    # Extract coefficients for the chosen controller type
    kp_coeff, ti_coeff, td_coeff = zn_coefficients[controller_type]

    # Calculate the gains
    kp = kp_coeff * ku
    ki = kp / (ti_coeff * tu) if ti_coeff != 0 else None
    kd = td_coeff * tu * kp if td_coeff != 0 else None

    # Prepare the return dictionary
    gains = {"Kp": kp}
    if ki is not None:
        gains["Ki"] = ki
    if kd is not None:
        gains["Kd"] = kd

    print(f"Calculated {controller_type} Controller Values:")
    for key, value in gains.items():
        print(f"{key} = {value}")

    return gains

if __name__ == '__main__':
    joint_id = 0  # Joint ID to tune
    regulation_displacement = 1.0  # Displacement from the initial joint position
    init_gain=10
    gain_step=1
    max_gain=20
    test_duration=20 # in seconds
    
    # TODO using simulate_with_given_pid_values() and perform_frequency_analysis() write you code to test different Kp values 
    # for each joint, bring the system to oscillation and compute the the PD parameters using the Ziegler-Nichols method
    #tune_kp_values_visual_analysis(sim, joint_id, regulation_displacement, init_gain, gain_step, max_gain, test_duration)   # Graph all the joint angles (measured vs desired)

    useful_Kp = 16  # The Kp value that will be used for the Ziegler-Nichols method
    joint_data = simulate_with_given_pid_values(sim, useful_Kp, joint_id, regulation_displacement, test_duration, plot = False)  # Graph the joint angle for the specified joint at the specified Kp value
    joint_data = np.array(joint_data)[:, joint_id]
    time_step = sim.GetTimeStep()
    # Perform frequency analysis on the joint angle data
    dominant_frequency = perform_frequency_analysis(joint_data, time_step, plot = True)

    # Print the dominant frequency
    print(f"Dominant Frequency for Joint {joint_id}: {dominant_frequency} Hz")

    # Calculate Ku and Tu
    ku = useful_Kp  # In this case, Ku is the same as the current Kp used to generate sustained oscillations
    tu = 1 / dominant_frequency  # Tu is the inverse of the dominant frequency

    # Calculate the values using Ziegler-Nichols method
    controller_type = "PD"  # Specify which controller to calculate values
    controller_values = calculate_zn_values(ku, tu, controller_type)

    # Extract Kp, Ki, Kd values and provide default values for missing components
    kp = controller_values.get("Kp", 0)
    ki = controller_values.get("Ki", 0)
    kd = controller_values.get("Kd", 0)

    # Run the simulation with the calculated PID values
    simulate_with_given_pid_values(sim, kp, joint_id, regulation_displacement, test_duration, kd, plot = True)