import pickle
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]
print("Initial x,y and theta")
print(x_init)
print(y_init)
print(th_init)

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]
#print("\nInput Signals - translational velocity")
#print(v)
# print("\nInput Signals - angular velocity")
# print(om)

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]
print("\nBearing to each Landmark\n",b)
print("\nRange Measurements \n",r)
print("\nx,y positions of landmarks\n",l)
print("\nDistance between robot centre and laser sensor\n",d)



import sympy as sp
from sympy import *
x_l, y_l, x_k, y_k, d_s, x, y, x_k1, y_k1, theta_k, theta_k1,T, v_k, om_k, w_k, Q = sp.symbols(
    'x_l y_l x_k y_k d_s x y x_k-1 y_k-1 theta_k theta_k-1 T v_k om_k w_k Q')
sp.init_printing(use_latex=True)


x_process = sp.Matrix([[x_k1], [y_k1], [theta_k1]]) + T*sp.Matrix([[cos(theta_k1), 0],[sin(theta_k1),0],[0,1]])@(
    sp.Matrix([[v_k], [om_k]]))
#state = sp.Matrix([x,y,theta])
process_state = sp.Matrix([x_k1,y_k1,theta_k1])
x_process


process_noise_state = sp.Matrix([v_k, om_k]) #using v and om since they have similar jac
L_k = x_process.jacobian(process_noise_state)
L_k


F_k = x_process.jacobian(process_state)
F_k


y_p = sp.Matrix([[sqrt((x_l - x_k - d_s*cos(theta_k))**2 + (y_l - y_k - d_s*sin(theta_k))**2)],
                 [(atan2(y_l - y_k - d_s*sin(theta_k), x_l - x_k - d_s*cos(theta_k))) - theta_k]])


# Simplify by sub-ing d since d= 0
y_meas = y_p.subs(d_s,d)
y_meas


# Also, use x to represent x_l -x_k and y for y_l -y_k: For jacobian computation
y_meas_jac = y_meas.subs([(x_k,0),(x_l,x),(y_k,0),(y_l,y)])
y_meas_jac


var_meas = sp.Matrix([x,y,theta_k])
H_k = y_meas_jac.jacobian(var_meas)
H_k


H_k = H_k.subs([(x,(x_l-x_k)),(y,(y_l-y_k))])
H_k


M_k = np.eye(2)
M_k

state_prev = np.array([[x_init,y_init,th_init]])
state_list = np.array([[50.0,0,1.5707963267948966]])

# Define f as a fxn of the variables f(x_k1,... ) = x_process
f = sp.lambdify((x_k1,y_k1,theta_k1,v_k,om_k,T), x_process,"numpy")

# Define g as a fxn of the variables g(x_k1,... ) = y_meas for measurement model
g = sp.lambdify((x_k, x_l,y_k, y_l,theta_k), y_meas,"numpy")

# Define Measurement jacobian in Symbols form
state_time_ind = sp.Matrix([x,y,theta_k])
H_k_sp = y_meas_jac.jacobian(state_time_ind)
H_k_sp = H_k_sp.subs([(x,(x_l-x_k)),(y,(y_l-y_k))])

# Define p as a fxn of the variables p(x_k1,... ) = H_k_sp for measurement Jacobian
p = sp.lambdify((x_k, x_l,y_k, y_l,theta_k), H_k_sp,"numpy")

# Define Process jacobian in Symbols form
process_state = sp.Matrix([x_k1,y_k1,theta_k1])
process_noise_state = sp.Matrix([v_k, om_k]) #using v and om since they have similar jac

F_k_sp = x_process.jacobian(process_state)
print(F_k_sp)
L_k_sp = x_process.jacobian(process_noise_state)
print(F_k_sp)

# Define q as a fxn of the variables q(x_k1,... ) = H_k_sp for process Jacobian
f_f = sp.lambdify((theta_k1, v_k,T), F_k_sp, "numpy")

# Define f_l as a fxn of the variables f_l(x_k1,... ) = H_k_sp for process Jacobian
f_l = sp.lambdify((theta_k1,T), L_k_sp,"numpy")



t_prev = 0
dt = 0

for i in range(len(t)):
    dt = t[i]-t_prev   # elapsed time since last time step/stamp

    # Pass variables to fxn f(var,..)
    state_curr = f(x_init,y_init,th_init,v[i],om[i],dt).T

    # Append data to array for storing values
    state_list = np.append(state_list,state_curr,axis=0)

    # Update variables for next iteration
    x_init, y_init, th_init = state_curr[0,0], state_curr[0,1], state_curr[0,2]
    t_prev = t[i]

print(state_list)
plt.plot(state_list[:,0],state_list[:,1])

# reset values for easy debugging
x_init, y_init, th_init = 50.0, 0, 1.5707963267948966


# Initializing Parameters
v_var = 0.01  # translation velocity variance
om_var = 0.01  # rotational velocity variance
r_var = 0.5  # range measurements variance
b_var = 0.5  # bearing measurement variance

Q_km = np.diag([v_var, om_var]) # input noise covariance
cov_y = np.diag([r_var, b_var])  # measurement noise covariance

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance


# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x


def measurement_update(lk, rk, bk, P_check, x_check):

    # 1. Compute measurement Jacobian
#     var_meas = sp.Matrix([x,y,theta_k])
#     H_k_sp = y_meas_jac.jacobian(var_meas)
#     H_k_sp = H_k_sp.subs([(x,(x_l-x_k)),(y,(y_l-y_k))])
    #print(H_k_sp)

    # Define h as a fxn of the variables h(x_k1,... ) = H_k_sp for measurement Jacobian
    h = sp.lambdify((x_k, x_l,y_k, y_l,theta_k), H_k_sp,"numpy")

    print("\nX_check before:\n",x_check)
    x0, x1, x2 = x_check[0,0], x_check[0,1], x_check[0,2]
    #print(x0, x1, x2)
    l_0, l_1 = lk[0], lk[1]
    #print(l_0, l_1)

    # calculate H_k with predefined function p(var,...)
    #H_k = p(x0, l_0, x1, l_1, x2)
    
    eval_list = H_k_sp.subs([(x_k, x0),(x_l,l_0),(y_k,x1), (y_l,l_1)])
    H_km = np.array(eval_list, dtype=np.float64)
    #print("H_k:\n",H_k)

    M_k = np.eye(2)

    # 2. Compute Kalman Gain
    val1 = (M_k.dot(cov_y).dot(M_k.T))
    val2 = (H_km.dot(P_check).dot(H_km.T))
    val3 = P_check.dot(H_km.T)
    K_k = val3.dot(inv((val2 + val1)))
    #print(val1,val2,val3)



    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    y_obs = np.array([[rk],[bk]])
    print("\nObserved LSR r & b\n",y_obs)

    # Calculate result of measurement model
    # g is already defined as g = sp.lambdify((x_k, x_l,y_k, y_l,theta_k), y_meas,"numpy")
    #y_calc = g(x0,l_0,x1,l_1,x2)

    eval_list = y_meas.subs([(x_k, x0),(x_l,l_0),(y_k,x1), (y_l,l_1), (theta_k,x2)])
    y_calc = np.array(eval_list, dtype=np.float64)

    y_calc[1,0] = wraptopi(y_calc[1,0])
    print("\n Calculated measurement\n",y_calc)

    x_check = x_check + (K_k@(y_obs-y_calc)).T
    x_check[0,2] = wraptopi(x_check[0,2])
    print("\nX_check after:\n",x_check)

    # 4. Correct covariance
    print("Kalman Gain:\n",K_k)
    print("P_check Before correction:\n",P_check)

    P_check = (np.eye(3)-(K_k@H_km))@(P_check)
    print("P_check After correction:\n",P_check)

    return x_check, P_check


# reset values for easy debugging
x_init, y_init, th_init = 50.0, 0, 1.5707963267948966
P_check = np.diag([1, 1, 0.1]) # initial state covariance
#### 5. Main Filter Loop #######################################################################
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)
    print("Iteration number for current step ================================",k)

    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])


    x_check_prev = np.array([[x_init,y_init,th_init]])
    # f is defined above as a fxn of the variables f(x_k1,... ) = x_process
    # f = sp.lambdify((x_k1,y_k1,theta_k1,v_k,om_k,T), x_process,"numpy")

    # Pass variables to fxn f(var,..)
    # x_check = f(x_init,y_init,th_init,v[k],om[k],delta_t).T

    eval_list = x_process.subs([(x_k1,x_init),(y_k1,y_init),(theta_k1, th_init),(v_k,v[k]),
                                (om_k,om[k]),(T,delta_t)])
    x_check = np.array(eval_list, dtype=np.float64).T
    print("x_check from process model:\n", x_check)

    # Wrap theta to [-pi,pi]
    x_check[0,2] = wraptopi(x_check[0,2])
    print(x_check)

    # 2. Motion model jacobian with respect to last state
    F_km = np.zeros([3, 3])
    #print(F_k_sp)

    #F_km = f_f((th_init), (v[k]), delta_t)
    # using declared function f_f is giving wrong computations. sympy.subs is used instead
    eval_list = F_k_sp.subs([(theta_k1, th_init),(v_k,v[k]),(T,delta_t)])
    F_km = np.array(eval_list, dtype=np.float64)
    #print(th_init, v[k], delta_t)
    #print("\nF_k:\n",F_km)


    # 3. Motion model jacobian with respect to noise
    L_km = np.ones([3, 2])
    # L_km = f_l(th_init,delta_t)

    eval_list = L_k_sp.subs([(theta_k1, th_init),(T,delta_t)])
    L_km = np.array(eval_list, dtype=np.float64)
    #print("L_k:\n",L_km)

    # 4. Propagate uncertainty
    # print("P_check before:\n", P_check)
    # P_check = (F_km.dot(P_check).dot(F_km.T)) + (L_km.dot(Q_km).dot(L_km.T)) # initial state covariance

    P_check = (F_km@P_check@F_km.T) + (L_km@Q_km@L_km.T) # initial state covariance
    #print("P_check after:\n", P_check)


    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0,0]
    x_est[k, 1] = x_check[0,1]
    x_est[k, 2] = x_check[0,2]
    P_est[k, :, :] = P_check

    # Update initial states
    x_init, y_init = x_check[0,0], x_check[0,1]
    th_init = wraptopi(x_check[0,2])
