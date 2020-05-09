v_var = 0.01  # translation velocity variance
om_var = 0.1  # rotational velocity variance
r_var = 1  # range measurements variance
b_var = 1  # bearing measurement variance

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

def calc_xk(x_prev,y_prev,th_prev,v_curr,om_curr,delta_T):
    xk = np.zeros([3,1])
    xk[0,0] = delta_T * v_curr*cos((th_prev)) + x_prev
    xk[1,0] = delta_T * v_curr*sin((th_prev)) + y_prev
    xk[2,0] = (delta_T * om_curr) + (th_prev)
    xk[2,0] =  wraptopi(xk[2,0])
    return xk

def calc_yk(x_lm,y_lm,x_curr,y_curr, th_curr):

    yk = np.zeros([2,1])
    yk[0,0] = sqrt((x_lm-x_curr)**2 +(y_lm-y_curr)**2)
    yk[1,0] = (np.arctan2( (y_lm-y_curr),(x_lm-x_curr))) - (np.pi + th_curr)
    yk[1,0] = wraptopi(yk[1,0])
    return yk

def calc_Fk(th_prev,v_curr,delta_T):
    fk = np.eye(3)
    fk[0,2] = -1 * delta_T * v_curr * sin((th_prev))
    fk[1,2] = 1 * delta_T * v_curr * cos((th_prev))
    return fk

def calc_Lk(th_prev,delta_T):
    lk = np.zeros([3,2])
    lk[0,0] = 1 * delta_T * cos((th_prev))
    lk[1,0] = 1 * delta_T * sin((th_prev))
    lk[2,1] = 1 * delta_T
    return lk

def calc_Hk(x_lm,y_lm,x_curr,y_curr):
    hk = np.zeros([2,3])
    hk[0,0] = ((x_lm-x_curr))/(sqrt((x_lm-x_curr)**2 +(y_lm-y_curr)**2))
    hk[0,1] = ((y_lm-y_curr))/(sqrt((x_lm-x_curr)**2 +(y_lm-y_curr)**2))
    hk[0,2] = 0
    hk[1,0] = -1*((y_lm-y_curr)/((x_lm-x_curr)**2 +(y_lm-y_curr)**2))
    hk[1,1] = 1*(x_lm-x_curr)/((x_lm-x_curr)**2 +(y_lm-y_curr)**2)
    hk[1,2] = -1
    return hk

def calc_Mk():
    return np.eye(2)


x_init, y_init, th_init = 50.0, 0, 1.5707963267948966
state_list = np.array([[50.0,0,1.5707963267948966]])
t_prev = 0
P_check = np.diag([1, 1, 0.1]) # initial state covariance
state_meas_list = np.array([[50.0,0,1.5707963267948966]])

for i in range(1,len(t)):
    print("================================STARTING========================", i)

    dt = t[i]-t_prev   # elapsed time since last time step/stamp


    # Pass variables to fxn f(var,..)
    state_curr = calc_xk(x_init,y_init,th_init,v[i],om[i],dt).T



    F_km = calc_Fk(th_init,v[i],dt)

    L_km = calc_Lk(th_init,dt)
#     L_km = np.ones([3,2])
    P_check = (F_km@P_check@F_km.T) + (L_km@Q_km@L_km.T) # initial state covariance

    x_meas = state_curr
#     x_meas = np.array([[49.0,0,1.5707963267948966]])
#     P_check = np.diag([1, 1, 0.1]) # initial state covariance
#     lk = np.array(l[i].tolist)

#     print(lk)
    for k in range(len(b[i])):
        print(x_meas)
        x0, x1 = x_meas[0,0], x_meas[0,1]
        x2 = (x_meas[0,2])
        lk = l[k]
        l_0, l_1 = lk[0], lk[1]

        y_obs = np.array([[r[i,k]],
                          [b[i,k]]])

        y_calc = calc_yk(l_0,l_1,x0,x1, x2)
#         if i>= 130:
#             print(lk)
#             print(y_calc)
#             print(y_obs)
#             input()

        y_diff = y_obs-y_calc
        y_diff[1,0] = wraptopi(y_diff[1,0])
        H_km = calc_Hk(l_0,l_1,x0,x1)
        M_k = calc_Mk()

        val1 = M_k@cov_y@M_k.T
        val2 = H_km@P_check@H_km.T
        val3 = P_check@H_km.T
        K_k = val3@(inv((val2 + val1)))

        P_check = (np.eye(3)-(K_k@H_km))@(P_check)


        x_term = (K_k@(y_diff)).T
        x_meas = x_meas - x_term
        x_meas[0,2] = wraptopi(x_meas[0,2])
#         state_curr = x_meas
        print(x_meas)




    state_curr = x_meas
    state_meas_list = np.append(state_meas_list,x_meas,axis=0)

    # Append data to array for storing values
    state_list = np.append(state_list,state_curr,axis=0)

    # Update variables for next iteration
    x_init, y_init, th_init = state_curr[0,0], state_curr[0,1], (state_curr[0,2])
    t_prev = t[i]

print(state_list)
# #plt.plot(state_list[:,0],state_list[:,1])
# plt.plot(t,state_list[:,2])

# print(state_meas_list)
# # plt.plot(state_meas_list[:,0],state_meas_list[:,1])



# reset values for easy debugging
x_init, y_init, th_init = 50.0, 0, 1.5707963267948966



e_fig = plt.figure()
ax = e_fig.add_subplot(111)
# ax.plot(state_list[:,0],state_list[:,1])
ax.plot(t[:],state_meas_list[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(state_meas_list[:,0],state_meas_list[:,1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()
