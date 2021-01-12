# cut-and-paste command to generate the milestone2.csv file
# python milestone2.py

import sys
import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, pinv

class youBot():
    def __init__(self):
        """youBot initialization"""
        self.r = 0.0475
        self.l = 0.47/2
        self.w = 0.3/2
        self.H0 = self.getH(0)
        self.F = pinv(self.H0, 1e-4)
        self.qinit = np.array([-np.pi/2, 0, 0.5])
        self.Tsb = [[np.cos(self.qinit[0]), -np.sin(self.qinit[0]),  0,  self.qinit[1]],
                    [np.sin(self.qinit[0]), np.cos(self.qinit[0]),   0,  self.qinit[2]],
                    [0,                     0,                       1,  0.0963       ],
                    [0,                     0,                       0,  1            ]]
        self.Tb0 = [[1,  0,  0,  0.1662],
                    [0,  1,  0,  0     ],
                    [0,  0,  1,  0.0026],
                    [0,  0,  0,  1     ]]
        self.Ts0 = np.matmul(self.Tsb, self.Tb0)
        self.M0e = [[1,  0,  0,  0.033 ],
                    [0,  1,  0,  0     ],
                    [0,  0,  1,  0.6546],
                    [0,  0,  0,  1     ]]
        self.Tseinit = np.matmul(self.Ts0, self.M0e)
        self.B1 = [0, 0, 1, 0, 0.033, 0]
        self.B2 = [0, -1, 0, -0.5076, 0, 0]
        self.B3 = [0, -1, 0, -0.3526, 0, 0]
        self.B4 = [0, -1, 0, -0.2176, 0, 0]
        self.B5 = [0, 0, 1, 0, 0, 0]
        self.Blist = np.array([self.B1, self.B2, self.B3, self.B4, self.B5]).T
        self.Tscinit = [[1,  0,  0,  1    ],
                        [0,  1,  0,  0    ],
                        [0,  0,  1,  0.025],
                        [0,  0,  0,  1    ]]
        self.Tscfinal = [[0,  1,  0,  0    ],
                        [-1, 0,  0,  -1   ],
                        [0,  0,  1,  0.025],
                        [0,  0,  0,  1    ]]
        self.Tcestandoff = [[-1, 0,  0, 0  ],
                            [0,  1,  0, 0  ],
                            [0,  0, -1, 0.1],
                            [0,  0,  0, 1  ]]
        self.Tcegrasp = [[-1, 0,  0, 0    ],
                         [0,  1,  0, 0    ],
                         [0,  0, -1, -0.01],
                         [0,  0,  0, 1    ]]
        # Operates at 100Hz
        self.k = 1

    def getH(self, phi):
        """Get the kinetic model matrix H given phi
        :param phi: chassis frame angle
        """
        beta=0; gamma1=-np.pi/4; gamma2=np.pi/4; gamma3=-np.pi/4; gamma4=np.pi/4
        x1=self.l; y1=self.w; x2=self.l; y2=-self.w; x3=-self.l; y3=-self.w; x4=-self.l; y4=self.w
        h1_0 = [(1/(self.r*np.cos(gamma1)))*i for i in [x1*np.sin(beta+gamma1)-y1*np.cos(beta+gamma1), np.cos(beta+gamma1+phi), np.sin(beta+gamma1+phi)]]
        h2_0 = [(1/(self.r*np.cos(gamma2)))*i for i in [x2*np.sin(beta+gamma2)-y2*np.cos(beta+gamma2), np.cos(beta+gamma2+phi), np.sin(beta+gamma2+phi)]]
        h3_0 = [(1/(self.r*np.cos(gamma3)))*i for i in [x3*np.sin(beta+gamma3)-y3*np.cos(beta+gamma3), np.cos(beta+gamma3+phi), np.sin(beta+gamma3+phi)]]
        h4_0 = [(1/(self.r*np.cos(gamma4)))*i for i in [x4*np.sin(beta+gamma4)-y4*np.cos(beta+gamma4), np.cos(beta+gamma4+phi), np.sin(beta+gamma4+phi)]]
        return np.around(np.array([h1_0, h2_0, h3_0, h4_0]), 4)

    def pseudoInv(self, J):
        """ Pesudo-inverse algorithm
        :param J: Matrix to be pesudo-inversed
        """
        m, n = np.shape(J)
        if n > m:
            invJ_JT = np.linalg.inv(np.matmul(J, J.T))
            return np.around(np.matmul(J.T, invJ_JT), 4)
        elif n < m:
            invJT_J = np.linalg.inv(np.matmul(J.T, J))
            return np.around(np.matmul(invJT_J, J.T), 4)
        return np.around(np.linalg.inv(J), 4)

    def NextState(self, currState, controls, dt, maxV):
        """Get next configuration
        :param currState: A 12-vector representing the current configuration of the robot
                    [0:2]: Chassis configuration
                    [3:7]: Arm configuration
                   [8:11]: Wheel angles
        :param controls: A 9-vector of controls indicating the arm joint speeds and the wheel speeds
                  [0:3]: Wheel speeds u
                  [4:8]: Joint speeds thetadot
        :param dt: timestep
        :param maxV: A positive real value indicating the maximum angular speed of the arm joints and the wheels
        :return nextState: A 12-vector representing the configuration of the robot dt time later
        new arm joint angles = (old arm joint angles) + (joint speeds) * dt.
        new wheel angles = (old wheel angles) + (wheel speeds) * dt.
        new chassis configuration is obtained from odometry
        """
        nextState = [0 for _ in range(12)]

        # Update arm joint angles
        for i in range(3,8):
            if controls[i+1] <= maxV:
                nextState[i] = currState[i] + (controls[i+1] * dt)
            else:
                nextState[i] = currState[i] + (maxV * dt)

        # Update wheel angles
        for i in range(8,12):
            if controls[i-8] <= maxV:
                nextState[i] = currState[i] + (controls[i-8] * dt)
            else:
                nextState[i] = currState[i] + (maxV * dt)

        # Update chassis configuration
        dtheta = np.around(np.array([[controls[0]*dt],[controls[1]*dt],[controls[2]*dt],[controls[3]*dt]]), 4)
        Vb = np.around(np.matmul(self.F, dtheta), 4)
        wbz = Vb[0][0]
        vbx = Vb[1][0]
        vby = Vb[2][0]
        dqb = []
        if wbz <= 0.01:
            dqb = np.array([[0], [vbx], [vby]])
        else:
            dqb = np.around(np.array([[wbz], [vbx*np.sin(wbz) + (vby*(np.cos(wbz)-1))/wbz], [vby*np.sin(wbz) + (vbx*(1-np.cos(wbz)))/wbz]]), 4)
        dq = np.around(np.matmul(np.array([[1, 0, 0], [0, np.cos(currState[0]), -np.sin(currState[0])], [0, np.sin(currState[0]), np.cos(currState[0])]]), dqb), 4)
        for i in range(0,3):
            nextState[i] = currState[i] + dq[i][0]

        return nextState

    def TestNextState(self, u, t, dt):
        """Test function for milestone 1
        :param u: wheel speeds
        :param t: duration
        :param dt: time step
        Loop through the duration and find the states.
        Save the resultant states to .csv file. (for scene 6)
        """
        currState_list = [[0 for _ in range(13)]]
        controls = [u[0], u[1], u[2], u[3], 0, 0, 0, 0, 0]
        for i in range(int(t/dt)):
            currState_list.append(self.NextState(currState_list[-1][:-1], controls, dt, 100000) + [0])

        try:
            np.savetxt("testNextState.csv", currState_list, delimiter = ",")
        except:
            print("Unable to save CSV file")
        

    def TrajectoryGenerator(self, Tseinit, Tscinit, Tscfinal, Tcegrasp, Tcestandoff, k, method):
        """Generates the trajectory given the initial conditions
        :param Tseinit: The initial configuration of the end-effector in the reference trajectory
        :param Tscinit: The cube's initial configuration
        :param Tscfinal: The cube's desired final configuration
        :param Tcegrasp: The end-effector's configuration relative to the cube when it is grasping the cube
        :param Tcestandoff: The end-effector's standoff configuration above the cube, before and after grasping, relative to the cube
        :param k: The number of trajectory reference configurations per 0.01 seconds
        :param method: Either "screw" or "cartesian" for corresponding trajectory
        Based on ScrewTrajectory/CartesianTrajectory from the Modern Robotics code library.
        Generate a reference trajectory for the end-effector frame {e} of youBot.
        This trajectory consists of eight concatenated trajectory segments.
        """
        traj = []

        try:
            # Computes standoff and grasp Tse
            Tseinitstandoff = np.matmul(Tscinit, Tcestandoff)
            Tseinitgrasp = np.matmul(Tscinit, Tcegrasp)
            Tsefinalstandoff = np.matmul(Tscfinal, Tcestandoff)
            Tsefinalgrasp = np.matmul(Tscfinal, Tcegrasp)

            # Get optimal durations
            Tf1 = self.getDuration(Tseinit, Tseinitstandoff)
            Tf2 = self.getDuration(Tseinitstandoff, Tseinitgrasp)
            Tf4 = self.getDuration(Tseinitgrasp, Tseinitstandoff)
            Tf5 = self.getDuration(Tseinitstandoff, Tsefinalstandoff)
            Tf6 = self.getDuration(Tsefinalstandoff, Tsefinalgrasp)
            Tf8 = self.getDuration(Tsefinalgrasp, Tsefinalstandoff)

            if method == "screw":
                print("Generating screw trajectory.")
                traj1 = mr.ScrewTrajectory(Tseinit, Tseinitstandoff, Tf1, Tf1*k/0.01, 3)
                traj2 = mr.ScrewTrajectory(Tseinitstandoff, Tseinitgrasp, Tf2, Tf2*k/0.01, 3)
                traj4 = mr.ScrewTrajectory(Tseinitgrasp, Tseinitstandoff, Tf4, Tf4*k/0.01, 3)
                traj5 = mr.ScrewTrajectory(Tseinitstandoff, Tsefinalstandoff, Tf5, Tf5*k/0.01, 3)
                traj6 = mr.ScrewTrajectory(Tsefinalstandoff, Tsefinalgrasp, Tf6, Tf6*k/0.01, 3)
                traj8 = mr.ScrewTrajectory(Tsefinalgrasp, Tsefinalstandoff, Tf8, Tf8*k/0.01, 3)
            elif method == "cartesian":
                print("Generating cartesian trajectory.")
                traj1 = mr.CartesianTrajectory(Tseinit, Tseinitstandoff, Tf1, Tf1*k/0.01, 3)            
                traj2 = mr.CartesianTrajectory(Tseinitstandoff, Tseinitgrasp, Tf2, Tf2*k/0.01, 3)
                traj4 = mr.CartesianTrajectory(Tseinitgrasp, Tseinitstandoff, Tf4, Tf4*k/0.01, 3)
                traj5 = mr.CartesianTrajectory(Tseinitstandoff, Tsefinalstandoff, Tf5, Tf5*k/0.01, 3)
                traj6 = mr.CartesianTrajectory(Tsefinalstandoff, Tsefinalgrasp, Tf6, Tf6*k/0.01, 3)
                traj8 = mr.CartesianTrajectory(Tsefinalgrasp, Tsefinalstandoff, Tf8, Tf8*k/0.01, 3)

            # 1 Move gripper from initial configuration to standoff configuration a few cm above the block
            for X in traj1:
                traj.append(self.getConfiguration(X, "open"))
            # 2 Move gripper down to the grasp position
            for X in traj2:
                traj.append(self.getConfiguration(X, "open"))
            # 3 Closing the gripper
            for i in range(630):
                traj.append(self.getConfiguration(traj2[-1], "closed"))
            # 4 move the gripper back up to the standoff configuration
            for X in traj4:
                traj.append(self.getConfiguration(X, "closed"))
            # 5 move the gripper to a standoff configuration above the final configuration
            for X in traj5:
                traj.append(self.getConfiguration(X, "closed"))
            # 6 move the gripper to the final configuration of the object
            for X in traj6:
                traj.append(self.getConfiguration(X, "closed"))
            # 7 Opening the gripper
            for i in range(630):
                traj.append(self.getConfiguration(traj6[-1], "open"))
            # 8 A trajectory to move the gripper back to the standoff configuration
            for X in traj8:
                traj.append(self.getConfiguration(X, "open"))
        except:
            print("Unplannable input configurations")

        return traj

    def getConfiguration(self, X, gripperState):
        """Translates the transformation matrix into a CoppeliaSim readable csv line
        :param X: The transformation matrix Tse
        :param gripperState: The gripper is "open" or "closed"
        Given Tse = [[r11, r12, r13, px], [r21, r22, r23, py], [r31, r32, r33, pz], [0, 0, 0, 1]].
        Return the csv line [r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper state].
        gripper state: 0 = open, 1 = closed.
        """
        if gripperState == "open":
            return np.array([X[0][0], X[0][1], X[0][2], X[1][0], X[1][1], X[1][2], X[2][0], X[2][1], X[2][2], X[0][3], X[1][3], X[2][3], 0])
        elif gripperState == "closed":
            return np.array([X[0][0], X[0][1], X[0][2], X[1][0], X[1][1], X[1][2], X[2][0], X[2][1], X[2][2], X[0][3], X[1][3], X[2][3], 1])

    def getDuration(self, T1, T2):
        """Computes the optimize duration between the frames
        :param T1: The initial frame
        :param T2: The goal frame
        The optimize duration is based on the maximum distance between the frames.
        """
        return int((np.abs(T1[0][3]-T2[0][3])+np.abs(T1[1][3]-T2[1][3])+np.abs(T1[2][3]-T2[2][3]))/0.05)

    def FeedbackControl(self, X, Xd, Xdnext, Kp, Ki, inteXerr, dt):
        """Calculate the kinematic task-space feedforward plus feedback control law
        :param X: The current actual end-effector configuration X
        :param Xd: The current end-effector reference configuration
        :param Xdnext: The end-effector reference configuration at the next timestep in the reference trajectory
        :param Kp: P gain matrix
        :param Ki: I gain matrix
        :param inteXerr: previous estimate of the integral of the error
        :param dt: timestep
        :return Ve: The commanded end-effector twist expressed in the end-effector frame {e}
        :return inteXerr: current estimate of the integral of the error
        Based on ScrewTrajectory/CartesianTrajectory from the Modern Robotics code library.
        Generate a reference trajectory for the end-effector frame {e} of youBot.
        This trajectory consists of eight concatenated trajectory segments.
        """
        # get Xerr from [Xerr] = log(invX, Xd)
        invX = mr.TransInv(X)
        invX_Xd = np.matmul(invX, Xd)
        Xerr_se3 = mr.MatrixLog6(invX_Xd)
        Xerr = mr.se3ToVec(Xerr_se3)

        # FeedbackControl also needs to maintain an estimate of the integral of the error by adding Xerrdt to a running total at each timestep.
        inteXerr = inteXerr + Xerr * dt

        # The feedforward reference twist Vd that takes Xd to Xdnext in time dt
        # [Vd] = (1/dt)log(invXd,Xdnext)
        invXd = mr.TransInv(Xd)
        invXd_Xdnext = np.matmul(invXd, Xdnext)
        Vd_se3 = (1/dt) * mr.MatrixLog6(invXd_Xdnext)
        Vd = mr.se3ToVec(Vd_se3)

        # Ve = [AdinvX_Xd]Vd + KpXerr + KiinteXerr
        AdinvX_Xd = mr.Adjoint(invX_Xd)
        Ve = np.around(np.matmul(AdinvX_Xd, Vd) + np.matmul(Kp, Xerr) + np.matmul(Ki, inteXerr), 4)
        # Ve = np.around(np.matmul(Kp, Xerr) + np.matmul(Ki, inteXerr), 4)

        return Ve, inteXerr, Xerr

    def RobotConfigToTseJe(self, config):
        """Computes the transformation matrix Tse given the robot configuration vector
        :param config: A 8-vector defines the robot configuration [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5]
        :return Tse: The corresponding transformation matrix Tse
        """
        Tsb = np.array([[np.cos(config[0]), -np.sin(config[0]),  0,  config[1]],
                        [np.sin(config[0]),  np.cos(config[0]),  0,  config[2]],
                        [                0,                  0,  1,     0.0963],
                        [                0,                  0,  0,          1]])
        Tb0 = np.array([[1, 0, 0, 0.1662], [0, 1, 0, 0], [0, 0, 1, 0.0026], [0, 0, 0, 1]])
        M = np.array([[1, 0, 0, 0.033 ], [0, 1, 0, 0], [0, 0, 1, 0.6546], [0, 0, 0, 1]])
        thetalist = np.array([config[3], config[4], config[5], config[6], config[7]])
        T0e = mr.FKinBody(M, self.Blist, thetalist)
        Ts0 = np.matmul(Tsb, Tb0)
        Tse = np.around(np.matmul(Ts0, T0e), 4)

        invTb0 = mr.TransInv(Tb0)
        invT0e = mr.TransInv(T0e)
        AdinvT0e_invTb0 = mr.Adjoint(np.matmul(invT0e, invTb0))
        F6 = np.array([np.zeros(4), np.zeros(4), self.F[0], self.F[1], self.F[2], np.zeros(4)])
        Jbase = np.matmul(AdinvT0e_invTb0, F6)
        Jarm = mr.JacobianBody(self.Blist, thetalist)
        # Je = np.append(Jbase, Jarm, axis=1)
        Je = np.around(np.array([[Jbase[0][0], Jbase[0][1], Jbase[0][2], Jbase[0][3], Jarm[0][0], Jarm[0][1], Jarm[0][2], Jarm[0][3], Jarm[0][4]],
                                 [Jbase[1][0], Jbase[1][1], Jbase[1][2], Jbase[1][3], Jarm[1][0], Jarm[1][1], Jarm[1][2], Jarm[1][3], Jarm[1][4]],
                                 [Jbase[2][0], Jbase[2][1], Jbase[2][2], Jbase[2][3], Jarm[2][0], Jarm[2][1], Jarm[2][2], Jarm[2][3], Jarm[2][4]],
                                 [Jbase[3][0], Jbase[3][1], Jbase[3][2], Jbase[3][3], Jarm[3][0], Jarm[3][1], Jarm[3][2], Jarm[3][3], Jarm[3][4]],
                                 [Jbase[4][0], Jbase[4][1], Jbase[4][2], Jbase[4][3], Jarm[4][0], Jarm[4][1], Jarm[4][2], Jarm[4][3], Jarm[4][4]],
                                 [Jbase[5][0], Jbase[5][1], Jbase[5][2], Jbase[5][3], Jarm[5][0], Jarm[5][1], Jarm[5][2], Jarm[5][3], Jarm[5][4]]]), 4)
        return Tse, Je

    def FullProgram(self, result):
        """The full program for pick and place the cube
        :param result: A string of the result name. 'best', 'overshoot' or 'newTask'
        First calculate the cartesian trajectory.
        Then find the correponding configuration based on different controllers.
        Finally save the results to files.
        """
        if result == "best" or result == "overshoot":
            Tseinit = np.array([[0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [-1, 0, 0, 0.536],
                                [0, 0, 0, 1]])
            initconfig = [0,-0.5,0,np.pi/4,0,0,-np.pi/2,0]
            traj = yb.TrajectoryGenerator(Tseinit, self.Tscinit, self.Tscfinal, self.Tcegrasp, self.Tcestandoff, self.k, method="cartesian")
        elif result == "newTask":
            Tseinit = np.array([[0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [-1, 0, 0, 0.536],
                                [0, 0, 0, 1]])
            Tscfinal = np.array([[1, 0, 0, 2],
                                 [0, 1, 0, 2],
                                 [0, 0, 1, 0.025],
                                 [0, 0, 0, 1]])
            initconfig = [0,-0.5,0,np.pi/4,0,0,-np.pi/2,0]
            traj = yb.TrajectoryGenerator(Tseinit, self.Tscinit, Tscfinal, self.Tcegrasp, self.Tcestandoff, self.k, method="cartesian")

        statelist = [initconfig + [0, 0, 0, 0, 0]]
        currState = initconfig + [0, 0, 0, 0, 0]
        inteXerr = np.array([0, 0, 0, 0, 0, 0])
        Xerrlist = []

        print("Generating animation csv files.")
        for i in range(len(traj)-1):
            X, Je = self.RobotConfigToTseJe(currState)
            Xd = np.array([[traj[i][0], traj[i][1], traj[i][2], traj[i][9]], 
                        [traj[i][3], traj[i][4], traj[i][5], traj[i][10]], 
                        [traj[i][6], traj[i][7], traj[i][8], traj[i][11]], 
                        [0, 0, 0, 1]])
            Xdnext = np.array([[traj[i+1][0], traj[i+1][1], traj[i+1][2], traj[i+1][9]], 
                            [traj[i+1][3], traj[i+1][4], traj[i+1][5], traj[i+1][10]], 
                            [traj[i+1][6], traj[i+1][7], traj[i+1][8], traj[i+1][11]], 
                            [0, 0, 0, 1]])
            Kp = np.zeros((6,6))
            Ki = np.zeros((6,6))
            np.fill_diagonal(Kp, 1)
            if result == "overshoot":
                np.fill_diagonal(Ki, 1)
            dt = 0.01/self.k
            Ve, inteXerr, Xerr = self.FeedbackControl(X, Xd, Xdnext, Kp, Ki, inteXerr, dt)
            pinvJe = pinv(Je, 1e-4)
            controls = np.matmul(pinvJe, Ve)
            currState = self.NextState(currState, controls, dt, 100000000) + [traj[i+1][-1]]
            if not (i+1) % self.k:
                statelist.append(currState)
                Xerrlist.append(Xerr)

        self.saveResults(result, traj, statelist, Xerrlist)

    def saveResults(self, name, traj, statelist, Xerrlist):
        """Save the result to local files
        :param name: name of the result
        :param traj: trajectory to be saved to .csv file (for scene 8)
        :param statelist: configuration to be saved to .csv file (for scene 6)
        :param Xerrlist: Xerr to be saved to .csv file and plotted
        Save the results to corresponding folders as .csv files.
        Plot the Xerr using matplotlib.
        """
        try:
            # np.savetxt(f"/Users/mindyli/Desktop/ME449/Final Project/Jin_Yuming_Capstone/results/{name}/trajectory.csv", traj, delimiter = ",")
            np.savetxt(f"/Users/mindyli/Desktop/ME449/Final Project/Jin_Yuming_Capstone/results/{name}/configuration.csv", statelist, delimiter = ",")
            np.savetxt(f"/Users/mindyli/Desktop/ME449/Final Project/Jin_Yuming_Capstone/results/{name}/xerr.csv", Xerrlist, delimiter = ",")
        except:
            print("Unable to save CSV file")
        
        print("Writing error plot data.")
        x = np.arange(len(Xerrlist))
        plt.plot(x, Xerrlist)
        plt.xlabel("Time")
        plt.savefig(f"/Users/mindyli/Desktop/ME449/Final Project/Jin_Yuming_Capstone/results/{name}/Xerr.png")

        print("Done.")

if __name__ == "__main__":
    yb = youBot()
    
    # # Example runs for Milestone 1 (Uncomment clock block to run)
    # # u = (10,10,10,10). The robot chassis should drive forward in the +xb direction by 0.475 meters.
    # yb.TestNextState([10, 10, 10, 10], 1, 0.01)
    # # u = ( − 10,10, − 10,10). The robot chassis should slide sideways in the +yb direction by 0.475 meters.
    # yb.TestNextState([-10, 10, -10, 10], 1, 0.01)
    # # u = ( − 10,10,10, − 10). The robot chassis should spin counterclockwise in place by 1.234 radians.
    # yb.TestNextState([-10, 10, 10, -10], 1, 0.01)

    # # Example runs for Milestone 2 (Uncomment clock block to run)
    # # Generate a screw trajectory using the default configurations
    # yb.TrajectoryGenerator(yb.Tseinit, yb.Tscinit, yb.Tscfinal, yb.Tcegrasp, yb.Tcestandoff, yb.k, method="screw")

    # # Example runs for Milestone 3 (Uncomment clock block to run)
    # robotConfig = [0, 0, 0, 0, 0, 0.2, -1.6, 0]
    # currState = robotConfig + [0,0,0,0,0]
    # X, Je = yb.RobotConfigToTseJe(robotConfig)
    # Xd = np.array([[0, 0, 1, 0.5], [0, 1, 0, 0], [-1, 0, 0, 0.5], [0, 0, 0, 1]])
    # Xdnext = np.array([[0, 0, 1, 0.6], [0, 1, 0, 0], [-1, 0, 0, 0.3], [0, 0, 0, 1]])
    # Kp = np.zeros((6,6))
    # Ki = np.zeros((6,6))
    # dt = 0.01
    # inteXerr = np.array([0, 0, 0, 0, 0, 0])
    # Ve, inteXerr, Xerr = yb.FeedbackControl(X, Xd, Xdnext, Kp, Ki, inteXerr, dt)
    # pinvJe = pinv(Je, 1e-4)
    # u_td = np.matmul(pinvJe, Ve)
    # controls = np.array([u_td[4], u_td[5], u_td[6], u_td[7], u_td[8], u_td[0], u_td[1], u_td[2], u_td[3]])
    # nextState = yb.NextState(currState, controls, dt, 10000000000)
    # # print infomation and check correctness
    # print("Xd:")
    # print(Xd)
    # print("Xdnext:")
    # print(Xdnext)
    # print("X:")
    # print(X)
    # print("Ve:")
    # print(Ve)
    # print("Xerr:")
    # print(Xerr)
    # print("Je:")
    # print(np.around(Je, 3))
    # print("Controls:")
    # print(np.around(controls, 3))
    # print("nextState:")
    # print(nextState)

    # Full program run
    yb.FullProgram(str(sys.argv[1]))
