import math, os, time

# Used for mathematical operations
import numpy as np
import numpy.matlib
import scipy.spatial as scip
import scipy.stats as stats

# Pathfinding module to find path between current position and desired position
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

# Import matplotlib to enable plotting of the GP model
import matplotlib.pyplot as plt

# Defining path to save files to the correct path
my_path = os.path.dirname(os.path.abspath('simulator.py'))



class Simulation():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = []
        self.salinity = []
        #self.first_lat = 63.906453 * (math.pi /180)
        #self.first_lon = 8.503611 * (math.pi /180)

        #self.lat = self.first_lat
        #self.lon = self.first_lon

        # Setup a mesh grid and values
        Lx = 10000  # [m] Length in x-direction (north)
        Ly = 10000  # [m] Length in y-direction (east)
        Nx = 50
        Ny = 50
        self.Nx, self.Ny = Nx, Ny
        dX, dY = Lx / Nx, Ly / Ny  # Grid partitioning
        self.xx, self.yy = np.meshgrid(np.arange(0, Lx, dX), np.arange(0, Ly, dY))
        
        self.N = Nx*Ny # Number of possible sampling locations
        
        
        # Initialize measured locations
        self.measured = np.empty((Nx,Ny))
        self.measured[:] = np.nan
        self.measured_last = np.empty((Nx,Ny))
        self.measured_last[:] = np.nan
        
        self.wp = []
        self.wp_last = []

        self.been = np.zeros((Nx, Ny))
        self.temp_checked = np.zeros((Nx, Ny))
        self.been_last = np.zeros((50,50))
        
        self.time_lapsed = time.time()
        self.iterations = 0
        
        self.GP_AUV = Gaussian_process()
        
        self.counter = 1
        
    def temp_to_z(self,real_field):
        
        # Increase iteration
        self.iterations += 1
        self.been_last[self.wp_last[0]][self.wp_last[1]] = 0
        
        # Analyse temp
        n_temp = len(self.temp)
        n_last = int(n_temp/2)
        if n_temp/2 is float:
            n_next = n_temp - n_last - 1
        else:
            n_next = n_temp - n_last
       
        # Calculate average temperature to be stored
        real_field = real_field.T
        avg_temp_last = real_field[self.wp_last[0],self.wp_last[1]]
        avg_temp_next = real_field[self.wp[0],self.wp[1]]

        # Save temperature to last wp and current wp and update text files.
        if np.isnan(self.measured[self.wp_last[0], self.wp_last[1]]):
            self.measured[self.wp_last[0], self.wp_last[1]] = avg_temp_last
        else:
            self.measured[self.wp_last[0], self.wp_last[1]] = (avg_temp_last + self.measured[self.wp_last[0],
                                                                                             self.wp_last[1]])/2
        self.measured[self.wp[0], self.wp[1]] = avg_temp_next
        # Update the current position in the grid
        self.been[self.wp[0], self.wp[1]] = self.iterations + 1
        self.been[self.wp_last[0], self.wp_last[1]] = self.iterations 
        # Keeping current position
        self.been_last[self.wp[0]][self.wp[1]] = self.iterations + 1
        
        self.temp = []
        
        return self.been, self.measured

        
    def next_waypoint(self, mode, real_field):
        # TODO: Implement different algorithms
        """
        Finds the new waypoint from temperature prediction.
        """
        # Update GP
        _, Sigma_finished ,self.pred_field, self.std_dev, Sigma_0 = self.GP_AUV.gp("AUV", self.been, self.measured,self.iterations)
       
        
        # State variables
        mypos = np.array(self.wp)
        z = np.array(self.pred_field) # - 272.15 # Change between deg C and deg K
        been = np.array(self.been) # Create array of visited locations
        uncertainty = np.array(self.std_dev.T) # Getting uncertainty from the GP model
        
        
        
        # Set up of adjecent nodes
        neighbours = [(-1,-1), (-1,1),(1,1),(1,-1),(-1,0),(0,1),(0,-1),(1,0)]

        def uncertainty_algorithm():
            
            visited = np.array(self.been_last)
            measured = np.array(self.measured)
            obj_list = -100
            pos_list = []
            
            # Evaluating adjecent nodes
            for i,j in neighbours:
                
                if ((self.wp[0] + i) < self.Nx and (self.wp[1] + j) < self.Ny) and ((self.wp[0] + i) >= 0 and (self.wp[1] + j) >= 0):
                    pass
                else:
                    # Skip if the node out of bounds
                    continue
                
                # visited has been used over self.been for simulations
                if self.been[self.wp[0] + i][self.wp[1] + j] != 0:
                    # Node has been visited by the AUV
                    continue
                   
                # Set neighbouring node to visited
                visited[self.wp[0] + i][self.wp[1] + j] = self.iterations + 2
                measured[self.wp[0] + i][self.wp[1] + j] = self.pred_field[self.wp[0] + i][self.wp[1] + j]
                
                # Reevaluate gaussian process
                C, sigma_post, _, _,_ = self.GP_AUV.gp("test",visited,measured, self.iterations)
                
                # Resetting temp "visited" nodes
                visited[self.wp[0] + i][self.wp[1] + j] = 0
                measured[self.wp[0] + i][self.wp[1] + j] = np.nan
                
                objective_func = (1/self.N)*(np.trace(C) - np.trace(sigma_post))
                #print("i,j er: ", i,j)
                #print("Objektfunksjon: ", objective_func)
                
                if (objective_func > obj_list):
                    # New location is "better" than previous
                    obj_list = objective_func
                    #print("Velger i,j: ", i,j)
                    # Clear list of possible locations and add the current evaluation
                    pos_list.clear()
                    pos_list.append(self.wp[0] + i) 
                    pos_list.append(self.wp[1] + j)
                    #print(pos_list)
                elif objective_func == obj_list:
                    # The new location is equal to some prev
                    pos_list.append(self.wp[0] + i) 
                    pos_list.append(self.wp[1] + j)
                else:
                    # New location is "worse", try next location
                    continue
                
            # Print error if no path could be found from current location
            if len(pos_list) == 0:
                print("No path could be found")
                return 0
            
            # Setting goal node
            end_point = [pos_list[0],pos_list[1]]
            #print(end_point)
            return end_point
        
        def uncertainty_with_magnitude():
            visited = np.array(self.been_last)
            measured = np.array(self.measured)
            pred_field = np.array(self.pred_field)
            if self.iterations > 0:
                # Get min temperature to normalize the magnitude term
                min_temp = np.nanmin(measured)
                #print(min_temp)
            obj_list = -100
            pos_list = []
            
            # Evaluating adjecent nodes
            for i,j in neighbours:
                #print(obj_list)
                if ((self.wp[0] + i) < self.Nx and (self.wp[1] + j) < self.Ny) and ((self.wp[0] + i) >= 0 and (self.wp[1] + j) >= 0):
                    pass
                else:
                    # Skip if the node out of bounds
                    continue
                
                # visited has been used over self.been for simulations
                if self.been[self.wp[0] + i][self.wp[1] + j] != 0:
                    # Node has been visited by the AUV
                    continue
           
                visited[self.wp[0] + i][self.wp[1] + j] = self.iterations + 2
                measured[self.wp[0] + i][self.wp[1] + j] = self.pred_field[self.wp[0] + i][self.wp[1] + j]
                
                # Reevaluate gaussian process
                C, sigma_post, _, _, _ = self.GP_AUV.gp("test",visited,measured, self.iterations)
                # Getting estimated temperature of evaluated position
                magnitude = self.pred_field[self.wp[0] + i][self.wp[1]+j]
                
                # Resetting temp "visited" nodes
                visited[self.wp[0] + i][self.wp[1] + j] = 0
                measured[self.wp[0] + i][self.wp[1] + j] = np.nan
                
                evaluate = (np.trace(C) - np.trace(sigma_post))
                
                
                # Weighting between the factors
                theta_1 = 0.8 
                theta_2 = 1 - theta_1
                
                # Normalizing temperature before the objective funciton
                if self.iterations > 0:
                    magnitude /= min_temp

                objective_func = 1/self.N * (theta_1* evaluate + theta_2 * magnitude)

                
                if (objective_func > obj_list):
                    # New location is "better" than previous
                    obj_list = objective_func
                    # Clear list of possible locations and add the current evaluation
                    pos_list.clear()
                    pos_list.append(self.wp[0] + i) 
                    pos_list.append(self.wp[1] + j)

                elif objective_func == obj_list:
                    # The new location is equal to some previous 
                    pos_list.append(self.wp[0] + i) 
                    pos_list.append(self.wp[1] + j)
                    
                else:
                    # New location is "worse", try next location
                    continue
            
            # Print error if no path could be found from current location
            if len(pos_list) == 0:
                print("No path could be found")
                return 0
            
            # Setting goal node
            end_point = [pos_list[0],pos_list[1]]
            
            return end_point
        
        def entropy_algorithm():
            # TODO: Rewrite to something comparable to uncertainty
            entropy = np.array(z[:]*math.log10(z[:]))
            max_entropy_pos = np.unravel_index(np.argmax(entropy, axis=None), entropy.shape)
            end_point = max_entropy_pos
            return end_point
        
        def mutual_information():
            visited = np.array(self.been_last)
            measured = np.array(self.measured)
            pred_field = np.array(self.pred_field.T)
            obj_list = -100
            pos_list = []
            
            # Evaluating adjecent nodes
            for i,j in neighbours:
                
                if ((self.wp[0] + i) < self.Nx and (self.wp[1] + j) < self.Ny) and ((self.wp[0] + i) >= 0 and (self.wp[1] + j) >= 0):
                    pass
                else:
                    # Skip if the node out of bounds
                    continue
                
                if self.been[self.wp[0] + i][self.wp[1] + j] != 0:
                    # Node has been visited by the AUV
                    continue
                   
                # Set neighbouring node to visited
                visited[self.wp[0] + i][self.wp[1] + j] = self.iterations + 2
                measured[self.wp[0] + i][self.wp[1] + j] = pred_field[self.wp[0] + i][self.wp[1] + j]
                
                # Reevaluate gaussian process
                C, sigma_post, _, _, _ = self.GP_AUV.gp("test",visited,measured, self.iterations)
                
                # Resetting temp "visited" nodes
                visited[self.wp[0] + i][self.wp[1] + j] = 0
                measured[self.wp[0] + i][self.wp[1] + j] = np.nan
                
                #objective_func = (1/2)*(math.log10(2*math.pi*math.e)**self.N)*(
                #    np.linalg.det(C) - np.linalg.det(sigma_post))

                objective_func = (np.linalg.det(C) - np.linalg.det(sigma_post))
                if (objective_func > obj_list):
                    # New location is "better" than previous
                    obj_list = objective_func
                    # Clear list of possible locations and add the current evaluation
                    pos_list.clear()
                    pos_list.append(self.wp[0] + i) 
                    pos_list.append(self.wp[1] + j)
                    #print(pos_list)
                elif objective_func == obj_list:
                    # The new location is equal to some prev
                    pos_list.append(self.wp[0] + i) 
                    pos_list.append(self.wp[1] + j)
                else:
                    # New location is "worse", try next location
                    continue
                
            # Print error if no path could be found from current location
            if len(pos_list) == 0:
                print("No path could be found")
                return 0
            
            # Setting goal node
            end_point = [pos_list[0],pos_list[1]]
            #print(end_point)
            return end_point
          
        def lawn_mower():
            # Defnining shape of lawn mower pattern
            width = 50
            length = 16
            
            if self.iterations < width:
                end_point = [self.wp[0] + 1,self.wp[1]]
            elif self.iterations < (width+length):
                end_point = [self.wp[0], self.wp[1] +1]
            elif self.iterations < (2*width + length - 1):
                end_point = [self.wp[0] - 1, self.wp[1]]
            elif self.iterations < (2* width + 2*length - 1):
                end_point = [self.wp[0], self.wp[1] + 1]
            elif self.iterations < (3*width + 2* length - 2):
                end_point = [self.wp[0] + 1, self.wp[1]]
            elif self.iterations < (3*width + 3*length -2):
                end_point = [self.wp[0], self.wp[1] + 1]
            elif self.iterations < (4*width + 3*length - 3):
                end_point = [self.wp[0] - 1, self.wp[1]]
            elif self.iterations < (4*width + 4*length -3):
                end_point = [self.wp[0], self.wp[1] + 1]
            elif self.iterations < (5*width + 4*length - 4):
                end_point = [self.wp[0] + 1, self.wp[1]]
            elif self.iterations < (5*width + 5*length -4):
                end_point = [self.wp[0], self.wp[1] + 1]
            elif self.iterations < (6*width + 5*length - 5):
                end_point = [self.wp[0] - 1, self.wp[1]]
            else: 
                end_point = self.wp

            return end_point
        
        # Selection of algorithm
        if mode == 1:
            end_point = uncertainty_algorithm()
        elif mode == 2:
            end_point = entropy_algorithm()
        elif mode == 3:
            end_point = mutual_information()
        elif mode == 4:
            end_point = uncertainty_with_magnitude()
        elif mode == 5:
            end_point = lawn_mower()
        else:
            print('Error: Mode should be either 1 to 5')
            print('Defaulting to path planning based on uncertainty (mode 1)')
            end_point = uncertainty_algorithm()
        
        if len(end_point) == 0:
            return

        self.wp_last = np.array(self.wp)
        self.wp = np.array(end_point)

        
        return self.std_dev, self.pred_field, Sigma_0,Sigma_finished
    
    def AUV_movement(self):
        # Setting next point in grid
        next_coord = [self.xx[self.wp[0],self.wp[1]], self.yy[self.wp[0],self.wp[1]]]
        # Update position
        self.position = next_coord
        


class Gaussian_process():
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.iterations = 0

        # Setup a mesh grid and values
        Lx = 10000  # [m] Length in x-direction (north)
        Ly = 10000  # [m] Length in y-direction (east)
        Nx = 50
        Ny = 50
        self.Nx , self.Ny= Nx, Ny
        dX, dY = Lx / Nx, Ly / Ny  # Grid partitioning
        self.xx, self.yy = np.meshgrid(np.arange(0, Lx, dX), np.arange(0, Ly, dY))
        
        # Initialize temperature prediction and true temperature field
        self.mu_field = np.ones((50,50)) + 4 # Init estimate of 7 deg in all points
        self.pred_field = np.ones((50,50)) + 4
        self.true_temp = np.array((50,50))

        self.time_elapsed = time.time()
        
        self.C = np.array((2500,2500))
        self.init_C = np.array((2500,2500))
        
    # Field of white noise to generate a random temperature field
    def white_noise(self):
        mean = 0 
        std = 1 
        # Creating enough samples for a 50 by 50 grid
        num_samples = 2500
        samples = np.random.normal(mean,std, size= num_samples)
        # Restructuring into a 50 by 50 grid
        restruct_samples = np.reshape(samples,[50,50])
        
        # Saving noise to create synthetic ocean model
        self.mu_field = restruct_samples

    def get_oceandata(self, field_number):
        
        true_field = np.loadtxt(str(my_path + '/Textfiles/' + 'truetemp_%i.txt' %field_number))   
        return true_field
    
    # From Trygve Olav Fossum: Basic_GP, for more info see Licence.
    def gp(self,name,visited,measured, iteration):
            
        def getcov(d_sites, p_sites, par):
            sig2 = par[0]
            crange = par[1]
            h = scip.distance.cdist(d_sites, p_sites, 'sqeuclidean')
            
            test_start_time = time.time()
            for i in np.nditer(h, op_flags=['readwrite']):  # Modifying array values
                i[...] = sig2 * math.exp(-3 * (1.0 / crange) * math.sqrt(i))  # Possible to use math, slightly faster

            #print("time getcov [seconds]:", (time.time() - test_start_time))
            return np.transpose(h)
        
        # Covariance calculation parameter [sigma, delta] design parameters (hyperparameters)
        # [design parameter for variance, correlation distance] this can be changed
        cov_param = [0.04, 70] # Initially set to [0.04,70]
        # Get grid information
        xx = len(self.xx)
        yy = len(self.yy)
        n = xx*yy

        # Grid constructor variables east
        nxs = np.arange(0, xx, 1)
        east = np.matlib.repmat(nxs, yy, 1)
        east = east.ravel()

        # Grid constructor variables north
        nxs = np.arange(0, yy, 1)
        north = np.repeat(nxs, xx, axis=0)
        north = north.ravel()
        
        # Prediction sites
        pred_sites = np.c_[east, north]
        
        # Data sites based on where AUVs have been
        data_sites = np.c_[np.nonzero(visited)]
                
        # Covariance matrices
        if iteration < 2:
            # Generate covariance matrix
            C = getcov(pred_sites, pred_sites, cov_param)
            self.init_C = C
            self.C = C
        else:
            # Avoid creating the covariance matrix for each iter
            C = self.C


        C_extra = getcov(data_sites, pred_sites, cov_param)
        C0 = getcov(data_sites, data_sites, cov_param)
        tau = 0.005
        C0 +=  tau ** 2 * np.eye(np.shape(C0)[0], np.shape(C0)[1])
        
        # Flattens predicted temperature and true temperature.
        pred_temp = np.array(self.mu_field.flatten())
        true_temp = np.array(measured.flatten())
        
            

         
        # Set up to create a synthetic temperature field
        if (name == "real"):
            # Regression generation of prior mean, mu
            x_gp = np.c_[north, east, np.ones(n)]
            b = np.linalg.lstsq(x_gp, pred_temp, rcond=-1)[0]
            mu = b[2] + b[0] * north + b[1] * east
            mu = mu.reshape(yy, xx)
            
            # Calculating the Cholesky decomposition
            Cholesky_decomp = np.linalg.cholesky(C0)
            
            gaussian_field = mu.flatten().T + np.matmul(Cholesky_decomp,pred_temp.T)
            gaussian_field = np.reshape(gaussian_field,(xx,yy))
            gaussian_field += 9
            
            # Return the "actual temperature field
            return gaussian_field
  
        # Prior estimate
        #d_n = np.array(list(zip(*data_sites))[0])
        #d_e = np.array(list(zip(*data_sites))[1])
        #mu0_pred = b[2] + b[0] * d_n + b[1] * d_e
        
        
        
        # Assimilate observations
        ind = []

        for north, east in data_sites:
            ind.append(north * xx + east)

        # Creating observation matrix            
        F = np.zeros((np.shape(C0)[0],n))
        counter = 0
        
        # Fill visited locations into the observation matrix
        
        for i in range(2500):
            if visited.flatten()[i] == 0:
                     continue
            elif counter == np.shape(C0)[0]:
                # Prevent index of going out of range
                break
            elif (visited.flatten()[i] == np.amax(visited)) and (name == "test"):
                F[counter][i] = 1
                counter += 1
            elif ((visited.flatten()[i] != 0) and (name == "AUV")):
                F[counter][i] = 1
                counter += 1

        # Measurement noise
        T = tau ** 2 * np.eye(np.shape(F)[0], np.shape(F)[0])
        mu0 = self.mu_field
        mu0 = mu0.flatten()[ind]
        measured = true_temp[ind]
        # @ for matrix multiplication in python 3.5+


        # Prediction
        posterior_core = (C @ F.T) @ np.linalg.inv(F @ C @ F.T + T)
        difference = np.reshape(measured,((len(measured),1))) - F @ np.reshape(self.mu_field, (2500,1))

        diff_mu = posterior_core @ difference
        #self.mu_field += np.reshape(diff_mu,(50,50))
        #print(np.shape(measured))
        #diff = measured - mu0_pred
        similarity = np.array(np.linalg.lstsq(C0, difference.flatten(), rcond=-1)[0])[:, None]
        resulting_mu = C_extra.dot(similarity)
        pred_field = self.mu_field + resulting_mu.reshape(yy, xx)
        #self.mu_field += resulting_mu.reshape(yy, xx)
        resulting_cov = np.diag(
            (cov_param[0] * np.ones((n, 1)) - np.diag(np.dot(C_extra, np.linalg.lstsq(C0, C_extra.T, rcond=-1)[0])).flatten()))
        resulting_cov = resulting_cov.reshape(yy, xx)
        cov_squared = np.sqrt(np.sqrt(resulting_cov**2))
        
        # Calculate the inverted matrix once

        
        sigma_post = C - (posterior_core @ (F @ C))
        self.C = sigma_post

        #self.pred_field = pred_field
       # self.std_dev = cov_squared
        
        # Returns prior and posterior covariance
        # + predicted field and the uncertainty of the field
        return C, sigma_post, pred_field, resulting_cov, self.init_C
    

     
        
# Unsure if this function is needed
def check_txtfile(name):
    # Checks if the given file exsists, and delete it if needed
    if os.path.exists(str(my_path +'/Textfiles/'+ name)):
        f = open(str(my_path + '/Textfiles/' +  name), "a")
        print("Appending results to: ", name)
    # If the file does not exsist, create it
    else:
        f = open(str(my_path + '/Textfiles/' + name), "a")
        print("Created: ", name)


if __name__ == '__main__':
    # Setting initial variables
    iterations = 0 # Starting interations
    #initial = 1 # Creating initial condition using this variable
    number_of_iterations = 150 # Number of grid cells visited by the AUV
    number_of_runs = 1
    mode = 5 # 1 for uncertainty,2 for entropy, 3 for MI, 4 for unc_with_magnitude and 5 for lawn mower
    R = 0 # Initalize performace metric
    R_high = 0
    R_low = 100
    RMSE_tot = 0
    RMSE_high = 0
    RMSE_low = 1
    fig_num = 1
    Simulation_time = time.time() # Starting simulation timer
    """
    check_txtfile("R2_05.txt") # Save to txt file to evaluate data later
    check_txtfile("RMSE_05.txt") # Save to txt file to evaluate data later
    check_txtfile('50_percentile_05.txt')
    check_txtfile('75_percentile_05.txt')
    check_txtfile('90_percentile_05.txt')
    """
    for sim_run in range(number_of_runs):
        print("Starting simulation number: ",sim_run)
        iterations = 0
        # Simualation set-up
        Simulation_run = Simulation()
        #Simulation_time = time.time()
        
        # Creating a gaussian model with the "real" field
        GP_real = Gaussian_process() 
        
        # Creating input gaussian white noise
        GP_real.white_noise() 
        # Initializing having "visited" all locations for the "real" field
        real_field = GP_real.gp("real",np.ones((50,50)),np.zeros((50,50)),0)
        
        # Using ocean data from SINMOD ocean model
        #real_field = GP_real.get_oceandata(2)
        #real_field -= 273.15 # Convertion from Kelvin to Celsuis 
        
        # Defining initial parameters        
        Simulation_run.wp = [0,0]
        Simulation_run.wp_last = [0,0]
    
        #Simulation_run.been[1,1] = 2
        Simulation_run.been[0,0] = 1
        
        fifty_threshold = np.percentile(real_field,50)
        sevnty_threshold = np.percentile(real_field, 70)
        ninety_threshold = np.percentile(real_field, 90)

        
        # Interating a number of steps for the AUV to move
        while (iterations < number_of_iterations):
            print("Starting interation:", iterations)
            been,measured = Simulation_run.temp_to_z(real_field)
            std_dev, pred_field, Sigma_0, Sigma = Simulation_run.next_waypoint(mode,real_field)
            Simulation_run.AUV_movement()
            
            #print("Ending interation:", iterations)
            
            
            # Print the total time spent on the simulation
            if iterations is (number_of_iterations - 1):

                t_max = np.amax(real_field)
                t_min = np.amin(real_field)
                
                Sigma = np.diagonal(Sigma)
                Sigma_0 = np.diagonal(Sigma_0)
                R_2 = 100*(1 - np.mean(Sigma)/np.mean(Sigma_0))
                RMSE = math.sqrt((1/2500)*(sum(sum(real_field)) - sum(sum(pred_field))))
                
                if R_2 > R_high:
                    R_high = R_2
                if R_2 < R_low:
                    R_low = R_2
                    
                if RMSE > RMSE_high:
                    RMSE_high = RMSE
                if RMSE < RMSE_low:
                    RMSE_low = RMSE
                    
                R += R_2
                RMSE_tot += RMSE
                print("Performance metric R^2 is: ",round(R_2), "%")
                print("Performance metric RMSE is: ",(RMSE))
                
                with open(str(my_path + '/Textfiles/Results/R2_09_test.txt'),'a') as f_1:
                    f_1.write('%f \n' %R_2)
                with open(str(my_path + '/Textfiles/Results/RMSE_09_test.txt'),'a') as f_2:
                    f_2.write('%f \n' %RMSE )
                
                # Removing nan entries in measurement matrix
                measured = measured[~np.isnan(measured)]
                
                fifty_counter = 0
                seventy_counter = 0
                ninety_counter = 0
                for i in range(len(measured.flatten())):
                    if measured.flatten()[i] > fifty_threshold:
                        fifty_counter += 1
                    if measured.flatten()[i] > sevnty_threshold:
                        seventy_counter += 1
                    if measured.flatten()[i] >  ninety_threshold:
                        ninety_counter += 1
                
                percent_over_fifty = 100*(fifty_counter/(np.amax(been) - 1))
                percent_over_seventy = 100*(seventy_counter/(np.amax(been) - 1))
                percent_over_ninety = 100*(ninety_counter/(np.amax(been) - 1))
                print("Percentage of time over 50 percentile:", percent_over_fifty)
                print("Percentage of time over 70 percentile:", percent_over_seventy)
                print("Percentage of time over 90 percentile:", percent_over_ninety)
                
                with open(str(my_path + '/Textfiles/Results/' + '50_percentile_09_test.txt'),'a') as f_3:
                    f_3.write('%f \n' %percent_over_fifty)
                with open(str(my_path + '/Textfiles/Results/' + '70_percentile_09_test.txt'),'a') as f_4:
                    f_4.write('%f \n' %percent_over_seventy)
                with open(str(my_path + '/Textfiles/Results/' + '90_percentile_09_test.txt'),'a') as f_5:
                    f_5.write('%f \n' %percent_over_ninety)
                
                # Set up for creating plot at the end of each simulation
                
                been_pos = np.nonzero(been)
                x1 = np.zeros(len(been_pos[0]))
                y1 = np.zeros(len(been_pos[1]))
               
                
                for i in range(len(x1)):
                    pos = math.floor(been[been_pos[0][i],been_pos[1][i]] - 2)
                    x1[pos] = been_pos[0][i]
                    y1[pos] = been_pos[1][i]
                
                """
                plt.figure(fig_num)
                plt.imshow(real_field, vmin=t_min,vmax =t_max)
                #plt.plot(x1,y1, '--k')
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.savefig(str(my_path + '/Figures/Results/' +  + '.jpg'), format='jpg')
                """
                
                plt.figure(fig_num + 1)
                plt.imshow(pred_field, vmin=t_min,vmax =t_max)
                plt.xlabel('West --- East')
                plt.ylabel('South --- North')
                plt.plot(x1,y1, '--k', label = 'AUV path')
                plt.gca().invert_yaxis()
                bar = plt.colorbar()
                bar.set_label('Temperature [$^\circ$C]')
                plt.savefig(str(my_path + '/Figures/Results/' + 'lawn_synth_pred' + '.jpg'), format='jpg')
                
                
                plt.figure(fig_num + 2)
                plt.imshow(std_dev)
                plt.xlabel('West --- East')
                plt.ylabel('South --- North')
                plt.plot(x1,y1, '--k')
                plt.gca().invert_yaxis()
                plt.colorbar()
                plt.savefig(str(my_path + '/Figures/Results/' + 'lawn_synth_unc' +'.jpg'), format='jpg')
                
                fig_num += 3 
                
            iterations += 1
            
    Tot_time = (time.time() - Simulation_time) # End time
    Tot_time = round(Tot_time)
    print("Total simulation time: ",Tot_time, "seconds")
    R /= number_of_runs # Average of performance
    RMSE_tot /= number_of_runs # Average of performance
    """
    print("Average performance metic R^2 over",number_of_runs,"runs: ", R)
    print("Lowest performance metic R^2 over",number_of_runs,"runs: ", R_low)
    print("Highest performance metic R^2 over",number_of_runs,"runs: ", R_high)
    print("Average performance metic RMSE over",number_of_runs,"runs: ", RMSE)
    print("Lowest performance metic RMSE over",number_of_runs,"runs: ", RMSE_low)
    print("Highest performance metic RMSE over",number_of_runs,"runs: ", RMSE_high)
    """
        
