import numpy as np
import scipy.linalg as sl
import pandas as pd
import h5py


def rho_of_vec (vec):
    vec.shape = (vec.shape[0], 3)
    rho = np.empty(shape=(vec.shape[0], 2, 2), dtype='complex')
    rho[:, 0, 0] = vec[:, 0]            # excited state population
    rho[:, 1, 1] = 1 - vec[:, 0]        # ground state population
    rho[:, 0, 1] = vec[:, 1] + 1j * vec[:, 2] # coherence
    rho[:, 1, 0] = vec[:, 1] - 1j * vec[:, 2] # conjugate of coherence

    return rho


def extract_rho(f_name, g):
    vec, dt = extract_y3(f_name, g)
    # vec [0] - p1 excited state population
    # vec [1] - coherence Real
    # vec [2] - coherence Imag
    rho = rho_of_vec(vec)
    # rho[:,0,0] excited state population
    # rho[:,1,1] ground state population
    # rho[:,0,1] coherence
    # rho[:,1,0] complex conjugate of coherence
    return rho, dt

def extract_y3(f_name, g):
    with h5py.File(f_name, 'r') as f:
        p0 = np.array(f[g]['p0'])  # ground state population
        p1 = 1 - p0
        # p1 = np.array(f[g]['p1'])   # exited state population
        assert (np.max(p0 + p1) - np.min(p0 + p1) < 1E-3)

        s_re = np.array(f[g]['s_re'])  # the coherence Real part
        s_im = np.array(f[g]['s_im'])   # the coherence Imag part

        t = np.array(f[g]['t']).reshape(-1)
        t = t.flatten()
        dt = t[1] - t[0]
        assert ((np.abs((t[1:] - t[:-1]) - dt) < 1e-6).all())

    y = np.vstack((p1, s_re, s_im)) # exited state population, coherence Real, Imag parts
    return y.T, dt

def fidelity_distances(data_directory, best_rho):
    
    init_rho = []
    fidelity_distances = []
    
    for i in range(0,20):
        gamma = '2.5133'
                
        file = data_directory + "State_D" + str(i+1) + '_2CUT_data.h5'
        rho, dt = extract_rho(file, gamma)
        
        init_rho.append(rho[1])
        
        dF = abs(np.trace(sl.sqrtm(sl.sqrtm(init_rho[i]) @ best_rho @ sl.sqrtm(init_rho[i]))))
        
        fidelity_distances.append(dF)
        
    return fidelity_distances
    
def extract_time(f_name, g):
    with h5py.File(f_name, 'r') as f:

        t = np.array(f[g]['t']).reshape(-1)
        t = t.flatten()
        dt = t[1] - t[0]
        assert ((np.abs((t[1:] - t[:-1]) - dt) < 1e-6).all())

    return t, dt

def read_as_dataframe(filename, fid_distances, data_dir, tests_dir):

    df = pd.DataFrame()
    
    print("Processing...")
    
    with h5py.File(tests_dir+filename, "r") as f:
        
        #gammas = ['0.079477', '0.25133', '0.79477', '2.5133', '7.9477', '25.133', '79.477', '251.33']
        gammas = f.keys()
        
        for gamma in gammas:
            
            print(gamma)
            
            g = gamma[6:]
            
            init_states = f[gamma].keys()
            
            for state in init_states:
                
                fidelity = f[gamma][state]["fidelity"][()]
                
                ser_len = len(fidelity)

                gamma_column = [g] * ser_len
                state_column = [state[7:]] * ser_len

                fid_dist_column = [fid_distances[int(state[7:])-1]] * ser_len      

                f_name = data_dir + "/" + state + "_2CUT_data.h5"
                #f_name = data_dir + "\\" + state + "_2CUT_data.h5"
                t, dt = extract_time(f_name, g)
                time_column = t
                gamma_time_column = np.array(t)*float(g) 

                d_ser = {'Gamma': gamma_column,
                         'State': state_column,
                         'Time': time_column, 
                         'gt': gamma_time_column,
                         'Fidelity': fidelity,
                         'Infidelity': 1-fidelity,
                         'Distance': fid_dist_column}

                df_ser = pd.DataFrame(data = d_ser)   
                df = pd.concat([df, df_ser])
    
    print(" done!")
    
    return df