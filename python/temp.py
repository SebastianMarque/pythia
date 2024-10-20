import numpy as np
import math as math

def read_pythia_event(file_name):
    event = {}

    with open(file_name, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        
        # Skip lines that are not part of the event listing (e.g., headers, footers)
        if line.startswith("no") or line.startswith("Charge sum") or line.startswith("--------"):
            continue
        
        # Split the line into components
        parts = line.split()

        # If this line has particle data (starts with an event number)
        if len(parts) > 1 and parts[0].isdigit():
            particle_no = int(parts[0])  # The event number (first column)

            # Define the particle data as a dictionary
            particle_data = {
                'id': int(parts[1]),          # Particle ID
                'name': parts[2],             # Particle name
                'status': int(parts[3]),      # Particle status
                'mothers': [int(parts[4]), int(parts[5])],  # Mother particles
                'daughters': [int(parts[6]), int(parts[7])],  # Daughter particles
                'colours': [int(parts[8]), int(parts[9])],    # Colour codes
                'momentum': {
                    'p_x': float(parts[10]),  # x-component of momentum
                    'p_y': float(parts[11]),  # y-component of momentum
                    'p_z': float(parts[12]),  # z-component of momentum
                    'e': float(parts[13]),    # Energy
                    'm': float(parts[14])     # Mass
                }
            }

            # Store the particle data in the event dictionary with the event number as the key
            event[particle_no] = particle_data
    
    return event

def read_pythia_dipole(file_name):
    dipEnd = {}

    with open(file_name, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()

        # Skip lines that are not part of the dipole listing (e.g., headers, footers)
        if line.startswith("i") or line.startswith("--------"):
            continue

        # Split the line into components
        parts = line.split()

        # If this line has dipole data (starts with an index number)
        if len(parts) > 1 and parts[0].isdigit():
            dipole_index = int(parts[0])  # The index number (first column)

            # Define the dipole data as a dictionary
            dipole_data = {
                'syst': int(parts[1]),        # syst
                'side': int(parts[2]),        # side
                'rad': int(parts[3]),         # rad
                'rec': int(parts[4]),         # rec
                'pTmax': float(parts[5]),     # pTmax
                'col': int(parts[6]),         # col
                'chg': int(parts[7]),         # chg
            }

            # Store the dipole data in the dipEnd dictionary with dipole_index as the key
            dipEnd[dipole_index] = dipole_data

    return dipEnd

def read_random_numbers(filename):
    # Open the specified file and read its contents
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Initialize an empty dictionary to store the variables
    variables = {}

    # Parse each line and store the values in the dictionary
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace

        # Check if the line is not empty and contains an '=' character
        if '=' in line:
            key, value = line.split('=')
            variables[key.strip()] = float(value.strip())
          
    return variables

def event_dict_to_numpy(event_dict):
  
    # Determine the number of particles in the event
    num_particles = len(event_dict)
    
    # Initialize an empty array to store the momentum (p_x, p_y, p_z, e) for each particle
    event_array = np.zeros((num_particles, 4))
    
    # Iterate through the event dictionary and extract the momentum values
    for i, particle_data in event_dict.items():
        event_array[i] = np.array([particle_data['momentum']['p_x'], particle_data['momentum']['p_y'], 
                          particle_data['momentum']['p_z'], particle_data['momentum']['e']])
    
    return event_array



def rot(Mat, theta, phi):
    # Set up rotation matrix
    cthe = np.cos(theta)
    sthe = np.sin(theta)
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    
    # Define the 4x4 rotation matrix
    Mrot = np.array([
        [1.,           0.,         0.,          0.],
        [0.,  cthe * cphi,     -sphi, sthe * cphi],
        [0.,  cthe * sphi,      cphi, sthe * sphi],
        [0.,        -sthe,         0.,       cthe ]
    ])
    
    
    return np.dot(Mrot,Mat)


def bst(Mat, betaX, betaY, betaZ):
    # Set a small value to avoid division by zero or negative values under sqrt
    TINY = 1e-12
    
    
    # Calculate the Lorentz factor (gamma)
    beta2 = betaX**2 + betaY**2 + betaZ**2
    gm = 1.0 / np.sqrt(max(TINY, 1.0 - beta2))
    
    # Calculate gf = γ^2 / (1 + γ)
    gf = gm**2 / (1.0 + gm)
    
    # Define the boost matrix
    Mbst = np.array([
        [gm,              gm*betaX,          gm*betaY,          gm*betaZ],
        [gm*betaX, 1.0 + gf*betaX*betaX, gf*betaX*betaY, gf*betaX*betaZ],
        [gm*betaY, gf*betaY*betaX, 1.0 + gf*betaY*betaY, gf*betaY*betaZ],
        [gm*betaZ, gf*betaZ*betaX, gf*betaZ*betaY, 1.0 + gf*betaZ*betaZ]
    ])
    
    M = np.dot(Mbst, Mat)
    
    return M


def rotbst(tensor, M):
    
    if np.size(tensor[0]) == 1 : 
        """
        Apply the rotation/boost matrix M to this four-vector.
        
        Parameters:
        M: RotBstMatrix - an instance of RotBstMatrix containing the transformation matrix.
        """
        # Store original vector components
        t = np.copy(tensor[3])
        z = np.copy(tensor[2])
        y = np.copy(tensor[1])
        x = np.copy(tensor[0])
        
        # Apply the matrix multiplication
        tt = M[0][0] * t + M[0][1] * x + M[0][2] * y + M[0][3] * z
        xx = M[1][0] * t + M[1][1] * x + M[1][2] * y + M[1][3] * z
        yy = M[2][0] * t + M[2][1] * x + M[2][2] * y + M[2][3] * z
        zz = M[3][0] * t + M[3][1] * x + M[3][2] * y + M[3][3] * z
        
        tensor[3] = tt
        tensor[2] = zz
        tensor[1] = yy
        tensor[0] = xx
        
    else:
        
        tensor = M @ tensor
    
    return tensor

def p(xx,yy,zz,tt, vec4):
    
    vec4[0],vec4[1],vec4[2],vec4[3] = xx , yy , zz , tt
    
    return vec4

def m(vec4):
    
    return np.sqrt( -vec4[0]**2 - vec4[1]**2 - vec4[2]**2 + vec4[3]**2 )


def transpose_all(): # function to turn my vectors into co-vectors such that they
                     # can be better visualized
    
    global     pMother, pSister, pNewRec , pNewColPartner, sumNew
    
    pMother         = np.transpose(pMother)
    pSister         = np.transpose(pSister)
    pNewRec         = np.transpose(pNewRec)
    pNewColPartner  = np.transpose(pNewColPartner)
    sumNew          = np.transpose(sumNew)
    
    
    return


def px(vec4):
    return vec4[0]
def py(vec4):
    return vec4[1]
def pz(vec4):
    return vec4[2]
def e(vec4):
    return vec4[3]

def Theta(vec4):
    xx = vec4[0]
    yy = vec4[1]
    zz = vec4[2]

    return math.atan2(np.sqrt(xx**2 + yy**2),zz)

def append(This,ToThat):
    
    This[0],This[1],This[2],This[3] = ToThat[0],ToThat[1],ToThat[2],ToThat[3]
    
    return This





def branch( phi , z,  pT2 , begin_event,  begin_dipEnd ):
      
    ################### VARIABLES DEF
    
    Ebeams   = begin_event[0]['momentum']['m']
    
    
    x1       = begin_event[3]['momentum']['e'] / begin_event[1]['momentum']['e']
    x2       = begin_event[4]['momentum']['e'] / begin_event[2]['momentum']['e']
    mII      = begin_event[5]['momentum']['m']
    m2Sister = begin_event[3]['momentum']['m']
    m2II     = mII**2
    m2Dip    = m2II
    sideSign = 1.0
    Q2       = pT2 / (1. - z);
    pT2corr  = Q2 - z * (m2Dip + Q2) * (Q2 + m2Sister) / m2Dip
    xMo      = x1/z
    x1New    = xMo
    x2New    = x2
    

    
    
    
    # // Kinematics for II dipole.
    # // Construct kinematics of mother, sister and recoiler in old rest frame.
    # // Normally both mother and recoiler are taken massless.
    
    eNewRec       = 0.5 * (m2II + Q2) / mII
    pzNewRec      = -sideSign * eNewRec
    pTbranch      = np.sqrt(pT2corr) * m2II / ( z * (m2II + Q2) )
    pzMother      = sideSign * 0.5 * mII * ( (m2II - Q2)
                    / ( z * (m2II + Q2) ) + (Q2 + m2Sister) / m2II )         
    
    
    # // Common final kinematics steps for both normal and rescattering.
    eMother  = np.sqrt( pTbranch**2 + pzMother**2 )
    pzSister = pzMother + pzNewRec
    eSister  = np.sqrt(pTbranch**2 + pzSister**2 + m2Sister )
    
    
    pMother         = np.zeros(4)
    pSister         = np.zeros(4)
    pNewRec         = np.zeros(4)
    pNewColPartner  = np.zeros(4)
    
    
    

    event     = event_dict_to_numpy(begin_event)

    #### 
    event    = np.vstack((event, np.copy(event[3])))
    event    = np.vstack((event, np.copy(event[4])))
    event    = np.vstack((event, np.copy(event[5])))
    event    = np.vstack((event, np.zeros(4      )))
    
    
    
    
    ## constructing the first vertext from beam energies. 
    
    
    pMother = p( pTbranch, 0., pzMother, eMother , pMother )
    pSister = p( pTbranch, 0., pzSister, eSister , pSister )
    pNewRec = p(       0., 0., pzNewRec, eNewRec , pNewRec )
    
    event[9] = np.copy(pSister)
    
    
    
    daughter     = np.copy(event[3])
    mother       = np.copy(event[6])
    newRecoiler  = np.copy(event[7])
    sister       = np.copy(event[9])
    
    
    mother       = np.copy(pMother)
    event[6]     = np.copy(pMother)
    
    newRecoiler  = np.copy(pNewRec)
    event[7]     = np.copy(pNewRec)
    
    
    
    ## Firstboost define as Mtot
    
    Mtot        = np.eye(4)
    Mtot        = bst(Mtot, 0.0 , 0.0 , (x2-x1) / (x1+x2) )
    Mtot        = rot(Mtot,0,-phi)
    
    
    MfromRest   = np.eye(4)
    
    sumNew      =  pMother + pNewRec
    
    betaX       = px(sumNew) / e(sumNew);
    betaZ       = pz(sumNew) / e(sumNew);
    
    MfromRest   = bst(MfromRest, -betaX, 0., -betaZ);
    
    pMother     = rotbst(pMother, MfromRest)
    
    theta       = Theta(pMother)
    
    
    MfromRest   = rot(MfromRest, -theta, phi)
    
    
    MfromRest   = bst( MfromRest,0., 0., (x1New - x2New) / (x1New + x2New) )
    
    
    Mtot        = rotbst(Mtot       , MfromRest)
    
    
    mother      = rotbst(mother     , MfromRest)
    event[6]    = rotbst(event[6]   , MfromRest)
    
    
    newRecoiler = rotbst(newRecoiler , MfromRest)
    event[7]    = rotbst(event[7]    , MfromRest)
    
    
    sister      = rotbst(sister      , MfromRest)
    event[9]    = rotbst(event[9]    , MfromRest)
    
    
    
    event[8]    = rotbst(event[8]   , Mtot)
    
    return event
    
#%%



begin_event = read_pythia_event('begin_event.txt')
begin_dipEnd = read_pythia_dipole('begin_dipole.txt')
variables = read_random_numbers('random_numbers.txt')

phi = variables.get('phi')
z = variables.get('z')
pT2 = variables.get('pT2')

e = branch(phi,z,pT2,begin_event,begin_dipEnd)

#%%




