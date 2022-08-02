import numpy as np
import pickle
from CogBook import cos_distance

"""
profiles_dict is a dictionary that maps names to the corresponding Profile object
"""

class Profile:
    def __init__(self, name):
        self.name = name
        self.vectors = [] # Array of length 512 vectors
    def add_vector(self, vector):
        self.vectors.append(vector)
    def clear_vectors(self):
        self.vectors.clear()
    def avg_vector(self):
        return np.mean(np.array(self.vectors), axis=0)
    def match_score(self, newVector):
        return cos_distance(np.mean(newVector, np.array(self.vectors), axis=0))
    

def save_profiles_to_file(profiles_dict):
    with open('profiles.pkl','wb') as f:
        pickle.dump(profiles_dict,f)

def load_profiles_from_file():
    with open('profiles.pkl','rb') as f:
        return pickle.load(f)

def add_empty_profile(profiles_dict, name):
    profiles_dict[name] = Profile(name)
    
def remove_profile(profiles_dict, name):
    # assumes name is a valid key of profiles_dict
    conf = input("Confirm deletion of profile "+name+"? (y/n)")
    if conf=="y":
        del profiles_dict[name]

def save_vector_to_profile(profiles_dict, name, vector):
    if name not in profiles_dict:
        add_empty_profile(profiles_dict, name)
    profiles_dict[name].add_vector(vector)

def query_database(vector, iden_threshold):
    """
    Queries the database for matches with the provided vector and identification threshold

    Parameters
    ----------
    vector : numpy.ndarray, shape-(512,)
        Descriptor vector for the input face

    iden_threshold : float
        The minimum cosine distance necessary for the function to consider two vectors a match

    Returns
    -------
    string
        The name of the profile that matches the provided vector most closely. Will return "unknown" if no match is found.
    """
    profiles_dict = load_profiles_from_file()
    
    mindist = 2
    for name, profile in profiles_dict.items():
        curdist = cos_distance(profile.avg_vector().reshape(1,-1),vector.reshape(1,-1))
        if  curdist < mindist:
            mindist = curdist
            minname = name
    if mindist<iden_threshold:
        return minname
    else:
        return "unknown"
