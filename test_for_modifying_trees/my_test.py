import uproot
import awkward as ak
import numpy as np

#┌────────┐
#│ TEST 1 │
#└────────┘
# Create some data
branch1 = ak.Array([[1, 2, 3], [4, 5, 6]])
branch2 = ak.Array([[1.1, 2.2, 3.3, 4.4], [5.5, 6.6]])


# Save the data to a ROOT file
with uproot.recreate("my_file.root") as f:
    f["tree1"] = {"branch1": branch1, "branch2": branch2}

# Read the data back from the ROOT file
#print every branch with name and content
with uproot.open("my_file.root") as f:
    print(f["tree1"].keys())
    print(f["tree1"]["branch1"].array())
    print(f["tree1"]["branch2"].array())

# UPROOT must save the number of entries in each subarray, so it creates a
# branch called "nbranch1" and "nbranch2" to store this information.
#If branch1 and branch2 have the same number of subentries and 
#we want to avoid the creation of nbranch1 and nbranch2, we can use the
# ak.zip function to zip the two arrays together.

#┌────────┐
#│ TEST 2 │
#└────────┘

branch1 = ak.Array([[1, 2, 3], [4, 5, 6]])
branch2 = ak.Array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])

# Save the data to a ROOT file
with uproot.recreate("my_file.root") as f:
    f["tree1"] = ak.zip({"branch1": branch1, "branch2": branch2})

# Read the data back from the ROOT file
#print every branch with name and content
with uproot.open("my_file.root") as f:
    print(f["tree1"].keys())
    print(f["tree1"]["branch1"].array())
    print(f["tree1"]["branch2"].array())


#Now it works better, because there is a single branch named "n" that stores
#the number of entries for both branch1 and branch2. Try and use
#a dict of NumPy arrays now, and see what happens.

#┌────────┐
#│ TEST 3 │
#└────────┘

branch1 = np.array([[1, 2, 3], [4, 5, 6]])
branch2 = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])

#make dict of NumPy arrays
dict = {"branch1": branch1, "branch2": branch2}

# Save the data to a ROOT file
with uproot.recreate("my_file.root") as f:
    f["tree1"] = dict

# Read the data back from the ROOT file
#print every branch with name and content
with uproot.open("my_file.root") as f:
    print(f["tree1"].keys())
    print(f["tree1"]["branch1"].array())
    print(f["tree1"]["branch2"].array())

#it actually works bettere because no "n" is necessary (I guess because it
#is a NumPy array and not an awkward array). Testing out of curiosity
#what happens if we use numpy arrays without dict.

#┌────────┐
#│ TEST 4 │
#└────────┘

branch1 = np.array([[1, 2, 3], [4, 5, 6]])
branch2 = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
branch3 = np.array([[1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8]])

# Save the data to a ROOT file
with uproot.recreate("my_file.root") as f:
    f["tree1"] = {"branch1": branch1, "branch2": branch2, "branch3": branch3}

# Read the data back from the ROOT file
#print every branch with name and content
with uproot.open("my_file.root") as f:
    print(f["tree1"].keys())
    print(f["tree1"]["branch1"].array())
    print(f["tree1"]["branch2"].array())
    print(f["tree1"]["branch3"].array())

#I'm a bit confused because there is no "n" branch of any kind, but
#the arrays have variable length. Maybe it is only created when using awkward
#arrays

