from numpy import zeros,sqrt,arange,array,repeat,digitize,histogram,pi

import numpy as np

from datetime import datetime

from collections import defaultdict

import json



# import tracemalloc

import sys

from pylab import plot, show, xlabel, ylabel, imshow, hot, xlim, ylim, gray,figure,legend

import matplotlib.pyplot as plt


fast = True


def rdf ( box, r_cut, r ):

    """Takes in box, cutoff range, and coordinate array, and calculates forces and potentials etc."""

    import numpy as np

    from itertools import product

    import math

    

    # It is assumed that positions are in units where box = 1

    # Forces are calculated in units where sigma = 1 and epsilon = 1

    # Uses neighbour lists



    n = r.shape[0]



    # Set up vectors to half the cells in neighbourhood of 3x3x3 cells in cubic lattice

    # The cells are chosen so that if (d0,d1,d2) appears, then (-d0,-d1,-d2) does not.

    d = np.array ( [ [ 0, 0, 0], [ 1, 0, 0], [ 1, 1, 0], [-1, 1, 0],

                     [ 0, 1, 0], [ 0, 0, 1], [-1, 0, 1], [ 1, 0, 1], [-1,-1, 1],

                     [ 0,-1, 1], [ 1,-1, 1], [-1, 1, 1], [ 0, 1, 1], [ 1, 1, 1] ] )



    r = r - np.rint(r) # Ensure all atoms in periodic box

    

    sr2_ovr      = 1.77 # Overlap threshold (pot > 100)

    r_cut_box    = r_cut / box

    r_cut_box_sq = r_cut_box ** 2

    box_sq       = box ** 2





    # Calculate cell index triplets

    sc = math.floor(box/r_cut)                          # Number of cells along box edge

    assert sc >= 3, 'System is too small for cells'     # Guard against box being too small

    c  = np.floor((r+0.5)*sc).astype(np.int_)           # N*3 array of cell indices for all atoms

    #assert np.all(c>=0) and np.all(c<sc), 'Index error' # Simplistic "guard" against roundoff

    print("*********")

    #print(c)

    if fast:

        

        # Build list of arrays, each array holding positions of atoms in a cell

        # At the same time, define a matching set of force arrays in each cell

        # i and j number the atoms in each cell; we do not refer explicitly to indices in r

        rc = []                     # Initially empty lists of positions and forces

        for ci in product(range(sc),repeat=3): # Triple loop over cells

            #print("ci:", ci)



            mask = np.all(c==ci,axis=1)        # Mask identifies atoms in this cell
            #print(mask)
            #exit()
            rc.append(r[mask,:])               # Copy atom coordinates into array, add to list
        #exit()    
        #print(rc)
        #print(len(rc))
        #exit()

        for ci1, rci in enumerate(rc):            # Loop over i-cells, getting all atoms in each i-cell as an array

            ci = np.unravel_index(ci1,(sc,sc,sc)) # Get i-cell triple-indices

            if rci.size==0:                       # Handle empty cell

                continue



            for dj in d:                                              # Loop over neighbouring j-cells

                cj  = ci + dj                                         # Compute neighbour j-cell triple-indices

                cj1 = np.ravel_multi_index(cj,(sc,sc,sc),mode='wrap') # Convert j-cell to single-index

                rcj = rc[cj1]                                         # Get atoms in j-cell as an array

                if rcj.size==0:                                       # Handle empty cell

                    continue



                rij      = rci[:,np.newaxis,:]-rcj[np.newaxis,:,:] # Separation vectors for all i and j

                rij      = rij - np.rint(rij)                      # PBCs in box=1 units

                rij_sq   = np.sum(rij**2,axis=2)                   # Squared separations

                in_range = rij_sq < r_cut_box_sq                   # Set flags for within cutoff



                if ci1==cj1:

                    np.fill_diagonal(in_range,False) # Eliminate i==j when i-cell==j-cell

                    np.fill_diagonal(rij_sq,1.0)     # Avoid divide-by-zero below



                rij_sq = rij_sq * box_sq                         # Now in sigma=1 units

                rij    = rij * box                               # Now in sigma=1 units

    return rij







start_time = datetime.now()

# tracemalloc.start()



    

def main():  

    if len(sys.argv) != 2:

        print("If running while in codes folder")

        print("python rdf_link_cell_events_addclass.py /Users/marx/Documents/Research/Molecular_Dynamics/Lammps_Files/moved_to_be_plotted/mil_dump.shear")

        sys.exit()

    else:

        choice = "range" #raw_input("Do you want range or list:")

        print(choice)

        while choice not in {"range", "list"}:

            choice = raw_input("Please enter range or list:")

            print("*",choice,"*")

        if choice == "range":

            lst = []

            range_end = 1 # raw_input("Range upto what frame:")

            for i in range (0,int(range_end)):

                lst.append(i)

            end_frame = lst

            #print("Values of endframe as range:", end_frame)

        if choice == "list":

            lst = []

            n = int(raw_input("Enter number of elements:"))

            for i in range (0,n):

                ele = int(raw_input("Frame Number:"))

                lst.append(ele)

            end_frame = lst

        print("Running for these selected frames:", end_frame)

            

    frames = []

    xlo = []

    ylo = []

    zlo = []

    lx = []

    ly = []

    lz = [] 

    rc = 2

    r11 = 1.4 # the first valley from the previous code for type 1-1

    r22 = 1.05 # check if it is 1.55 second valley

    r12 = 1.25

    r11_count = 0



    t = 0

    #time_frame = int(sys.argv[2])

    #print("timeframe:", time_frame)

    nframes = 0

    dr = 0.05

    atom_not_added = 0 

    print("Reading upto frames from input:", max(end_frame)+1)

    with open(sys.argv[1], 'r') as reader:

    #with open('/Users/marx/Documents/Research/PL_Files/lata4olivia', 'r') as reader:

        # Read and print the entire file line by line

        line = reader.readline()

        while line != '' and float(t)<=  (max(end_frame)+1) * 10000:

              # The EOF char is an empty string

            #print("In while loop")

            #print("linein while",  line)

            if line.rstrip() == "ITEM: TIMESTEP":

                line = reader.readline()

                line = line.rstrip('\n') # Removes any white spaces at the end of the string



                t = line

            

                #print ("t is:", t,nframes)

                x = []

                frames.append(x)

                nframes = nframes + 1

            if line.rstrip() == "ITEM: NUMBER OF ATOMS":

                line = reader.readline()

                line = line.rstrip('\n')

                line_split = line.split (" ")  

                na = int(line_split[0])  

                #na = 10

                #print ("Number of atoms =", na, str(t))

            if line.rstrip() == "ITEM: BOX BOUNDS pp pp pp":

                line = reader.readline()

                line = line.rstrip('\n')

                line_split = line.split (" ")

                xlo_num = float(line_split[0])

                xhi_num = float(line_split[1])

                line = reader.readline()

                line = line.rstrip('\n')

                line_split = line.split (" ")

                ylo_num = float(line_split[0])

                yhi_num = float(line_split[1])

                line = reader.readline()

                line = line.rstrip('\n')

                line_split = line.split (" ")

                zlo_num = float(line_split[0])

                zhi_num = float(line_split[1])

                #print("lx before append:", lx)

                

                lx.append(xhi_num - xlo_num)

                ly.append(yhi_num - ylo_num)

                lz.append(zhi_num - zlo_num)

                xlo.append(xlo_num)

                ylo.append(ylo_num)

                zlo.append(zlo_num)

            

                line = reader.readline()

                #print (xlo,xhi,ylo,yhi,zlo,zhi)

                #print(na)

                for i in range (0,na,1):

                                                      

                    line = reader.readline()

                    line = line.rstrip('\n')

                    #line = "100474 1 38.1935 50.6922 45.9594 -0.529932 1.21147 0.848939 8.47367 -7.96965 5.1452"

                    #print(i, ":::::", na, ":::::", line)

                    line_split = line.split (" ")

                #line_split = line.split(100474 1 38.1935 50.6922 45.9594 -0.529932 1.21147 0.848939 8.47367 -7.96965 5.1452

                    (iid,itype,ix,iy,iz,ivx,ivy,ivz,ifx,ify,ifz) = list(np.float_(line_split))

                    #min 30..the x in 2 lines below is the current frame

                    #print (iid,ix)

                    x = frames[nframes - 1]

                    #print("From file: ", iid, t)

                    x.append([ix,iy,iz])

                    #print("x added")

                



                    #print (nframes)





                #print (line_split)

            line = reader.readline()

            #print("T at end of while:", t)



            #print("line at end of while",  line)

    #exit()

    #time_frame= 0

    #print(frames)

    #a = np.array (frames)

    #vs = np.hstack(frames)

    y=np.array([np.array(xi) for xi in frames[1]])

    #print(a)

    #print(vs)

    print(y[1][0])



    #exit()  

    ix = int(lx[0]/rc)

    iy = int(ly[0]/rc)

    iz = int(lz[0]/rc) 

    rdf_rt = rdf (ix, rc, y )

    print("FINAL RESULT:")

    print(rdf_rt)
    print(rdf_rt.shape[0])
    print(rdf_rt.shape[1])
    print("One row... ")
    print(rdf_rt[0])


    print('Duration: {}'.format(end_time - start_time))

        # print("Current: %d, Peak %d" % .get_traced_memory())





if __name__ == "__main__":

    main()

