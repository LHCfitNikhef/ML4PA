#!/usr/bin/env python
import numpy as np
import sys
import lhapdf
cfg = lhapdf.getConfig()
cfg.set_entry("Verbosity", 0)

def get_pdf(pdf_name, Xgrid, Q0, rep=0):

    Nx = len(Xgrid)

    GLUON = np.zeros(Nx, dtype=float)

    GLUON_LHAPDF = lhapdf.mkPDF(pdf_name, rep)


    for ix in range(Nx):
        x = Xgrid[ix]
        GLUON[ix] = GLUON_LHAPDF.xfxQ(0, x, Q0)

    return GLUON

        
if __name__ == '__main__':
    
    pdf_name = sys.argv[1]

    xmin = -6
    xmax = 0
    nx = 1001
    Nrep = 100
    Q0=2

    Xgrid = np.logspace(xmin, xmax, nx)
    Xgrid = Xgrid[:-1]

    gluon_avg = get_pdf(pdf_name,Xgrid,Q0=Q0, rep=0)

    gluon_reps = []
    for rep in range(1,Nrep+1):
        gluon_rep = get_pdf(pdf_name, Xgrid, Q0=Q0, rep=rep)
        gluon_reps.append(gluon_rep)

    gluon_std = np.std(gluon_reps, axis=0)

    data = open("gluon_"+pdf_name+"_xmin1e"+str(xmin)+".dat", "w")
    data.write("# pdf: "+pdf_name+"\n")
    data.write("# Q0: "+str(Q0)+" GeV \n")
    data.write("# Nrep: "+str(Nrep)+"\n")
    data.write("# [xmin, xmax, nx]: [10^("+str(xmin)+"), 10^("+str(xmax)+"), "+str(nx)+"]\n")
    data.write("# \t x \t gluon(cv) \t gluon(sd) \n ")

    float_f = '{:>10.8f}'

    for ix in range(nx-1):
        line = []
        line.append(str(float_f.format(Xgrid[ix], 4))+" \t ")
        line.append(str(float_f.format(gluon_avg[ix],4))+" \t ")
        line.append(str(float_f.format(gluon_std[ix],4))+" \n ")
        data.writelines(line)

    data.close()

        
