from Opt_Problems.Deterministic.S2MPJ_support_files.s2mpjlib import *
class  n3PK(CUTEst_problem):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : n3PK
#    *********
# 
#    A problem arising in the estimation of structured O/D matrix
# 
#    Source:  
#    M. Bierlaire, private communication
#    see also
#    M. Bierlaire and Ph. L. Toint,
#    MEUSE: an origin-destination estimator that exploits structure,
#    Transportation Research B, 29, 1, 47--60, 1995.
# 
#    SIF input: Ph. Toint, Dec 1989, Corrected July 1993.
# 
#    classification = "SBR2-MN-30-0"
# 
#  Parameters
# 
#  Number of parking columns
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'n3PK'

    def __init__(self, *args): 
        import numpy as np
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['NPKC'] = 3
        v_['NPKC-1'] = -1+v_['NPKC']
        v_['NPKC+1'] = 1+v_['NPKC']
        v_['NCENT'] = 6
        v_['NCENT-1'] = -1+v_['NCENT']
        v_['RNCENT-1'] = float(v_['NCENT-1'])
        v_['GAMMA'] = 1.0000e+04
        v_['FT0'] = 0.500000
        v_['FT1'] = 0.500000
        v_['FT2'] = 0.500000
        v_['WFT0'] = 1.000000
        v_['WFT1'] = 1.000000
        v_['WFT2'] = 1.000000
        v_['COUNT'] = 9
        v_['COUNT-1'] = -1+v_['COUNT']
        v_['DEFW'] = 999.999953
        v_['0'] = 0
        v_['1'] = 1
        v_['COU0'] = 910.000000
        v_['COU1'] = 175.000000
        v_['COU2'] = 1915.000000
        v_['COU3'] = 450.000000
        v_['COU4'] = 260.000000
        v_['COU5'] = 80.000000
        v_['COU6'] = 670.000000
        v_['COU7'] = 1450.000000
        v_['COU8'] = 990.000000
        v_['PHI0'] = 1.000000
        v_['PHI1'] = 1.000000
        v_['PHI2'] = 1.000000
        v_['PHI3'] = 1.000000
        v_['PHI4'] = 1.000000
        v_['PHI5'] = 1.000000
        v_['PHI6'] = 1.000000
        v_['PHI7'] = 1.000000
        v_['PHI8'] = 1.000000
        for I in range(int(v_['0']),int(v_['COUNT-1'])+1):
            v_['PHI'+str(I)] = v_['PHI'+str(I)]/v_['GAMMA']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = np.array([])
        self.xscale = np.array([])
        intvars   = np.array([])
        binvars   = np.array([])
        [iv,ix_,_] = s2mpj_ii('A1,0',ix_)
        self.xnames=arrset(self.xnames,iv,'A1,0')
        [iv,ix_,_] = s2mpj_ii('A2,0',ix_)
        self.xnames=arrset(self.xnames,iv,'A2,0')
        [iv,ix_,_] = s2mpj_ii('A3,0',ix_)
        self.xnames=arrset(self.xnames,iv,'A3,0')
        [iv,ix_,_] = s2mpj_ii('A4,0',ix_)
        self.xnames=arrset(self.xnames,iv,'A4,0')
        [iv,ix_,_] = s2mpj_ii('A5,0',ix_)
        self.xnames=arrset(self.xnames,iv,'A5,0')
        [iv,ix_,_] = s2mpj_ii('A0,1',ix_)
        self.xnames=arrset(self.xnames,iv,'A0,1')
        [iv,ix_,_] = s2mpj_ii('A2,1',ix_)
        self.xnames=arrset(self.xnames,iv,'A2,1')
        [iv,ix_,_] = s2mpj_ii('A3,1',ix_)
        self.xnames=arrset(self.xnames,iv,'A3,1')
        [iv,ix_,_] = s2mpj_ii('A4,1',ix_)
        self.xnames=arrset(self.xnames,iv,'A4,1')
        [iv,ix_,_] = s2mpj_ii('A5,1',ix_)
        self.xnames=arrset(self.xnames,iv,'A5,1')
        [iv,ix_,_] = s2mpj_ii('A0,2',ix_)
        self.xnames=arrset(self.xnames,iv,'A0,2')
        [iv,ix_,_] = s2mpj_ii('A1,2',ix_)
        self.xnames=arrset(self.xnames,iv,'A1,2')
        [iv,ix_,_] = s2mpj_ii('A3,2',ix_)
        self.xnames=arrset(self.xnames,iv,'A3,2')
        [iv,ix_,_] = s2mpj_ii('A4,2',ix_)
        self.xnames=arrset(self.xnames,iv,'A4,2')
        [iv,ix_,_] = s2mpj_ii('A5,2',ix_)
        self.xnames=arrset(self.xnames,iv,'A5,2')
        for J in range(int(v_['NPKC']),int(v_['NCENT-1'])+1):
            v_['J+1'] = 1+J
            v_['J-1'] = -1+J
            for I in range(int(v_['0']),int(v_['J-1'])+1):
                [iv,ix_,_] = s2mpj_ii('T'+str(I)+','+str(J),ix_)
                self.xnames=arrset(self.xnames,iv,'T'+str(I)+','+str(J))
            for I in range(int(v_['J+1']),int(v_['NCENT-1'])+1):
                [iv,ix_,_] = s2mpj_ii('T'+str(I)+','+str(J),ix_)
                self.xnames=arrset(self.xnames,iv,'T'+str(I)+','+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.A       = lil_matrix((1000000,1000000))
        self.gscale  = np.array([])
        self.grnames = np.array([])
        cnames      = np.array([])
        self.cnames = np.array([])
        gtype       = np.array([])
        [ig,ig_,_] = s2mpj_ii('G0,3',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T0,3']
        self.A[ig,iv] = float(0.010000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G1,3',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T1,3']
        self.A[ig,iv] = float(0.007143)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G2,3',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T2,3']
        self.A[ig,iv] = float(0.008333)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G4,3',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T4,3']
        self.A[ig,iv] = float(0.050000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G5,3',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T5,3']
        self.A[ig,iv] = float(0.050000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G0,4',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T0,4']
        self.A[ig,iv] = float(0.005000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G1,4',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T1,4']
        self.A[ig,iv] = float(0.005556)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G2,4',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T2,4']
        self.A[ig,iv] = float(0.050000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G3,4',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T3,4']
        self.A[ig,iv] = float(0.001667)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G5,4',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T5,4']
        self.A[ig,iv] = float(0.025000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G0,5',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T0,5']
        self.A[ig,iv] = float(0.020000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G1,5',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T1,5']
        self.A[ig,iv] = float(0.033333)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G2,5',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T2,5']
        self.A[ig,iv] = float(0.014286)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G3,5',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T3,5']
        self.A[ig,iv] = float(0.006667)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G4,5',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T4,5']
        self.A[ig,iv] = float(0.050000)+self.A[ig,iv]
        v_['TMP'] = 5.000000*v_['FT0']
        v_['TMP1'] = 1.0/v_['TMP']
        [ig,ig_,_] = s2mpj_ii('H0',ig_)
        gtype = arrset(gtype,ig,'<>')
        self.gscale = arrset(self.gscale,ig,float(v_['WFT0']))
        iv = ix_['A1,0']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A2,0']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A3,0']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A4,0']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A5,0']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        v_['TMP'] = 5.000000*v_['FT1']
        v_['TMP1'] = 1.0/v_['TMP']
        [ig,ig_,_] = s2mpj_ii('H1',ig_)
        gtype = arrset(gtype,ig,'<>')
        self.gscale = arrset(self.gscale,ig,float(v_['WFT1']))
        iv = ix_['A0,1']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A2,1']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A3,1']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A4,1']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A5,1']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        v_['TMP'] = 5.000000*v_['FT2']
        v_['TMP1'] = 1.0/v_['TMP']
        [ig,ig_,_] = s2mpj_ii('H2',ig_)
        gtype = arrset(gtype,ig,'<>')
        self.gscale = arrset(self.gscale,ig,float(v_['WFT2']))
        iv = ix_['A0,2']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A1,2']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A3,2']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A4,2']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        iv = ix_['A5,2']
        self.A[ig,iv] = float(v_['TMP1'])+self.A[ig,iv]
        for I in range(int(v_['0']),int(v_['COUNT-1'])+1):
            [ig,ig_,_] = s2mpj_ii('K'+str(I),ig_)
            gtype = arrset(gtype,ig,'<>')
            self.gscale = arrset(self.gscale,ig,float(v_['PHI'+str(I)]))
        v_['TMP1'] = 200.000000/v_['COU7']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A1,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 200.000000/v_['COU4']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K4',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A1,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 200.000000/v_['COU2']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A1,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 480.000000/v_['COU8']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K8',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A2,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 480.000000/v_['COU7']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A2,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 480.000000/v_['COU6']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K6',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A2,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 480.000000/v_['COU2']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A2,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 120.000000/v_['COU2']
        v_['TMP'] = 1.000000*v_['TMP1']
        iv = ix_['A3,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 360.000000/v_['COU7']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A4,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 360.000000/v_['COU2']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A4,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 560.000000/v_['COU8']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K8',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A5,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 560.000000/v_['COU7']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A5,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 560.000000/v_['COU2']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A5,0']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 240.000000/v_['COU0']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 400.000000/v_['COU8']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K8',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A2,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 400.000000/v_['COU7']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A2,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 400.000000/v_['COU6']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K6',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A2,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 400.000000/v_['COU2']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A2,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 400.000000/v_['COU0']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A2,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 420.000000/v_['COU2']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A3,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 420.000000/v_['COU0']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A3,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 180.000000/v_['COU7']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A4,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 180.000000/v_['COU2']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A4,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 180.000000/v_['COU0']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A4,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 320.000000/v_['COU8']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K8',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A5,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 320.000000/v_['COU7']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A5,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 320.000000/v_['COU2']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A5,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 320.000000/v_['COU0']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A5,1']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 20.000000/v_['COU1']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,2']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 20.000000/v_['COU0']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,2']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 60.000000/v_['COU1']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A1,2']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 40.000000/v_['COU2']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A3,2']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 40.000000/v_['COU1']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A3,2']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 40.000000/v_['COU0']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A3,2']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 120.000000/v_['COU5']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K5',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A4,2']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 20.000000/v_['COU8']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K8',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A5,2']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP1'] = 20.000000/v_['COU5']
        v_['TMP'] = 1.000000*v_['TMP1']
        [ig,ig_,_] = s2mpj_ii('K5',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A5,2']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU7']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T0,3']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU3']
        [ig,ig_,_] = s2mpj_ii('K3',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T0,3']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU7']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T1,3']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU4']
        [ig,ig_,_] = s2mpj_ii('K4',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T1,3']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU8']
        [ig,ig_,_] = s2mpj_ii('K8',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T2,3']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU7']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T2,3']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU6']
        [ig,ig_,_] = s2mpj_ii('K6',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T2,3']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU7']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T4,3']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU8']
        [ig,ig_,_] = s2mpj_ii('K8',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T5,3']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU7']
        [ig,ig_,_] = s2mpj_ii('K7',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T5,3']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU3']
        [ig,ig_,_] = s2mpj_ii('K3',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T0,4']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU4']
        [ig,ig_,_] = s2mpj_ii('K4',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T1,4']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU8']
        [ig,ig_,_] = s2mpj_ii('K8',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T2,4']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU6']
        [ig,ig_,_] = s2mpj_ii('K6',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T2,4']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU3']
        [ig,ig_,_] = s2mpj_ii('K3',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T3,4']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU2']
        [ig,ig_,_] = s2mpj_ii('K2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T3,4']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU8']
        [ig,ig_,_] = s2mpj_ii('K8',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T5,4']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU6']
        [ig,ig_,_] = s2mpj_ii('K6',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T0,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU1']
        [ig,ig_,_] = s2mpj_ii('K1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T0,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU0']
        [ig,ig_,_] = s2mpj_ii('K0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T0,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU6']
        [ig,ig_,_] = s2mpj_ii('K6',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T1,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU1']
        [ig,ig_,_] = s2mpj_ii('K1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T1,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU6']
        [ig,ig_,_] = s2mpj_ii('K6',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T2,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU6']
        iv = ix_['T3,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU2']
        [ig,ig_,_] = s2mpj_ii('K2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T3,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU1']
        [ig,ig_,_] = s2mpj_ii('K1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T3,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU0']
        [ig,ig_,_] = s2mpj_ii('K0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T3,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU6']
        [ig,ig_,_] = s2mpj_ii('K6',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T4,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        v_['TMP'] = 1.000000/v_['COU5']
        [ig,ig_,_] = s2mpj_ii('K5',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['T4,5']
        self.A[ig,iv] = float(v_['TMP'])+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L1,0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A1,0']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A2,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L2,0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A1,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A2,0']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A3,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L3,0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A1,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A2,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,0']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A4,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L4,0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A1,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A2,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,0']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A5,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L5,0',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A1,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A2,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,0']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,0']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        [ig,ig_,_] = s2mpj_ii('L0,1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,1']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A2,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L2,1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A2,1']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A3,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L3,1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A2,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,1']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A4,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L4,1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A2,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,1']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A5,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L5,1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A2,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,1']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,1']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        [ig,ig_,_] = s2mpj_ii('L0,2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,2']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A1,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L1,2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A1,2']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A3,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L3,2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A1,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,2']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A4,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L4,2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A1,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,2']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        iv = ix_['A5,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('L5,2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['A0,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A1,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A3,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A4,2']
        self.A[ig,iv] = float(0.200000)+self.A[ig,iv]
        iv = ix_['A5,2']
        self.A[ig,iv] = float(-0.800000)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(0.500000))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = np.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = np.zeros((ngrp,1))
        for J in range(int(v_['NPKC']),int(v_['NCENT-1'])+1):
            v_['J+1'] = 1+J
            v_['J-1'] = -1+J
            for I in range(int(v_['0']),int(v_['J-1'])+1):
                self.gconst = arrset(self.gconst,ig_['G'+str(I)+','+str(J)],float(1.0))
            for I in range(int(v_['J+1']),int(v_['NCENT-1'])+1):
                self.gconst = arrset(self.gconst,ig_['G'+str(I)+','+str(J)],float(1.0))
        for J in range(int(v_['0']),int(v_['NPKC-1'])+1):
            self.gconst = arrset(self.gconst,ig_['H'+str(J)],float(1.0))
        for J in range(int(v_['0']),int(v_['COUNT-1'])+1):
            self.gconst = arrset(self.gconst,ig_['K'+str(J)],float(1.0))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.zeros((self.n,1))
        self.x0[ix_['A1,0']] = float(v_['FT0'])
        self.x0[ix_['A2,0']] = float(v_['FT0'])
        self.x0[ix_['A3,0']] = float(v_['FT0'])
        self.x0[ix_['A4,0']] = float(v_['FT0'])
        self.x0[ix_['A5,0']] = float(v_['FT0'])
        self.x0[ix_['A0,1']] = float(v_['FT1'])
        self.x0[ix_['A2,1']] = float(v_['FT1'])
        self.x0[ix_['A3,1']] = float(v_['FT1'])
        self.x0[ix_['A4,1']] = float(v_['FT1'])
        self.x0[ix_['A5,1']] = float(v_['FT1'])
        self.x0[ix_['A0,2']] = float(v_['FT2'])
        self.x0[ix_['A1,2']] = float(v_['FT2'])
        self.x0[ix_['A3,2']] = float(v_['FT2'])
        self.x0[ix_['A4,2']] = float(v_['FT2'])
        self.x0[ix_['A5,2']] = float(v_['FT2'])
        self.x0[ix_['T0,3']] = float(100.000000)
        self.x0[ix_['T1,3']] = float(140.000000)
        self.x0[ix_['T2,3']] = float(120.000000)
        self.x0[ix_['T4,3']] = float(20.000000)
        self.x0[ix_['T5,3']] = float(20.000000)
        self.x0[ix_['T0,4']] = float(200.000000)
        self.x0[ix_['T1,4']] = float(180.000000)
        self.x0[ix_['T2,4']] = float(20.000000)
        self.x0[ix_['T3,4']] = float(600.000000)
        self.x0[ix_['T5,4']] = float(40.000000)
        self.x0[ix_['T0,5']] = float(50.000000)
        self.x0[ix_['T1,5']] = float(30.000000)
        self.x0[ix_['T2,5']] = float(70.000000)
        self.x0[ix_['T3,5']] = float(150.000000)
        self.x0[ix_['T4,5']] = float(20.000000)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = s2mpj_ii('gSQUARE',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in np.arange(0,ngrp):
            self.grelt.append(np.array([]))
        self.grftype = np.array([])
        self.grelw   = []
        nlc         = np.array([])
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        self.xlower = np.zeros((self.n,1))
        self.xupper = np.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%  RESIZE A %%%%%%%%%%%%%%%%%%%%%%
        self.A.resize(ngrp,self.n)
        self.A     = self.A.tocsr()
        sA1,sA2    = self.A.shape
        self.Ashape = [ sA1, sA2 ]
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass = "SBR2-MN-30-0"

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gSQUARE(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= GVAR_*GVAR_
        if nargout>1:
            g_ = GVAR_+GVAR_
            if nargout>2:
                H_ = np.zeros((1,1))
                H_ = 2.0
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

