from Opt_Problems.Deterministic.S2MPJ_support_files.s2mpjlib import *
class  CHENHARK(CUTEst_problem):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : CHENHARK
#    --------
# 
#    A bound-constrained version the Linear Complementarity problem
# 
#    Find x such that w = M x + q, x and w nonnegative and x^T w = 0,
#    where
# 
#    M = (  6   -4   1   0  ........ 0 ) 
#        ( -4    6  -4   1  ........ 0 )
#        (  1   -4   6  -4  ........ 0 )
#        (  0    1  -4   6  ........ 0 )  
#           ..........................
#        (  0   ........... 0  1 -4  6 )
# 
#    and q is given.
# 
#    Source: 
#    B. Chen and P. T. Harker,
#    SIMAX 14 (1993) 1168-1190
# 
#    SDIF input: Nick Gould, November 1993.
# 
#    classification = "QBR2-AN-V-V"
# 
#    Number of variables
# 
#           Alternative values for the SIF file parameters:
# IE N                   10             $-PARAMETER
# IE N                   100            $-PARAMETER
# IE N                   1000           $-PARAMETER     original value
# IE N                   5000           $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'CHENHARK'

    def __init__(self, *args): 
        import numpy as np
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(10);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE N                   10000          $-PARAMETER
# IE N                   50000          $-PARAMETER
# IE NFREE               5              $-PARAMETER
# IE NFREE               50             $-PARAMETER
# IE NFREE               500            $-PARAMETER     original value
# IE NFREE               2500           $-PARAMETER
        if nargin<2:
            v_['NFREE'] = int(5);  #  SIF file default value
        else:
            v_['NFREE'] = int(args[1])
# IE NFREE               5000           $-PARAMETER
# IE NFREE               10000          $-PARAMETER
# IE NDEGEN              2              $-PARAMETER
# IE NDEGEN              20             $-PARAMETER
# IE NDEGEN              200            $-PARAMETER     original value
# IE NDEGEN              500            $-PARAMETER
        if nargin<3:
            v_['NDEGEN'] = int(2);  #  SIF file default value
        else:
            v_['NDEGEN'] = int(args[2])
# IE NDEGEN              1000           $-PARAMETER
# IE NDEGEN              2000           $-PARAMETER
        v_['-1'] = -1
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['N-1'] = -1+v_['N']
        v_['N+1'] = 1+v_['N']
        v_['N+2'] = 2+v_['N']
        v_['NFREE+1'] = 1+v_['NFREE']
        v_['NF+ND'] = v_['NFREE']+v_['NDEGEN']
        v_['NF+ND+1'] = 1+v_['NF+ND']
        v_['X'+str(int(v_['-1']))] = 0.0
        v_['X'+str(int(v_['0']))] = 0.0
        for I in range(int(v_['1']),int(v_['NFREE'])+1):
            v_['X'+str(I)] = 1.0
        for I in range(int(v_['NFREE+1']),int(v_['N+2'])+1):
            v_['X'+str(I)] = 0.0
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = np.array([])
        self.xscale = np.array([])
        intvars   = np.array([])
        binvars   = np.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = s2mpj_ii('X'+str(I),ix_)
            self.xnames=arrset(self.xnames,iv,'X'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.A       = lil_matrix((1000000,1000000))
        self.gscale  = np.array([])
        self.grnames = np.array([])
        cnames      = np.array([])
        self.cnames = np.array([])
        gtype       = np.array([])
        for I in range(int(v_['2']),int(v_['N-1'])+1):
            v_['I+1'] = 1+I
            v_['I-1'] = -1+I
            [ig,ig_,_] = s2mpj_ii('Q'+str(I),ig_)
            gtype = arrset(gtype,ig,'<>')
            iv = ix_['X'+str(int(v_['I+1']))]
            self.A[ig,iv] = float(1.0)+self.A[ig,iv]
            iv = ix_['X'+str(int(v_['I-1']))]
            self.A[ig,iv] = float(1.0)+self.A[ig,iv]
            iv = ix_['X'+str(I)]
            self.A[ig,iv] = float(-2.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('Q'+str(int(v_['0'])),ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['X'+str(int(v_['1']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('Q'+str(int(v_['1'])),ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['X'+str(int(v_['1']))]
        self.A[ig,iv] = float(2.0)+self.A[ig,iv]
        iv = ix_['X'+str(int(v_['2']))]
        self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('Q'+str(int(v_['N'])),ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['X'+str(int(v_['N']))]
        self.A[ig,iv] = float(2.0)+self.A[ig,iv]
        iv = ix_['X'+str(int(v_['N-1']))]
        self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('Q'+str(int(v_['N+1'])),ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['X'+str(int(v_['N']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        for I in range(int(v_['1']),int(v_['NF+ND'])+1):
            v_['I+1'] = 1+I
            v_['I+2'] = 2+I
            v_['I-1'] = -1+I
            v_['I-2'] = -2+I
            v_['Q1'] = -6.0*v_['X'+str(I)]
            v_['Q2'] = 4.0*v_['X'+str(int(v_['I+1']))]
            v_['Q3'] = 4.0*v_['X'+str(int(v_['I-1']))]
            v_['Q4'] = -1.0*v_['X'+str(int(v_['I+2']))]
            v_['Q5'] = -1.0*v_['X'+str(int(v_['I-2']))]
            v_['Q'] = v_['Q1']+v_['Q2']
            v_['Q'] = v_['Q']+v_['Q3']
            v_['Q'] = v_['Q']+v_['Q4']
            v_['Q'] = v_['Q']+v_['Q5']
            [ig,ig_,_] = s2mpj_ii('L',ig_)
            gtype = arrset(gtype,ig,'<>')
            iv = ix_['X'+str(I)]
            self.A[ig,iv] = float(v_['Q'])+self.A[ig,iv]
        for I in range(int(v_['NF+ND+1']),int(v_['N'])+1):
            v_['I+1'] = 1+I
            v_['I+2'] = 2+I
            v_['I-1'] = -1+I
            v_['I-2'] = -2+I
            v_['Q1'] = -6.0*v_['X'+str(I)]
            v_['Q2'] = 4.0*v_['X'+str(int(v_['I+1']))]
            v_['Q3'] = 4.0*v_['X'+str(int(v_['I-1']))]
            v_['Q4'] = -1.0*v_['X'+str(int(v_['I+2']))]
            v_['Q5'] = -1.0*v_['X'+str(int(v_['I-2']))]
            v_['Q'] = v_['Q1']+v_['Q2']
            v_['Q'] = v_['Q']+v_['Q3']
            v_['Q'] = v_['Q']+v_['Q4']
            v_['Q'] = v_['Q']+v_['Q5']
            v_['Q'] = 1.0+v_['Q']
            [ig,ig_,_] = s2mpj_ii('L',ig_)
            gtype = arrset(gtype,ig,'<>')
            iv = ix_['X'+str(I)]
            self.A[ig,iv] = float(v_['Q'])+self.A[ig,iv]
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = np.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.zeros((self.n,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.x0[ix_['X'+str(I)]] = float(0.5)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = s2mpj_ii('gHALFL2',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in np.arange(0,ngrp):
            self.grelt.append(np.array([]))
        self.grftype = np.array([])
        self.grelw   = []
        nlc         = np.array([])
        for I in range(int(v_['0']),int(v_['N+1'])+1):
            ig = ig_['Q'+str(I)]
            self.grftype = arrset(self.grftype,ig,'gHALFL2')
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 1.0
#    Solution
# LO SOLTN               -0.5
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        self.xlower = np.zeros((self.n,1))
        self.xupper = np.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%  RESIZE A %%%%%%%%%%%%%%%%%%%%%%
        self.A.resize(ngrp,self.n)
        self.A     = self.A.tocsr()
        sA1,sA2    = self.A.shape
        self.Ashape = [ sA1, sA2 ]
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass = "QBR2-AN-V-V"
# ********************
#  SET UP THE GROUPS *
#  ROUTINE           *
# ********************

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gHALFL2(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= 5.0e-1*GVAR_*GVAR_
        if nargout>1:
            g_ = GVAR_
            if nargout>2:
                H_ = np.zeros((1,1))
                H_ = 1.0e+0
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

