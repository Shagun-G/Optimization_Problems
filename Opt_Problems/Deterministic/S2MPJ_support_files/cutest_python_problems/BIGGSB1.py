from Opt_Problems.Deterministic.S2MPJ_support_files.s2mpjlib import *
class  BIGGSB1(CUTEst_problem):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    A quadratic test problem with bounds on the variables
# 
#    Source:
#    M. Batholomew-Biggs and F.G. Hernandez,
#    "Some improvements to the subroutine OPALQP for dealing with large
#     problems",
#    Numerical Optimization Centre, Hatfield, 1992.
# 
#    SIF input: Ph Toint, April 1992.
# 
#    classification = "QBR2-AN-V-V"
# 
#    Number of variables
# 
#           Alternative values for the SIF file parameters:
# IE N                   25             $-PARAMETER     original value
# IE N                   100            $-PARAMETER
# IE N                   1000           $-PARAMETER
# IE N                   5000           $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'BIGGSB1'

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
# IE N                   20000          $-PARAMETER
        v_['0'] = 0
        v_['1'] = 1
        v_['N-1'] = -1+v_['N']
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
        [ig,ig_,_] = s2mpj_ii('G'+str(int(v_['0'])),ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['X'+str(int(v_['1']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            v_['I+1'] = 1+I
            [ig,ig_,_] = s2mpj_ii('G'+str(I),ig_)
            gtype = arrset(gtype,ig,'<>')
            iv = ix_['X'+str(int(v_['I+1']))]
            self.A[ig,iv] = float(1.0)+self.A[ig,iv]
            iv = ix_['X'+str(I)]
            self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('G'+str(int(v_['N'])),ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['X'+str(int(v_['N']))]
        self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = np.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = np.zeros((ngrp,1))
        self.gconst = arrset(self.gconst,ig_['G'+str(int(v_['0']))],float(1.0))
        self.gconst = arrset(self.gconst,ig_['G'+str(int(v_['N']))],float(-1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xupper = np.full((self.n,1),0.9)
        self.xlower = np.zeros((self.n,1))
        self.xlower[ix_['X'+str(int(v_['N']))]] = -float('Inf')
        self.xupper[ix_['X'+str(int(v_['N']))]] = +float('Inf')
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = s2mpj_ii('gL2',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in np.arange(0,ngrp):
            self.grelt.append(np.array([]))
        self.grftype = np.array([])
        self.grelw   = []
        nlc         = np.array([])
        for ig in range(0,ngrp):
            self.grftype = arrset(self.grftype,ig,'gL2')
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO BIGGSB1             0.015
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%%%%%%  RESIZE A %%%%%%%%%%%%%%%%%%%%%%
        self.A.resize(ngrp,self.n)
        self.A     = self.A.tocsr()
        sA1,sA2    = self.A.shape
        self.Ashape = [ sA1, sA2 ]
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass = "QBR2-AN-V-V"
        self.x0        = np.zeros((self.n,1))
# ********************
#  SET UP THE GROUPS *
#  ROUTINE           *
# ********************

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gL2(self,nargout,*args):

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

