from Opt_Problems.Deterministic.S2MPJ_support_files.s2mpjlib import *
class  PORTSNQP(CUTEst_problem):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : PORTSNQP
#    *********
# 
#    A convex quadratic program of portfolio type
# 
#    The objective function is of the form
#       sum (i=1,n) ( x_i - ( 2 i - n ) / n )^2
#    There is a single equality constraint of the form
#       sum(i=1,n) x_i = 1
#    Finally, there are simple bounds
#          0 <= x_i   (i=1,n)
# 
#    SIF input: Nick Gould, June 2001
# 
#    classification = "QLR2-AN-V-1"
# 
#    The number of equality constraints
# 
# IE N                   10
# IE N                   100
# IE N                   1000
# IE N                   10000
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'PORTSNQP'

    def __init__(self, *args): 
        import numpy as np
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 100000
        v_['N'] = 10
        v_['1'] = 1
        v_['RN'] = float(v_['N'])
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['2I'] = 2.0*v_['RI']
            v_['2I-N'] = v_['2I']-v_['RN']
            v_['C'] = v_['2I-N']/v_['RN']
            [ig,ig_,_] = s2mpj_ii('O'+str(I),ig_)
            gtype = arrset(gtype,ig,'<>')
            iv = ix_['X'+str(I)]
            self.A[ig,iv] = float(1.0)+self.A[ig,iv]
            [ig,ig_,_] = s2mpj_ii('E',ig_)
            gtype = arrset(gtype,ig,'==')
            cnames = arrset(cnames,ig,'E')
            iv = ix_['X'+str(I)]
            self.A[ig,iv] = float(1.0)+self.A[ig,iv]
            [ig,ig_,_] = s2mpj_ii('E2',ig_)
            gtype = arrset(gtype,ig,'==')
            cnames = arrset(cnames,ig,'E2')
            iv = ix_['X'+str(I)]
            self.A[ig,iv] = float(v_['C'])+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('O',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['X'+str(int(v_['N']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(-0.5))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        legrps = np.where(gtype=='<=')[0]
        eqgrps = np.where(gtype=='==')[0]
        gegrps = np.where(gtype=='>=')[0]
        self.nle = len(legrps)
        self.neq = len(eqgrps)
        self.nge = len(gegrps)
        self.m   = self.nle+self.neq+self.nge
        self.congrps = np.concatenate((legrps,eqgrps,gegrps))
        self.cnames= cnames[self.congrps]
        self.nob = ngrp-self.m
        self.objgrps = np.where(gtype=='<>')[0]
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = np.zeros((ngrp,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['2I'] = 2.0*v_['RI']
            v_['2I-N'] = v_['2I']-v_['RN']
            v_['C'] = v_['2I-N']/v_['RN']
            self.gconst = arrset(self.gconst,ig_['O'+str(I)],float(v_['C']))
        self.gconst = arrset(self.gconst,ig_['E'],float(1.0))
        self.gconst = arrset(self.gconst,ig_['E2'],float(0.5))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = np.zeros((self.n,1))
        self.xupper = np.full((self.n,1),float('inf'))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.xlower[ix_['X'+str(I)]] = 0.0
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.full((self.n,1),float(0.5))
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['O'+str(I)]
            self.grftype = arrset(self.grftype,ig,'gL2')
        ig = ig_['O']
        self.grftype = arrset(self.grftype,ig,'gL2')
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = np.full((self.m,1),-float('Inf'))
        self.cupper = np.full((self.m,1),+float('Inf'))
        self.clower[np.arange(self.nle,self.nle+self.neq)] = np.zeros((self.neq,1))
        self.cupper[np.arange(self.nle,self.nle+self.neq)] = np.zeros((self.neq,1))
        #%%%%%%%%%%%%%%%%%  RESIZE A %%%%%%%%%%%%%%%%%%%%%%
        self.A.resize(ngrp,self.n)
        self.A     = self.A.tocsr()
        sA1,sA2    = self.A.shape
        self.Ashape = [ sA1, sA2 ]
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons   = np.arange(len(self.congrps))
        self.pbclass = "QLR2-AN-V-1"
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

