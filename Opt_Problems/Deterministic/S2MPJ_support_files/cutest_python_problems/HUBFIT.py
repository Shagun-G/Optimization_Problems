from Opt_Problems.Deterministic.S2MPJ_support_files.s2mpjlib import *
class  HUBFIT(CUTEst_problem):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HUBFIT
#    *********
#    Variable dimension full rank linear problem
#    An elementary fit using the Huber loss function
# 
#    Source:
#    A.R. Conn, N. Gould and Ph.L. Toint,
#    "The LANCELOT User's Manual",
#    Dept of Maths, FUNDP, 1991.
# 
#    SIF input: Ph. Toint, Jan 1991.
# 
#    classification = "OLR2-AN-2-1"
# 
#    Data points
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HUBFIT'

    def __init__(self, *args): 
        import numpy as np
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['X1'] = 0.1
        v_['X2'] = 0.3
        v_['X3'] = 0.5
        v_['X4'] = 0.7
        v_['X5'] = 0.9
        v_['Y1'] = 0.25
        v_['Y2'] = 0.3
        v_['Y3'] = 0.625
        v_['Y4'] = 0.701
        v_['Y5'] = 1.0
        v_['C'] = 0.85
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = np.array([])
        self.xscale = np.array([])
        intvars   = np.array([])
        binvars   = np.array([])
        [iv,ix_,_] = s2mpj_ii('a',ix_)
        self.xnames=arrset(self.xnames,iv,'a')
        [iv,ix_,_] = s2mpj_ii('b',ix_)
        self.xnames=arrset(self.xnames,iv,'b')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.A       = lil_matrix((1000000,1000000))
        self.gscale  = np.array([])
        self.grnames = np.array([])
        cnames      = np.array([])
        self.cnames = np.array([])
        gtype       = np.array([])
        [ig,ig_,_] = s2mpj_ii('Obj1',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['a']
        self.A[ig,iv] = float(v_['X1'])+self.A[ig,iv]
        iv = ix_['b']
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(2.0))
        [ig,ig_,_] = s2mpj_ii('Obj2',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['a']
        self.A[ig,iv] = float(v_['X2'])+self.A[ig,iv]
        iv = ix_['b']
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(2.0))
        [ig,ig_,_] = s2mpj_ii('Obj3',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['a']
        self.A[ig,iv] = float(v_['X3'])+self.A[ig,iv]
        iv = ix_['b']
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(2.0))
        [ig,ig_,_] = s2mpj_ii('Obj4',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['a']
        self.A[ig,iv] = float(v_['X4'])+self.A[ig,iv]
        iv = ix_['b']
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(2.0))
        [ig,ig_,_] = s2mpj_ii('Obj5',ig_)
        gtype = arrset(gtype,ig,'<>')
        iv = ix_['a']
        self.A[ig,iv] = float(v_['X5'])+self.A[ig,iv]
        iv = ix_['b']
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        self.gscale = arrset(self.gscale,ig,float(2.0))
        [ig,ig_,_] = s2mpj_ii('Cons',ig_)
        gtype = arrset(gtype,ig,'<=')
        cnames = arrset(cnames,ig,'Cons')
        iv = ix_['a']
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        iv = ix_['b']
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
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
        self.gconst = arrset(self.gconst,ig_['Obj1'],float(v_['Y1']))
        self.gconst = arrset(self.gconst,ig_['Obj2'],float(v_['Y2']))
        self.gconst = arrset(self.gconst,ig_['Obj3'],float(v_['Y3']))
        self.gconst = arrset(self.gconst,ig_['Obj4'],float(v_['Y4']))
        self.gconst = arrset(self.gconst,ig_['Obj5'],float(v_['Y5']))
        self.gconst = arrset(self.gconst,ig_['Cons'],float(v_['C']))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = np.zeros((self.n,1))
        self.xupper = np.full((self.n,1),float('inf'))
        self.xlower[ix_['b']] = -float('Inf')
        self.xupper[ix_['b']] = +float('Inf')
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = s2mpj_ii('gHUBER',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in np.arange(0,ngrp):
            self.grelt.append(np.array([]))
        self.grftype = np.array([])
        self.grelw   = []
        nlc         = np.array([])
        ig = ig_['Obj1']
        self.grftype = arrset(self.grftype,ig,'gHUBER')
        ig = ig_['Obj2']
        self.grftype = arrset(self.grftype,ig,'gHUBER')
        ig = ig_['Obj3']
        self.grftype = arrset(self.grftype,ig,'gHUBER')
        ig = ig_['Obj4']
        self.grftype = arrset(self.grftype,ig,'gHUBER')
        ig = ig_['Obj5']
        self.grftype = arrset(self.grftype,ig,'gHUBER')
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = np.full((self.m,1),-float('Inf'))
        self.cupper = np.full((self.m,1),+float('Inf'))
        self.cupper[np.arange(self.nle)] = np.zeros((self.nle,1))
        #%%%%%%%%%%%%%%%%%  RESIZE A %%%%%%%%%%%%%%%%%%%%%%
        self.A.resize(ngrp,self.n)
        self.A     = self.A.tocsr()
        sA1,sA2    = self.A.shape
        self.Ashape = [ sA1, sA2 ]
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons   = np.arange(len(self.congrps))
        self.pbclass = "OLR2-AN-2-1"
        self.x0        = np.zeros((self.n,1))

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gHUBER(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        HUBERK = 1.5
        ABSA = np.absolute(GVAR_)
        OUT = ABSA>HUBERK
        if OUT!=0:
            FF = HUBERK*ABSA-0.5*HUBERK*HUBERK
        NEGOUT = OUT and (GVAR_<0.0)
        POSOUT = OUT and (GVAR_>=0.0)
        if POSOUT!=0:
            GG = HUBERK
        if NEGOUT!=0:
            GG = -HUBERK
        if OUT!=0:
            HH = 0.0
        if OUT==0:
            FF = 0.5*ABSA*ABSA
        if OUT==0:
            GG = GVAR_
        if OUT==0:
            HH = 1.0
        f_= FF
        if nargout>1:
            g_ = GG
            if nargout>2:
                H_ = np.zeros((1,1))
                H_ = HH
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

