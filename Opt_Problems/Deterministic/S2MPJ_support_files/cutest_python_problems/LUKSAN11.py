from Opt_Problems.Deterministic.S2MPJ_support_files.s2mpjlib import *
class  LUKSAN11(CUTEst_problem):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : LUKSAN11
#    *********
# 
#    Problem 11 (chained serpentine) in the paper
# 
#      L. Luksan
#      Hybrid methods in large sparse nonlinear least squares
#      J. Optimization Theory & Applications 89(3) 575-595 (1996)
# 
#    SIF input: Nick Gould, June 2017.
# 
#    classification = "NOR2-AN-V-V"
# 
#   seed for dimensions
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LUKSAN11'

    def __init__(self, *args): 
        import numpy as np
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['S'] = 99
        v_['N'] = v_['S']
        v_['N'] = 1+v_['N']
        v_['M'] = 2*v_['S']
        v_['1'] = 1
        v_['2'] = 2
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
        v_['K'] = 1
        for I in range(int(v_['1']),int(v_['S'])+1):
            v_['I+1'] = 1+I
            [ig,ig_,_] = s2mpj_ii('E'+str(int(v_['K'])),ig_)
            gtype = arrset(gtype,ig,'==')
            cnames = arrset(cnames,ig,'E'+str(int(v_['K'])))
            iv = ix_['X'+str(int(v_['I+1']))]
            self.A[ig,iv] = float(-10.0)+self.A[ig,iv]
            v_['K'] = 1+v_['K']
            [ig,ig_,_] = s2mpj_ii('E'+str(int(v_['K'])),ig_)
            gtype = arrset(gtype,ig,'==')
            cnames = arrset(cnames,ig,'E'+str(int(v_['K'])))
            iv = ix_['X'+str(I)]
            self.A[ig,iv] = float(1.0)+self.A[ig,iv]
            v_['K'] = 1+v_['K']
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
        for I in range(int(v_['1']),int(v_['M'])+1,int(v_['2'])):
            v_['I+1'] = 1+I
            self.gconst = arrset(self.gconst,ig_['E'+str(int(v_['I+1']))],float(1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = np.full((self.n,1),-float('Inf'))
        self.xupper = np.full((self.n,1),+float('Inf'))
        self.xlower = np.zeros((self.n,1))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.zeros((self.n,1))
        self.y0 = np.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.x0[ix_['X'+str(I)]] = float(-0.8)
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = s2mpj_ii( 'eL1', iet_)
        elftv = loaset(elftv,it,0,'V')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = np.array([])
        ielftype     = np.array([])
        self.elvar   = []
        v_['K'] = 1
        for I in range(int(v_['1']),int(v_['S'])+1):
            ename = 'L'+str(int(v_['K']))
            [ie,ie_,newelt] = s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = arrset(self.elftype,ie,'eL1')
                ielftype = arrset( ielftype,ie,iet_['eL1'])
            vname = 'X'+str(I)
            [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = np.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = loaset(self.elvar,ie,posev[0],iv)
            v_['K'] = 2+v_['K']
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in np.arange(0,ngrp):
            self.grelt.append(np.array([]))
        self.grftype = np.array([])
        self.grelw   = []
        nlc         = np.array([])
        v_['K'] = 1
        for I in range(int(v_['1']),int(v_['S'])+1):
            ig = ig_['E'+str(int(v_['K']))]
            posel = len(self.grelt[ig])
            self.grelt = loaset(self.grelt,ig,posel,ie_['L'+str(int(v_['K']))])
            nlc = np.union1d(nlc,np.array([ig]))
            self.grelw = loaset(self.grelw,ig,posel,1.)
            v_['K'] = 2+v_['K']
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN               0.0
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
        self.lincons =  np.where(self.congrps in np.setdiff1d(nlc,self.congrps))[0]
        self.pbclass = "NOR2-AN-V-V"
# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eL1(self, nargout,*args):

        import numpy as np
        EV_  = args[0]
        iel_ = args[1]
        D = 1.0e0+EV_[0]*EV_[0]
        f_   = 2.0e1*EV_[0]/D
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 2.0e1*(1.0e0-EV_[0]*EV_[0])/D**2
            if nargout>2:
                H_ = np.zeros((1,1))
                H_[0,0] = -4.0e1*EV_[0]*(3.0e0-EV_[0]*EV_[0])/D**3
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

