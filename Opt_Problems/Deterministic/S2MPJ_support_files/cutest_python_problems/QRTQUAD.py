from Opt_Problems.Deterministic.S2MPJ_support_files.s2mpjlib import *
class  QRTQUAD(CUTEst_problem):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : QRTQUAD
#    *********
# 
#    An unconstrained problem involving elements with different degree of
#    nonlinearity (quartics and quadratics).  It contains N elements the first
#    M of which are quartic.  The Hessian is tridiagonal for the first M
#    elements and arrow-head for the remaining ones.  The first M variables
#    have lower and upper bounds.
# 
#    SIF input: Ph. Toint, June 1992.
#               minor correction by Ph. Shott, Jan 1995.
#               correction by S. Gratton & Ph. Toint, May 2024
# 
#    classification = "OBR2-MN-V-0"
# 
#           Alternative values for the SIF file parameters:
# IE N                   12             $-PARAMETER
# IE N                   120            $-PARAMETER     original value
# IE N                   1200           $-PARAMETER
# IE N                   5000           $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'QRTQUAD'

    def __init__(self, *args): 
        import numpy as np
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(12);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE M                   6              $-PARAMETER
# IE M                   10             $-PARAMETER     original value
# IE M                   100            $-PARAMETER
# IE M                   1100           $-PARAMETER
        if nargin<2:
            v_['M'] = int(100);  #  SIF file default value
        else:
            v_['M'] = int(args[1])
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['M+1'] = 1+v_['M']
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['C'] = - 10.0*v_['RI']
            [ig,ig_,_] = s2mpj_ii('OBJ',ig_)
            gtype = arrset(gtype,ig,'<>')
            iv = ix_['X'+str(I)]
            self.A[ig,iv] = float(v_['C'])+self.A[ig,iv]
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = np.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = np.zeros((ngrp,1))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = np.full((self.n,1),-float('Inf'))
        self.xupper = np.full((self.n,1),+float('Inf'))
        self.xlower = np.zeros((self.n,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.xlower[ix_['X'+str(I)]] = 0.0
            self.xupper[ix_['X'+str(I)]] = 10.0
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.zeros((self.n,1))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = s2mpj_ii( 'eQUART', iet_)
        elftv = loaset(elftv,it,0,'X')
        elftv = loaset(elftv,it,1,'Y')
        elftp = []
        elftp = loaset(elftp,it,0,'P')
        [it,iet_,_] = s2mpj_ii( 'eQUAD', iet_)
        elftv = loaset(elftv,it,0,'X')
        elftv = loaset(elftv,it,1,'Y')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = np.array([])
        ielftype     = np.array([])
        self.elvar   = []
        self.elpar   = []
        v_['RM'] = float(v_['M'])
        for I in range(int(v_['1']),int(v_['M'])+1):
            v_['RI'] = float(I)
            v_['C'] = v_['RI']/v_['RM']
            v_['I+1'] = I+v_['1']
            ename = 'E'+str(I)
            [ie,ie_,newelt] = s2mpj_ii(ename,ie_)
            self.elftype = arrset(self.elftype,ie,'eQUART')
            ielftype = arrset(ielftype, ie, iet_["eQUART"])
            vname = 'X'+str(I)
            [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = np.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['I+1']))
            [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = np.where(elftv[ielftype[ie]]=='Y')[0]
            self.elvar = loaset(self.elvar,ie,posev[0],iv)
            posep = np.where(elftp[ielftype[ie]]=='P')[0]
            self.elpar = loaset(self.elpar,ie,posep[0],float(v_['C']))
        for I in range(int(v_['M+1']),int(v_['N-1'])+1):
            ename = 'E'+str(I)
            [ie,ie_,newelt] = s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = arrset(self.elftype,ie,'eQUAD')
                ielftype = arrset( ielftype,ie,iet_['eQUAD'])
            vname = 'X'+str(I)
            [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = np.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['N']))
            [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = np.where(elftv[ielftype[ie]]=='Y')[0]
            self.elvar = loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in np.arange(0,ngrp):
            self.grelt.append(np.array([]))
        self.grftype = np.array([])
        self.grelw   = []
        nlc         = np.array([])
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            ig = ig_['OBJ']
            posel = len(self.grelt[ig])
            self.grelt = loaset(self.grelt,ig,posel,ie_['E'+str(I)])
            self.grelw = loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%%%%%%  RESIZE A %%%%%%%%%%%%%%%%%%%%%%
        self.A.resize(ngrp,self.n)
        self.A     = self.A.tocsr()
        sA1,sA2    = self.A.shape
        self.Ashape = [ sA1, sA2 ]
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass = "OBR2-MN-V-0"
        self.x0        = np.zeros((self.n,1))
# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eQUART(self, nargout,*args):

        import numpy as np
        EV_  = args[0]
        iel_ = args[1]
        XY = EV_[0]*EV_[1]
        f_   = self.elpar[iel_][0]*XY**4
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = self.elpar[iel_][0]*EV_[1]*4.0*XY**3
            g_[1] = self.elpar[iel_][0]*EV_[0]*4.0*XY**3
            if nargout>2:
                H_ = np.zeros((2,2))
                H_[0,0] = 12.0*self.elpar[iel_][0]*(EV_[1]**2)*(XY**2)
                H_[1,1] = 12.0*self.elpar[iel_][0]*(EV_[0]**2)*(XY**2)
                H_[0,1]  = (
                      4.0*self.elpar[iel_][0]*(XY**3)+12.0*self.elpar[iel_][0]*EV_[1]*EV_[0]*(XY**2))
                H_[1,0] = H_[0,1]
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eQUAD(self, nargout,*args):

        import numpy as np
        EV_  = args[0]
        iel_ = args[1]
        f_   = 4.0*EV_[0]*EV_[0]+2.0*EV_[1]*EV_[1]+EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 8.0*EV_[0]+EV_[1]
            g_[1] = 4.0*EV_[1]+EV_[0]
            if nargout>2:
                H_ = np.zeros((2,2))
                H_[0,0] = 8.0
                H_[0,1] = 1.0
                H_[1,0] = H_[0,1]
                H_[1,1] = 4.0
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

