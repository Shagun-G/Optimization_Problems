from Opt_Problems.Deterministic.S2MPJ_support_files.s2mpjlib import *
class  LUKVLE9(CUTEst_problem):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : LUKVLE9
#    *********
# 
#    Source: Problem 5.9, the modified Brown function with
#    simplified seven-diagonal constraints, due to L. Luksan and J. Vlcek,
#    "Sparse and partially separable test problems for 
#    unconstrained and equality constrained optimization",
#    Technical Report 767, Inst. Computer Science, Academy of Sciences
#    of the Czech Republic, 182 07 Prague, Czech Republic, 1999
# 
#    SIF input: Nick Gould, April 2001
# 
#    classification = "OOR2-AY-V-V"
# 
#    some useful parameters, including N, the number of variables.
# 
#           Alternative values for the SIF file parameters:
# IE N                   100            $-PARAMETER
# IE N                   1000           $-PARAMETER
# IE N                   10000          $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LUKVLE9'

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
# IE N                   100000         $-PARAMETER
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['4'] = 4
        v_['5'] = 5
        v_['6'] = 6
        v_['N/2'] = int(np.fix(v_['N']/v_['2']))
        v_['N-1'] = -1+v_['N']
        v_['N-2'] = -2+v_['N']
        v_['N-3'] = -3+v_['N']
        v_['N-4'] = -4+v_['N']
        v_['N-5'] = -5+v_['N']
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
        for I in range(int(v_['1']),int(v_['N/2'])+1):
            v_['2I'] = 2*I
            v_['2I-1'] = -1+v_['2I']
            [ig,ig_,_] = s2mpj_ii('OBJ1'+str(I),ig_)
            gtype = arrset(gtype,ig,'<>')
            iv = ix_['X'+str(int(v_['2I-1']))]
            self.A[ig,iv] = float(1.0)+self.A[ig,iv]
            [ig,ig_,_] = s2mpj_ii('OBJ2',ig_)
            gtype = arrset(gtype,ig,'<>')
            iv = ix_['X'+str(int(v_['2I-1']))]
            self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
            iv = ix_['X'+str(int(v_['2I']))]
            self.A[ig,iv] = float(1.0)+self.A[ig,iv]
            [ig,ig_,_] = s2mpj_ii('OBJ3'+str(I),ig_)
            gtype = arrset(gtype,ig,'<>')
            iv = ix_['X'+str(int(v_['2I-1']))]
            self.A[ig,iv] = float(1.0)+self.A[ig,iv]
            iv = ix_['X'+str(int(v_['2I']))]
            self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['1'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['1'])))
        iv = ix_['X'+str(int(v_['1']))]
        self.A[ig,iv] = float(4.0)+self.A[ig,iv]
        iv = ix_['X'+str(int(v_['2']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['1'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['1'])))
        iv = ix_['X'+str(int(v_['3']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['2'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['2'])))
        iv = ix_['X'+str(int(v_['2']))]
        self.A[ig,iv] = float(6.0)+self.A[ig,iv]
        iv = ix_['X'+str(int(v_['3']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['2'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['2'])))
        iv = ix_['X'+str(int(v_['4']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['3'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['3'])))
        iv = ix_['X'+str(int(v_['3']))]
        self.A[ig,iv] = float(6.0)+self.A[ig,iv]
        iv = ix_['X'+str(int(v_['4']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['3'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['3'])))
        iv = ix_['X'+str(int(v_['5']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        iv = ix_['X'+str(int(v_['1']))]
        self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['4'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['4'])))
        iv = ix_['X'+str(int(v_['N-2']))]
        self.A[ig,iv] = float(6.0)+self.A[ig,iv]
        iv = ix_['X'+str(int(v_['N-1']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['4'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['4'])))
        iv = ix_['X'+str(int(v_['N']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        iv = ix_['X'+str(int(v_['N-4']))]
        self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['4'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['4'])))
        iv = ix_['X'+str(int(v_['N-5']))]
        self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['5'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['5'])))
        iv = ix_['X'+str(int(v_['N-1']))]
        self.A[ig,iv] = float(6.0)+self.A[ig,iv]
        iv = ix_['X'+str(int(v_['N-3']))]
        self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['5'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['5'])))
        iv = ix_['X'+str(int(v_['N']))]
        self.A[ig,iv] = float(1.0)+self.A[ig,iv]
        iv = ix_['X'+str(int(v_['N-4']))]
        self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['6'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['6'])))
        iv = ix_['X'+str(int(v_['N']))]
        self.A[ig,iv] = float(2.0)+self.A[ig,iv]
        iv = ix_['X'+str(int(v_['N-3']))]
        self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
        [ig,ig_,_] = s2mpj_ii('C'+str(int(v_['6'])),ig_)
        gtype = arrset(gtype,ig,'==')
        cnames = arrset(cnames,ig,'C'+str(int(v_['6'])))
        iv = ix_['X'+str(int(v_['N-2']))]
        self.A[ig,iv] = float(-1.0)+self.A[ig,iv]
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
        self.gconst = arrset(self.gconst,ig_['C'+str(int(v_['2']))],float(2.0))
        self.gconst = arrset(self.gconst,ig_['C'+str(int(v_['3']))],float(2.0))
        self.gconst = arrset(self.gconst,ig_['C'+str(int(v_['4']))],float(2.0))
        self.gconst = arrset(self.gconst,ig_['C'+str(int(v_['5']))],float(2.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = np.full((self.n,1),-float('Inf'))
        self.xupper = np.full((self.n,1),+float('Inf'))
        self.xlower = np.zeros((self.n,1))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.zeros((self.n,1))
        self.y0 = np.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.x0[ix_['X'+str(I)]] = float(-1.0)
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = s2mpj_ii( 'eSQR', iet_)
        elftv = loaset(elftv,it,0,'V')
        [it,iet_,_] = s2mpj_ii( 'eCUBEP', iet_)
        elftv = loaset(elftv,it,0,'V')
        elftv = loaset(elftv,it,1,'W')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = np.array([])
        ielftype     = np.array([])
        self.elvar   = []
        ename = 'C1'+str(int(v_['1']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C1'+str(int(v_['1']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['2']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C2'+str(int(v_['1']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C2'+str(int(v_['1']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['3']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C3'+str(int(v_['1']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C3'+str(int(v_['1']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['4']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C1'+str(int(v_['2']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eCUBEP')
        ielftype = arrset(ielftype, ie, iet_["eCUBEP"])
        ename = 'C1'+str(int(v_['2']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['2']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C1'+str(int(v_['2']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['1']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C2'+str(int(v_['2']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C2'+str(int(v_['2']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['3']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C3'+str(int(v_['2']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C3'+str(int(v_['2']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['1']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C4'+str(int(v_['2']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C4'+str(int(v_['2']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['4']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C5'+str(int(v_['2']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C5'+str(int(v_['2']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['5']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C1'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eCUBEP')
        ielftype = arrset(ielftype, ie, iet_["eCUBEP"])
        ename = 'C1'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['3']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C1'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['2']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C2'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C2'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['4']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C3'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C3'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['2']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C4'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C4'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['5']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C5'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C5'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['1']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C6'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C6'+str(int(v_['3']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['6']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C1'+str(int(v_['4']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eCUBEP')
        ielftype = arrset(ielftype, ie, iet_["eCUBEP"])
        ename = 'C1'+str(int(v_['4']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-2']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C1'+str(int(v_['4']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-3']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C2'+str(int(v_['4']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C2'+str(int(v_['4']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-1']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C3'+str(int(v_['4']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C3'+str(int(v_['4']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-3']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C4'+str(int(v_['4']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C4'+str(int(v_['4']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C5'+str(int(v_['4']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C5'+str(int(v_['4']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-4']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C1'+str(int(v_['5']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eCUBEP')
        ielftype = arrset(ielftype, ie, iet_["eCUBEP"])
        ename = 'C1'+str(int(v_['5']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-1']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C1'+str(int(v_['5']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-2']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C2'+str(int(v_['5']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C2'+str(int(v_['5']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C3'+str(int(v_['5']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C3'+str(int(v_['5']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-2']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C4'+str(int(v_['5']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C4'+str(int(v_['5']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-3']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C1'+str(int(v_['6']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eCUBEP')
        ielftype = arrset(ielftype, ie, iet_["eCUBEP"])
        ename = 'C1'+str(int(v_['6']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C1'+str(int(v_['6']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-1']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C2'+str(int(v_['6']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C2'+str(int(v_['6']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-1']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        ename = 'C3'+str(int(v_['6']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        self.elftype = arrset(self.elftype,ie,'eSQR')
        ielftype = arrset(ielftype, ie, iet_["eSQR"])
        ename = 'C3'+str(int(v_['6']))
        [ie,ie_,_] = s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['N-2']))
        [iv,ix_] = s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = np.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = s2mpj_ii('gAL2',igt_)
        [it,igt_,_] = s2mpj_ii('gEXP20',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in np.arange(0,ngrp):
            self.grelt.append(np.array([]))
        self.grftype = np.array([])
        self.grelw   = []
        nlc         = np.array([])
        for I in range(int(v_['1']),int(v_['N/2'])+1):
            ig = ig_['OBJ1'+str(I)]
            self.grftype = arrset(self.grftype,ig,'gAL2')
            ig = ig_['OBJ3'+str(I)]
            self.grftype = arrset(self.grftype,ig,'gEXP20')
        ig = ig_['C'+str(int(v_['1']))]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C1'+str(int(v_['1']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-4.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C2'+str(int(v_['1']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-1.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C3'+str(int(v_['1']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['C'+str(int(v_['2']))]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C1'+str(int(v_['2']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(8.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C2'+str(int(v_['2']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-4.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C3'+str(int(v_['2']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(1.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C4'+str(int(v_['2']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-1.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C5'+str(int(v_['2']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['C'+str(int(v_['3']))]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C1'+str(int(v_['3']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(8.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C2'+str(int(v_['3']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-4.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C3'+str(int(v_['3']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(1.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C4'+str(int(v_['3']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-1.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C5'+str(int(v_['3']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(1.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C6'+str(int(v_['3']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['C'+str(int(v_['4']))]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C1'+str(int(v_['4']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(8.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C2'+str(int(v_['4']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-4.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C3'+str(int(v_['4']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(1.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C4'+str(int(v_['4']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-1.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C5'+str(int(v_['4']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(1.0))
        ig = ig_['C'+str(int(v_['5']))]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C1'+str(int(v_['5']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(8.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C2'+str(int(v_['5']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(-4.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C3'+str(int(v_['5']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(1.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C4'+str(int(v_['5']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(1.0))
        ig = ig_['C'+str(int(v_['6']))]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C1'+str(int(v_['6']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(8.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C2'+str(int(v_['6']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(1.0))
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt,ig,posel,ie_['C3'+str(int(v_['6']))])
        nlc = np.union1d(nlc,np.array([ig]))
        self.grelw = loaset(self.grelw,ig,posel,float(1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN               1.0111E+2
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
        self.pbclass = "OOR2-AY-V-V"
# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eSQR(self, nargout,*args):

        import numpy as np
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 2.0*EV_[0]
            if nargout>2:
                H_ = np.zeros((1,1))
                H_[0,0] = 2.0
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eCUBEP(self, nargout,*args):

        import numpy as np
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]**3-EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = 3.0*EV_[0]**2-EV_[1]
            g_[1] = -EV_[0]
            if nargout>2:
                H_ = np.zeros((2,2))
                H_[0,0] = 6.0*EV_[0]
                H_[0,1] = -1.0
                H_[1,0] = H_[0,1]
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gAL2(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= 0.001*GVAR_*GVAR_
        if nargout>1:
            g_ = 0.002*GVAR_
            if nargout>2:
                H_ = np.zeros((1,1))
                H_ = 0.002
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def gEXP20(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        EXP20A = np.exp(20.0*GVAR_)
        f_= EXP20A
        if nargout>1:
            g_ = 20.0*EXP20A
            if nargout>2:
                H_ = np.zeros((1,1))
                H_ = 400.0*EXP20A
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

