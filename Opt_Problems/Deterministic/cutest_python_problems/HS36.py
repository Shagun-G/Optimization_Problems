from Opt_Problems.Deterministic.s2mpjlib import *


class HS36(CUTEst_problem):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #
    #    Problem : HS36
    #    *********
    #
    #    Source: problem 36 in
    #    W. Hock and K. Schittkowski,
    #    "Test examples for nonlinear programming codes",
    #    Lectures Notes in Economics and Mathematical Systems 187, Springer
    #    Verlag, Heidelberg, 1981.
    #
    #    SIF input: A.R. Conn, April 1990
    #
    #    classification = "C-COLR2-AN-3-1"
    #
    #
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #   Translated to Python by S2MPJ version 25 XI 2024
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = "HS36"

    def __init__(self, *args):
        import numpy as np
        from scipy.sparse import csr_matrix

        nargin = len(args)

        # %%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_ = {}
        ix_ = {}
        ig_ = {}
        # %%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = np.array([])
        self.xscale = np.array([])
        intvars = np.array([])
        binvars = np.array([])
        irA = np.array([], dtype=int)
        icA = np.array([], dtype=int)
        valA = np.array([], dtype=float)
        [iv, ix_, _] = s2mpj_ii("X1", ix_)
        self.xnames = arrset(self.xnames, iv, "X1")
        [iv, ix_, _] = s2mpj_ii("X2", ix_)
        self.xnames = arrset(self.xnames, iv, "X2")
        [iv, ix_, _] = s2mpj_ii("X3", ix_)
        self.xnames = arrset(self.xnames, iv, "X3")
        # %%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale = np.array([])
        self.grnames = np.array([])
        cnames = np.array([])
        self.cnames = np.array([])
        gtype = np.array([])
        [ig, ig_, _] = s2mpj_ii("OBJ", ig_)
        gtype = arrset(gtype, ig, "<>")
        [ig, ig_, _] = s2mpj_ii("CON1", ig_)
        gtype = arrset(gtype, ig, ">=")
        cnames = arrset(cnames, ig, "CON1")
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["X1"]])
        valA = np.append(valA, float(-1.0))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["X2"]])
        valA = np.append(valA, float(-2.0))
        irA = np.append(irA, [ig])
        icA = np.append(icA, [ix_["X3"]])
        valA = np.append(valA, float(-2.0))
        # %%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n = len(ix_)
        ngrp = len(ig_)
        legrps = np.where(gtype == "<=")[0]
        eqgrps = np.where(gtype == "==")[0]
        gegrps = np.where(gtype == ">=")[0]
        self.nle = len(legrps)
        self.neq = len(eqgrps)
        self.nge = len(gegrps)
        self.m = self.nle + self.neq + self.nge
        self.congrps = np.concatenate((legrps, eqgrps, gegrps))
        self.cnames = cnames[self.congrps]
        self.nob = ngrp - self.m
        self.objgrps = np.where(gtype == "<>")[0]
        # %%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = np.zeros((ngrp, 1))
        self.gconst = arrset(self.gconst, ig_["CON1"], float(-72.0))
        # %%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = np.zeros((self.n, 1))
        self.xupper = np.full((self.n, 1), float("inf"))
        self.xupper[ix_["X1"]] = 20.0
        self.xupper[ix_["X2"]] = 11.0
        self.xupper[ix_["X3"]] = 42.0
        # %%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = np.full((self.n, 1), float(10.0))
        # %%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_ = {}
        elftv = []
        [it, iet_, _] = s2mpj_ii("ePROD3", iet_)
        elftv = loaset(elftv, it, 0, "V1")
        elftv = loaset(elftv, it, 1, "V2")
        elftv = loaset(elftv, it, 2, "V3")
        # %%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = np.array([])
        ielftype = np.array([])
        self.elvar = []
        ename = "E1"
        [ie, ie_, _] = s2mpj_ii(ename, ie_)
        self.elftype = arrset(self.elftype, ie, "ePROD3")
        ielftype = arrset(ielftype, ie, iet_["ePROD3"])
        vname = "X1"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, float(10.0))
        posev = np.where(elftv[ielftype[ie]] == "V1")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "X2"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, float(10.0))
        posev = np.where(elftv[ielftype[ie]] == "V2")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        vname = "X3"
        [iv, ix_] = s2mpj_nlx(self, vname, ix_, 1, None, None, float(10.0))
        posev = np.where(elftv[ielftype[ie]] == "V3")[0]
        self.elvar = loaset(self.elvar, ie, posev[0], iv)
        # %%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt = []
        for ig in np.arange(0, ngrp):
            self.grelt.append(np.array([]))
        self.grftype = np.array([])
        self.grelw = []
        nlc = np.array([])
        ig = ig_["OBJ"]
        posel = len(self.grelt[ig])
        self.grelt = loaset(self.grelt, ig, posel, ie_["E1"])
        nlc = np.union1d(nlc, np.array([ig]))
        self.grelw = loaset(self.grelw, ig, posel, 1.0)
        # %%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        #    Solution
        # LO SOLTN               -3300.0
        # %%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = csr_matrix((valA, (irA, icA)), shape=(ngrp, self.n))
        # %%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        # %%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = np.full((self.m, 1), -float("Inf"))
        self.cupper = np.full((self.m, 1), +float("Inf"))
        self.clower[np.arange(self.nle + self.neq, self.m)] = np.zeros((self.nge, 1))
        # %%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons = np.where(np.isin(self.congrps, np.setdiff1d(self.congrps, nlc)))[
            0
        ]
        self.pbclass = "C-COLR2-AN-3-1"
        self.objderlvl = 2
        self.conderlvl = [2]

    # **********************
    #  SET UP THE FUNCTION *
    #  AND RANGE ROUTINES  *
    # **********************

    # %%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def ePROD3(self, nargout, *args):

        import numpy as np

        EV_ = args[0]
        iel_ = args[1]
        f_ = -EV_[0] * EV_[1] * EV_[2]
        if not isinstance(f_, float):
            f_ = f_.item()
        if nargout > 1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = np.zeros(dim)
            g_[0] = -EV_[1] * EV_[2]
            g_[1] = -EV_[0] * EV_[2]
            g_[2] = -EV_[1] * EV_[0]
            if nargout > 2:
                H_ = np.zeros((3, 3))
                H_[0, 1] = -EV_[2]
                H_[1, 0] = H_[0, 1]
                H_[0, 2] = -EV_[1]
                H_[2, 0] = H_[0, 2]
                H_[1, 2] = -EV_[0]
                H_[2, 1] = H_[1, 2]
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_, g_
        elif nargout == 3:
            return f_, g_, H_


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
